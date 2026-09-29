import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import weight_norm
import torchvision
from einops import rearrange
try:
    from backbone import Gaussian_block
    from arch_util_mine import CustomSequential, LayerNorm2d
except:
    from .backbone import Gaussian_block
    from .arch_util_mine import CustomSequential, LayerNorm2d


def normalize_per_sample(x, eps=1e-6):
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    B = x.shape[0]
    x_flat = x.reshape(B, -1)
    x_min = x_flat.min(dim=1)[0].view(B, 1, 1, 1)
    x_max = x_flat.max(dim=1)[0].view(B, 1, 1, 1)
    denom = (x_max - x_min).clamp_min(eps)
    return torch.nan_to_num((x - x_min) / denom, nan=0.0, posinf=1.0, neginf=0.0)

class CompressionUncertaintyPrior(nn.Module):
    def __init__(
        self,
        lambda_dct=0.5,
        lambda_lap=0.5,
        lambda_block=0.5,
        block_size=8,
        eps=1e-6,
        use_dct=True,
        use_lap=True,
        use_block=True,
    ):
        super().__init__()
        self.lambda_dct = lambda_dct
        self.lambda_lap = lambda_lap
        self.lambda_block = lambda_block
        self.block_size = block_size
        self.eps = eps
        self.use_dct = use_dct
        self.use_lap = use_lap
        self.use_block = use_block
        self.prior_fuse = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.register_buffer(
            "lap_kernel",
            torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3),
            persistent=False,
        )

    def _resize_map(self, x, size):
        if x.shape[-2:] == size:
            return x
        h, w = x.shape[-2:]
        target_h, target_w = size
        if h >= target_h and w >= target_w:
            return F.adaptive_avg_pool2d(x, size)
        return F.interpolate(x, size=size, mode="nearest")

    def _to_gray(self, img):
        img = torch.nan_to_num(img.float(), nan=0.0, posinf=1.0, neginf=0.0)
        if img.shape[1] == 1:
            return img
        return 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]

    def _lap_response(self, y, size):
        kernel = self.lap_kernel.to(device=y.device, dtype=y.dtype)
        u_lap = torch.abs(F.conv2d(y, kernel, padding=1))
        u_lap = normalize_per_sample(u_lap, self.eps)
        return self._resize_map(u_lap, size)

    def _block_response(self, y, size):
        dx = F.pad(torch.abs(y[:, :, :, 1:] - y[:, :, :, :-1]), (0, 1, 0, 0)) if y.shape[-1] > 1 else torch.zeros_like(y)
        dy = F.pad(torch.abs(y[:, :, 1:, :] - y[:, :, :-1, :]), (0, 0, 0, 1)) if y.shape[-2] > 1 else torch.zeros_like(y)
        grad = dx + dy
        local_mean = F.avg_pool2d(grad, kernel_size=7, stride=1, padding=3)
        u_block = torch.log1p(grad / (local_mean + self.eps))
        u_block = F.avg_pool2d(u_block, kernel_size=3, stride=1, padding=1)
        u_block = normalize_per_sample(u_block, self.eps)
        return self._resize_map(u_block, size)

    def _dct_response(self, y, size):
        y_safe = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
        low = F.avg_pool2d(y_safe, kernel_size=5, stride=1, padding=2)
        high = torch.abs(y_safe - low)
        local_energy = F.avg_pool2d(high, kernel_size=5, stride=1, padding=2)
        u_dct = normalize_per_sample(local_energy, self.eps)
        return self._resize_map(u_dct, size)

    def forward(self, feat, img):
        target_size = feat.shape[-2:]
        feat_safe = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        u_std = normalize_per_sample(torch.std(feat_safe.float(), dim=1, keepdim=True, unbiased=False).to(dtype=feat.dtype), self.eps)
        y = self._to_gray(img)

        if y.shape[-2:] != img.shape[-2:]:
            raise RuntimeError("Unexpected grayscale conversion shape mismatch.")

        zero = torch.zeros(feat.shape[0], 1, *target_size, device=feat.device, dtype=feat.dtype)
        u_lap = self._lap_response(y, target_size).to(device=feat.device, dtype=feat.dtype) if self.use_lap else zero
        u_block = self._block_response(y, target_size).to(device=feat.device, dtype=feat.dtype) if self.use_block else zero
        u_dct = self._dct_response(y, target_size).to(device=feat.device, dtype=feat.dtype) if self.use_dct else zero

        prior_stack = torch.cat([u_std, u_dct, u_block], dim=1)
        u_comp = self.prior_fuse(prior_stack)
        u_comp = normalize_per_sample(u_comp, self.eps)
        if not torch.isfinite(u_comp).all():
            print("[CompressionUncertaintyPrior] warning: non-finite u_comp detected; applying nan_to_num.")
            u_comp = torch.nan_to_num(u_comp, nan=0.0, posinf=1.0, neginf=0.0)

        return u_comp, {
            "u_std": u_std.detach(),
            "u_dct": u_dct.detach(),
            "u_lap": u_lap.detach(),
            "u_block": u_block.detach(),
            "u_comp": u_comp.detach(),
        }

class LocalDegradationQueryGate(nn.Module):
    def __init__(self, channels, hidden=None, eps=1e-6):
        super().__init__()
        hidden = max(channels // 4, 16) if hidden is None else hidden
        self.eps = eps
        self.prior_gate = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.query_norm = LayerNorm2d(channels)

    def _resize_map(self, x, size):
        if x.shape[-2:] == size:
            return x
        h, w = x.shape[-2:]
        target_h, target_w = size
        if h >= target_h and w >= target_w:
            return F.adaptive_avg_pool2d(x, size)
        return F.interpolate(x, size=size, mode="nearest")

    def forward(self, decoder_feature, u_sobel, u_lvar):
        H, W = decoder_feature.shape[-2:]
        dtype = decoder_feature.dtype
        device = decoder_feature.device

        u_sobel = self._resize_map(u_sobel, (H, W)).to(device=device, dtype=dtype)
        u_lvar = self._resize_map(u_lvar, (H, W)).to(device=device, dtype=dtype)
        dec_safe = torch.nan_to_num(decoder_feature, nan=0.0, posinf=0.0, neginf=0.0)
        u_cstd = torch.std(dec_safe.float(), dim=1, keepdim=True, unbiased=False).to(dtype=dtype)
        u_cstd = normalize_per_sample(u_cstd, self.eps)
        if u_cstd.shape[-2:] != (H, W):
            u_cstd = self._resize_map(u_cstd, (H, W))

        prior_stack = torch.cat([u_sobel, u_lvar, u_cstd], dim=1)
        gate = self.prior_gate(torch.nan_to_num(prior_stack, nan=0.0, posinf=1.0, neginf=0.0))
        gate = torch.clamp(torch.nan_to_num(gate, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)
        query = self.query_norm(decoder_feature * gate)
        return query, gate, {
            "query_u_sobel": u_sobel.detach(),
            "query_u_lvar": u_lvar.detach(),
            "query_u_cstd": u_cstd.detach(),
            "query_gate": gate.detach(),
        }

class CompressionAdaptiveSkipRecalibration(nn.Module):

    def __init__(
        self,
        channels,
        reduction=4,
        dilation=2,
        use_ucomp=True,
        gate_type="spatial",
        use_softmax_selection=False,
        debug=False,
        ucomp_mode="external",
        gate_mode="spatial",
        local_branch_dilation=4,
    ):
        super().__init__()
        if gate_type not in ("spatial", "channel"):
            raise ValueError("gate_type must be 'spatial' or 'channel'.")
        if ucomp_mode not in ("external", "dct_residual"):
            raise ValueError("ucomp_mode must be 'external' or 'dct_residual'.")
        if gate_mode not in ("spatial", "channel_spatial"):
            raise ValueError("gate_mode must be 'spatial' or 'channel_spatial'.")
        self.channels = channels
        self.use_ucomp = use_ucomp
        self.gate_type = gate_type
        self.use_softmax_selection = use_softmax_selection
        self.debug = debug
        self.ucomp_mode = ucomp_mode
        self.gate_mode = gate_mode
        self.local_branch_dilation = local_branch_dilation if local_branch_dilation is not None else dilation
        self.register_buffer("dct_basis", torch.empty(0), persistent=False)

        self.local_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GELU(),

            nn.Conv2d(
                channels,
                channels,
                3,
                padding=self.local_branch_dilation,
                dilation=self.local_branch_dilation,
                groups=channels,
            ),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        self.dec_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        if use_ucomp:
            self.comp_branch = nn.Sequential(
                nn.Conv2d(1, channels, 1),
                nn.GELU(),
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                nn.GELU(),
                nn.Conv2d(channels, channels, 1),
            )

        in_channels = channels * (3 if use_ucomp else 2)
        hidden = max(channels // reduction, 8)
        if use_softmax_selection:
            self.selector = nn.Sequential(
                nn.Conv2d(in_channels, hidden, 1),
                nn.GELU(),
                nn.Conv2d(hidden, 2, 1),
            )
        elif gate_mode == "channel_spatial":
            self.channel_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.Linear(in_channels, hidden),
                nn.GELU(),
                nn.Linear(hidden, channels),
                nn.Sigmoid(),
            )
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(in_channels, 1, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            out_gate_ch = 1 if gate_type == "spatial" else channels
            self.gate = nn.Sequential(
                nn.Conv2d(in_channels, hidden, 1),
                nn.GELU(),
                nn.Conv2d(hidden, hidden, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden, out_gate_ch, 1),
                nn.Sigmoid(),
            )
        self.out_proj = nn.Conv2d(channels, channels, 1)

    @staticmethod
    def _build_dct_basis():
        return torch.empty(0, dtype=torch.float32)

    @staticmethod
    def compute_ucomp_from_dct(f_skip, dct_basis=None):
        B, _, H, W = f_skip.shape
        x = f_skip.float().mean(dim=1, keepdim=True)
        low = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        residual = torch.abs(x - low)
        u = F.avg_pool2d(residual, kernel_size=5, stride=1, padding=2)
        u_flat = u.flatten(1)
        u_min = u_flat.min(dim=1).values.view(B, 1, 1, 1)
        u_max = u_flat.max(dim=1).values.view(B, 1, 1, 1)
        return (u - u_min) / (u_max - u_min + 1e-8)

    def _stats(self, name, x):
        x_detached = x.detach()
        return "{} shape={} mean={:.6f} std={:.6f} min={:.6f} max={:.6f}".format(
            name,
            tuple(x_detached.shape),
            float(x_detached.mean()),
            float(x_detached.std(unbiased=False)),
            float(x_detached.min()),
            float(x_detached.max()),
        )

    def forward(self, f_skip, f_dec, u_comp=None):
        if f_dec.shape[-2:] != f_skip.shape[-2:]:
            f_dec = F.interpolate(f_dec, size=f_skip.shape[-2:], mode="bilinear", align_corners=False)

        local_feat = self.local_branch(f_skip)
        dec_feat = self.dec_branch(f_dec)
        feats = [local_feat, dec_feat]

        if self.use_ucomp:
            if self.ucomp_mode == "dct_residual":
                u_comp = self.compute_ucomp_from_dct(f_skip, self.dct_basis).to(f_skip.dtype)
            if u_comp is None:
                raise ValueError("u_comp is required when CASR use_ucomp=True.")
            if u_comp.shape[-2:] != f_skip.shape[-2:]:
                u_comp = F.interpolate(u_comp, size=f_skip.shape[-2:], mode="bilinear", align_corners=False)
            feats.append(self.comp_branch(u_comp))

        fused = torch.cat(feats, dim=1)

        if self.use_softmax_selection:
            sw = torch.softmax(self.selector(fused), dim=1)
            sw_skip = sw[:, 0:1]
            sw_dec = sw[:, 1:2]
            f_fuse = sw_skip * f_skip + sw_dec * f_dec
            out = self.out_proj(f_fuse)
            gate_for_debug = sw_skip
            skip_recal = sw_skip * f_skip
        else:
            if self.gate_mode == "channel_spatial":
                B = fused.shape[0]
                cg = self.channel_gate(fused).view(B, self.channels, 1, 1)
                sg = self.spatial_gate(fused)
                gate = cg * sg
            else:
                gate = self.gate(fused)
            skip_recal = self.out_proj(gate * f_skip)
            out = f_dec + skip_recal
            gate_for_debug = gate

        if self.debug:
            if not torch.isfinite(gate_for_debug).all():
                print("[CASR] warning: non-finite gate detected.")
            if not torch.isfinite(skip_recal).all():
                print("[CASR] warning: non-finite skip_recal detected.")
            stats = [
                self._stats("f_skip", f_skip),
                self._stats("f_dec", f_dec),
                self._stats("u_comp", u_comp) if u_comp is not None else "u_comp=None",
                self._stats("gate", gate_for_debug),
                self._stats("skip_recal", skip_recal),
                self._stats("out", out),
            ]
            print("\n".join(["[CASR]"] + stats))
            assert out.shape == f_dec.shape,\
                f"[CASR] Output shape mismatch: {out.shape} vs {f_dec.shape}"

        return out

class Adaptive_Gated_Fusion(nn.Module):
    def __init__(self, in_dim, out_dim=None):
        super(Adaptive_Gated_Fusion, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim if out_dim is not None else in_dim
        hidden_dim = max(1, in_dim // 2)
        num_groups = min(8, in_dim)
        while in_dim % num_groups != 0 and num_groups > 1:
            num_groups -= 1

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_dim * 2, in_dim, kernel_size=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=in_dim) if in_dim > 1 else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim, kernel_size=1),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_gate = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_dim),
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_dim * 2, self.out_dim, kernel_size=1),
            nn.GELU()
        )

    def forward(self, f_enc, f_dec):
        if f_enc.shape[-2:] != f_dec.shape[-2:]:
            f_enc = F.interpolate(f_enc, size=f_dec.shape[-2:], mode="bilinear", align_corners=False)
        combined = torch.cat([f_enc, f_dec], dim=1)
        spatial_logit = self.spatial_gate(combined)

        b, c, _, _ = combined.shape
        y = self.avg_pool(combined).view(b, c)
        channel_logit = self.channel_gate(y).view(b, self.in_dim, 1, 1)

        atten_weight = torch.sigmoid(spatial_logit + channel_logit)
        f_enc_filtered = f_enc * atten_weight
        out = torch.cat([f_enc_filtered, f_dec], dim=1)
        return self.fusion_conv(out)

class CompressionAwareSkipCorrection(nn.Module):
    """Identity-initialized skip correction for compressed-image artifacts."""

    def __init__(
        self,
        channels,
        bottleneck=16,
        large_kernel=7,
        alpha_init=0.1,
        gamma_init=1.0,
        use_prior=True,
    ):
        super().__init__()
        self.channels = channels
        self.mid_channels = max(4, min(bottleneck, max(1, channels // 4)))
        self.large_kernel = max(3, int(large_kernel) | 1)
        self.use_prior = use_prior

        self.squeeze = nn.Conv2d(channels, self.mid_channels, kernel_size=1)
        self.local_dw = nn.Conv2d(
            self.mid_channels,
            self.mid_channels,
            kernel_size=3,
            padding=0,
            groups=self.mid_channels,
            bias=True,
        )
        self.bg_h = nn.Conv2d(
            self.mid_channels,
            self.mid_channels,
            kernel_size=(1, self.large_kernel),
            padding=0,
            groups=self.mid_channels,
            bias=False,
        )
        self.bg_v = nn.Conv2d(
            self.mid_channels,
            self.mid_channels,
            kernel_size=(self.large_kernel, 1),
            padding=0,
            groups=self.mid_channels,
            bias=False,
        )
        self.struct_mask = nn.Sequential(
            nn.Conv2d(self.mid_channels * 2 + 1, self.mid_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.delta_net = nn.Sequential(
            nn.Conv2d(self.mid_channels * 3, self.mid_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(
                self.mid_channels,
                self.mid_channels,
                kernel_size=3,
                padding=1,
                groups=self.mid_channels,
            ),
            nn.GELU(),
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=1),
        )
        self.prior_gate = nn.Sequential(
            nn.Conv2d(1, self.mid_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.expand = nn.Conv2d(self.mid_channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))
        alpha_init = min(max(float(alpha_init), 1e-4), 1.0 - 1e-4)
        self.alpha_raw = nn.Parameter(torch.tensor(math.log(alpha_init / (1.0 - alpha_init)), dtype=torch.float32))

        blur_1d = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=torch.float32)
        blur_2d = torch.outer(blur_1d, blur_1d)
        blur_2d = blur_2d / blur_2d.sum()
        self.register_buffer("blur_kernel", blur_2d.view(1, 1, 5, 5), persistent=False)

        self._init_filters()

    def _init_filters(self):
        nn.init.zeros_(self.local_dw.weight)
        nn.init.zeros_(self.local_dw.bias)
        with torch.no_grad():
            self.local_dw.weight[:, 0, 1, 1] = 1.0
            self.bg_h.weight.fill_(1.0 / float(self.large_kernel))
            self.bg_v.weight.fill_(1.0 / float(self.large_kernel))
        nn.init.zeros_(self.expand.weight)
        if self.expand.bias is not None:
            nn.init.zeros_(self.expand.bias)

    def _safe_pad(self, x, pad):
        if isinstance(pad, int):
            pad = (pad, pad, pad, pad)
        left, right, top, bottom = pad
        if max(left, right, top, bottom) <= 0:
            return x
        can_reflect = x.shape[-2] > max(top, bottom) and x.shape[-1] > max(left, right)
        mode = "reflect" if can_reflect else "replicate"
        return F.pad(x, pad, mode=mode)

    def _resize_map(self, x, size):
        if x is None:
            return None
        if x.shape[-2:] == size:
            return x
        h, w = x.shape[-2:]
        target_h, target_w = size
        if h >= target_h and w >= target_w:
            return F.adaptive_avg_pool2d(x, size)
        return F.interpolate(x, size=size, mode="nearest")

    def _fixed_blur(self, x):
        kernel = self.blur_kernel.to(device=x.device, dtype=x.dtype).repeat(x.shape[1], 1, 1, 1)
        return F.conv2d(self._safe_pad(x, 2), kernel, groups=x.shape[1])

    def _background(self, x):
        pad = self.large_kernel // 2
        x = self.bg_h(self._safe_pad(x, (pad, pad, 0, 0)))
        x = self.bg_v(self._safe_pad(x, (0, 0, pad, pad)))
        return x

    def forward(self, f_enc, u_prior=None, edge_y=None):
        e = self.squeeze(f_enc)
        ll = self._fixed_blur(e)
        hf = e - ll

        local_hf = self.local_dw(self._safe_pad(hf, 1))
        alpha = torch.sigmoid(self.alpha_raw).to(dtype=hf.dtype)
        anomaly = local_hf - alpha * self._background(hf)

        edge_y = self._resize_map(edge_y, hf.shape[-2:])
        if edge_y is None:
            edge_y = hf.new_zeros(hf.shape[0], 1, hf.shape[-2], hf.shape[-1])
        edge_y = normalize_per_sample(edge_y.to(device=hf.device, dtype=hf.dtype))

        mask = self.struct_mask(torch.cat([ll, torch.abs(hf), edge_y], dim=1))
        artifact = mask * anomaly
        delta_s = self.delta_net(torch.cat([hf, artifact, ll], dim=1))

        if self.use_prior:
            u_prior = self._resize_map(u_prior, hf.shape[-2:])
            if u_prior is None:
                u_prior = hf.new_zeros(hf.shape[0], 1, hf.shape[-2], hf.shape[-1])
            u_prior = torch.clamp(torch.nan_to_num(u_prior.to(device=hf.device, dtype=hf.dtype), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            gate = self.prior_gate(u_prior)
        else:
            gate = 1.0

        delta = self.expand(gate * torch.tanh(delta_s))
        return f_enc + self.gamma.to(dtype=f_enc.dtype) * delta

class CompressionAwareGatedFusion(nn.Module):
    """AGF-style skip fusion with interpretable compression-aware channel gating."""

    def __init__(
        self,
        in_dim,
        out_dim=None,
        use_prior=True,
        gamma_init=0.0,
        reduction=4,
        debug=False,
        spd_large_kernel=7,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim if out_dim is not None else in_dim
        self.use_prior = use_prior
        self.debug = debug

        num_groups = min(8, in_dim)
        while in_dim % num_groups != 0 and num_groups > 1:
            num_groups -= 1

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_dim * 2 + 1, in_dim, kernel_size=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=in_dim) if in_dim > 1 else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim, kernel_size=1),
        )

        self.enc_align = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.enc_norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_dim, affine=False) if in_dim > 1 else nn.Identity()
        self.dec_norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_dim, affine=False) if in_dim > 1 else nn.Identity()
        self.prior_gate = nn.Conv2d(1, 1, kernel_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        channel_hidden = max(in_dim // reduction, 8)
        self.sem_channel_gate = nn.Sequential(
            nn.Linear(in_dim * 2, channel_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(channel_hidden, in_dim),
        )
        self.diff_channel_gate = nn.Sequential(
            nn.Linear(in_dim * 2, channel_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(channel_hidden, in_dim),
        )
        self.comp_channel_gate = nn.Sequential(
            nn.Linear(2, channel_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(channel_hidden, in_dim),
        )
        self.sem_norm = nn.LayerNorm(in_dim)
        self.diff_norm = nn.LayerNorm(in_dim)
        self.comp_norm = nn.LayerNorm(in_dim)
        self.mix_gate = nn.Sequential(
            nn.Linear(2, channel_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(channel_hidden, 2),
        )
        self.skip_correction = CompressionAwareSkipCorrection(
            in_dim,
            bottleneck=16,
            large_kernel=spd_large_kernel,
            alpha_init=0.1,
            gamma_init=1.0,
            use_prior=use_prior,
        )
        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_dim * 2, self.out_dim, kernel_size=1),
            nn.GELU(),
        )

    def _resize_map(self, x, size):
        if x.shape[-2:] == size:
            return x
        h, w = x.shape[-2:]
        target_h, target_w = size
        if h >= target_h and w >= target_w:
            return F.adaptive_avg_pool2d(x, size)
        return F.interpolate(x, size=size, mode="nearest")

    def _prepare_prior(self, u_prior, ref):
        B, _, H, W = ref.shape
        if not self.use_prior or u_prior is None:
            return ref.new_zeros(B, 1, H, W)
        if u_prior.shape[-2:] != (H, W):
            u_prior = self._resize_map(u_prior, (H, W))
        return torch.clamp(torch.nan_to_num(u_prior, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    def _stats(self, name, x):
        x_detached = x.detach()
        return "{} shape={} mean={:.6f} std={:.6f} min={:.6f} max={:.6f}".format(
            name,
            tuple(x_detached.shape),
            float(x_detached.mean()),
            float(x_detached.std(unbiased=False)),
            float(x_detached.min()),
            float(x_detached.max()),
        )

    def forward(self, f_enc, f_dec, u_prior=None, edge_y=None):
        if f_dec.shape[-2:] != f_enc.shape[-2:]:
            f_dec = F.interpolate(f_dec, size=f_enc.shape[-2:], mode="bilinear", align_corners=False)

        u_prior = self._prepare_prior(u_prior, f_enc)
        f_enc = self.skip_correction(f_enc, u_prior=u_prior, edge_y=edge_y)
        spatial_input = torch.cat([f_enc, f_dec, u_prior], dim=1)
        spatial_logit = self.spatial_gate(spatial_input)

        semantic_desc = self.avg_pool(torch.cat([f_enc, f_dec], dim=1)).flatten(1)
        aligned_diff = torch.abs(self.enc_norm(self.enc_align(f_enc)) - self.dec_norm(f_dec))
        prior_weight = torch.sigmoid(self.prior_gate(u_prior))
        diff_comp = aligned_diff * prior_weight
        artifact_desc = torch.cat([
            self.avg_pool(diff_comp).flatten(1),
            self.max_pool(diff_comp).flatten(1),
        ], dim=1)
        prior_pool = torch.cat([
            self.avg_pool(u_prior).flatten(1),
            self.max_pool(u_prior).flatten(1),
        ], dim=1)

        sem_logit = self.sem_norm(self.sem_channel_gate(semantic_desc))
        diff_logit = self.diff_norm(self.diff_channel_gate(artifact_desc))
        comp_logit = self.comp_norm(self.comp_channel_gate(prior_pool))
        channel_logit = sem_logit - F.softplus(diff_logit) - F.softplus(comp_logit)
        channel_logit = channel_logit.view(f_enc.shape[0], self.in_dim, 1, 1)

        mix = F.softmax(self.mix_gate(prior_pool), dim=1).view(f_enc.shape[0], 2, 1, 1, 1)
        fused_logit = mix[:, 0] * spatial_logit + mix[:, 1] * channel_logit
        skip_weight = torch.exp(self.gamma.to(fused_logit.dtype) * torch.tanh(fused_logit))
        f_enc_filtered = f_enc * skip_weight
        out = self.fusion_conv(torch.cat([f_enc_filtered, f_dec], dim=1))

        if self.debug:
            stats = [
                self._stats("u_prior", u_prior),
                self._stats("aligned_diff", aligned_diff),
                self._stats("prior_weight", prior_weight),
                self._stats("diff_comp", diff_comp),
                self._stats("spatial_logit", spatial_logit),
                self._stats("channel_logit", channel_logit),
                self._stats("mix", mix),
                self._stats("fused_logit", fused_logit),
                self._stats("skip_weight", skip_weight),
                self._stats("out", out),
                "gamma={:.6f}".format(float(self.gamma.detach())),
            ]
            print("\n".join(["[CompressionAwareGatedFusion]"] + stats))
        return out

class FactorizedDynamicSkipWeighting(nn.Module):
    """Lightweight skip recalibration for compressed image restoration.

    This module is not a full dynamic convolution. It factorizes dynamic local
    weighting into a spatial neighborhood selector W_s and a channel-neighborhood
    modulator W_c. The goal is to suppress degraded encoder skip artifacts before
    they pass into the decoder, while keeping true local edges/textures available.
    """

    def __init__(
        self,
        in_dim,
        out_dim=None,
        k=3,
        use_prior=True,
        use_safety_gate=False,
        softmax_channel_kernel=False,
        use_abs_diff=True,
        gamma_init=0.0,
        reduction=4,
        debug=False,
        pad_mode="reflect",
    ):
        super().__init__()
        if k != 3:
            raise ValueError("FactorizedDynamicSkipWeighting currently supports k=3 only.")
        self.in_dim = in_dim
        self.out_dim = out_dim if out_dim is not None else in_dim
        self.k = k
        self.K = k * k
        self.use_prior = use_prior
        self.use_safety_gate = use_safety_gate
        self.softmax_channel_kernel = softmax_channel_kernel
        self.use_abs_diff = use_abs_diff
        self.debug = debug
        self.pad_mode = pad_mode
        self.pad = k // 2
        self.offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 0), (0, 1),
            (1, -1), (1, 0), (1, 1),
        ]

        self.proj_enc = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.proj_dec = nn.Conv2d(in_dim, in_dim, kernel_size=1)

        guide_dim = in_dim * 2 + (in_dim if use_abs_diff else 0) + (1 if use_prior else 0)
        hidden_dim = max(in_dim // 2, 16)
        num_groups = min(8, hidden_dim)
        while hidden_dim % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.spatial_weight_branch = nn.Sequential(
            nn.Conv2d(guide_dim, hidden_dim, kernel_size=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, self.K, kernel_size=1),
        )

        mlp_hidden = max(in_dim // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_kernel_mlp = nn.Sequential(
            nn.Linear(in_dim * 2, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, in_dim * self.K),
        )
        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))
        self._init_channel_kernel_identity()

        agf_hidden = max(1, in_dim // 2)
        agf_groups = min(8, in_dim)
        while in_dim % agf_groups != 0 and agf_groups > 1:
            agf_groups -= 1
        self.agf_spatial_gate = nn.Sequential(
            nn.Conv2d(in_dim * 2, in_dim, kernel_size=1),
            nn.GroupNorm(num_groups=agf_groups, num_channels=in_dim) if in_dim > 1 else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim, kernel_size=1),
        )
        self.agf_channel_gate = nn.Sequential(
            nn.Linear(in_dim * 2, agf_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(agf_hidden, in_dim),
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_dim * 2, in_dim, kernel_size=1),
            nn.GELU(),
        )

        if use_safety_gate:
            gate_dim = in_dim * 2 + (in_dim if use_abs_diff else 0)
            self.safety_gate = nn.Sequential(
                nn.Conv2d(gate_dim, hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, 1, kernel_size=1),
            )
            self.lambda_prior_raw = nn.Parameter(torch.tensor(0.1))
        else:
            self.safety_gate = None
            self.lambda_prior_raw = None

    def _init_channel_kernel_identity(self):
        final = self.channel_kernel_mlp[-1]
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, -4.0)
        with torch.no_grad():
            bias = final.bias.view(self.in_dim, self.K)
            bias[:, self.K // 2] = 4.0

    def _stats(self, name, x):
        x_detached = x.detach()
        return "{} shape={} mean={:.6f} std={:.6f} min={:.6f} max={:.6f}".format(
            name,
            tuple(x_detached.shape),
            float(x_detached.mean()),
            float(x_detached.std(unbiased=False)),
            float(x_detached.min()),
            float(x_detached.max()),
        )

    def _safe_pad(self, x):
        if self.pad <= 0:
            return x
        can_reflect = x.shape[-2] > self.pad and x.shape[-1] > self.pad
        mode = self.pad_mode if self.pad_mode == "reflect" and can_reflect else "replicate"
        return F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode=mode)

    def get_shifted_features(self, x):
        B, C, H, W = x.shape
        padded = self._safe_pad(x)
        shifted = []
        for dy, dx in self.offsets:
            y0 = self.pad + dy
            x0 = self.pad + dx
            shifted.append(padded[:, :, y0:y0 + H, x0:x0 + W])
        return torch.stack(shifted, dim=2)

    def forward(self, f_enc, f_dec, u_prior=None):
        if f_dec.shape[-2:] != f_enc.shape[-2:]:
            f_dec = F.interpolate(f_dec, size=f_enc.shape[-2:], mode="bilinear", align_corners=False)
        E = self.proj_enc(f_enc)
        D = self.proj_dec(f_dec)
        H, W = E.shape[-2:]

        if self.use_prior and u_prior is not None:
            if u_prior.shape[-2:] != (H, W):
                u_prior = F.interpolate(u_prior, size=(H, W), mode="bilinear", align_corners=False)
            u_prior = torch.nan_to_num(u_prior, nan=0.0, posinf=1.0, neginf=0.0)
        elif self.use_prior:
            u_prior = E.new_zeros(E.shape[0], 1, H, W)
        else:
            u_prior = None

        guide_parts = [E, D]
        if self.use_abs_diff:
            guide_parts.append(torch.abs(E - D))
        if self.use_prior:
            guide_parts.append(u_prior)
        guide = torch.cat(guide_parts, dim=1)

        spatial_logits = self.spatial_weight_branch(guide)
        center_idx = self.K // 2
        center_bias = torch.zeros(
            1, self.K, 1, 1,
            device=spatial_logits.device,
            dtype=spatial_logits.dtype,
        )
        center_bias[:, center_idx:center_idx + 1, :, :] = 5.0
        spatial_logits = spatial_logits + center_bias
        W_s = F.softmax(spatial_logits, dim=1)

        context = self.avg_pool(torch.cat([E, D], dim=1)).flatten(1)
        channel_logits = self.channel_kernel_mlp(context).view(E.shape[0], self.in_dim, self.K, 1, 1)
        if self.softmax_channel_kernel:
            W_c = torch.softmax(channel_logits, dim=2)
        else:
            W_c = torch.sigmoid(channel_logits)

        shifted = self.get_shifted_features(E)
        E_dyn = (shifted * W_s.unsqueeze(1) * W_c).sum(dim=2)
        E_dyn = E + self.gamma.to(dtype=E.dtype) * (E_dyn - E)

        if self.use_safety_gate and self.safety_gate is not None:
            gate_parts = [E, D]
            if self.use_abs_diff:
                gate_parts.append(torch.abs(E - D))
            base_gate_logit = self.safety_gate(torch.cat(gate_parts, dim=1))
            if u_prior is not None:
                lambda_prior = F.softplus(self.lambda_prior_raw)
                gate = torch.sigmoid(base_gate_logit - lambda_prior * u_prior)
            else:
                gate = torch.sigmoid(base_gate_logit)
            E_dyn = gate * E_dyn

        agf_combined = torch.cat([E_dyn, D], dim=1)
        agf_spatial_logit = self.agf_spatial_gate(agf_combined)
        agf_context = self.avg_pool(agf_combined).flatten(1)
        agf_channel_logit = self.agf_channel_gate(agf_context).view(E.shape[0], self.in_dim, 1, 1)
        agf_weight = torch.sigmoid(agf_spatial_logit + agf_channel_logit)
        E_fused = E_dyn * agf_weight
        out = self.fusion_conv(torch.cat([E_fused, f_dec], dim=1))

        if self.debug:
            stats = [
                self._stats("W_s", W_s),
                self._stats("W_c", W_c),
                self._stats("E_dyn", E_dyn),
                self._stats("agf_weight", agf_weight),
                self._stats("E_fused", E_fused),
                self._stats("out", out),
                "gamma={:.6f}".format(float(self.gamma.detach())),
            ]
            print("\n".join(["[FactorizedDynamicSkipWeighting]"] + stats))
        return out

class DynamicEncoderContextBank(nn.Module):
    def __init__(
        self,
        in_channels_list,
        bank_dim=64,
        bank_size=16,
        use_ucomp_filter=False,
        debug=False,
    ):
        super().__init__()
        if len(in_channels_list) != 3:
            raise ValueError("DynamicEncoderContextBank expects exactly three encoder channel sizes.")
        self.bank_dim = bank_dim
        self.bank_size = bank_size
        self.use_ucomp_filter = use_ucomp_filter
        self.debug = debug
        self.proj_layers = nn.ModuleList([nn.Conv2d(ch, bank_dim, 1) for ch in in_channels_list])
        self.token_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.k_proj = nn.Linear(bank_dim, bank_dim)
        self.v_proj = nn.Linear(bank_dim, bank_dim)

    def _stats(self, name, x):
        x_detached = x.detach()
        return "{} shape={} mean={:.6f} std={:.6f} min={:.6f} max={:.6f}".format(
            name,
            tuple(x_detached.shape),
            float(x_detached.mean()),
            float(x_detached.std(unbiased=False)),
            float(x_detached.min()),
            float(x_detached.max()),
        )

    def forward(self, encoder_feats, u_comp=None):
        if len(encoder_feats) != 3:
            raise ValueError("DynamicEncoderContextBank forward expects three encoder features.")
        token_list = []
        for idx, feat in enumerate(encoder_feats):
            feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
            feat = self.proj_layers[idx](feat)
            if self.use_ucomp_filter and u_comp is not None:
                u = F.interpolate(u_comp, size=feat.shape[-2:], mode="bilinear", align_corners=False)
                reliability = 1.0 - torch.clamp(torch.nan_to_num(u, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
                feat = feat * reliability
            pooled = self.token_pool(feat)
            tokens = pooled.flatten(2).transpose(1, 2).contiguous()
            token_list.append(tokens)

        x = torch.cat(token_list, dim=1)
        k_bank = torch.nan_to_num(self.k_proj(x), nan=0.0, posinf=0.0, neginf=0.0)
        v_bank = torch.nan_to_num(self.v_proj(x), nan=0.0, posinf=0.0, neginf=0.0)
        if self.debug:
            lines = ["[DynamicEncoderContextBank]"]
            lines.extend(self._stats("encoder_feat_{}".format(i), f) for i, f in enumerate(encoder_feats))
            lines.append(self._stats("k_bank", k_bank))
            lines.append(self._stats("v_bank", v_bank))
            if not torch.isfinite(k_bank).all():
                lines.append("warning: non-finite k_bank detected.")
            if not torch.isfinite(v_bank).all():
                lines.append("warning: non-finite v_bank detected.")
            print("\n".join(lines))
        return k_bank, v_bank

class FixedHaarDWT2D(nn.Module):
    def __init__(self):
        super().__init__()
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        lo = torch.tensor([inv_sqrt2, inv_sqrt2], dtype=torch.float32)
        hi = torch.tensor([-inv_sqrt2, inv_sqrt2], dtype=torch.float32)
        ll = torch.outer(lo, lo)
        lh = torch.outer(hi, lo)
        hl = torch.outer(lo, hi)
        hh = torch.outer(hi, hi)
        kernel = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer("kernel", kernel, persistent=False)

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        _, c, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        kernel = self.kernel.to(device=x.device, dtype=x.dtype).repeat(c, 1, 1, 1)
        return F.conv2d(x, kernel, stride=2, groups=c)

class WaveletResidualReduce(nn.Module):
    def __init__(self, in_channels, out_channels, gamma_init=0.1):
        super().__init__()
        groups = 8 if out_channels % 8 == 0 else 1
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.body = nn.Sequential(
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels, bias=True),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))

    def forward(self, x):
        y = self.proj(torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
        return y + self.gamma.to(y.dtype) * self.body(y)

class ContextualCodebookBank(nn.Module):
    def __init__(
        self,
        in_channels_list,
        bank_dim=64,
        pool_size=8,
        debug=False,
    ):
        super().__init__()
        self.bank_dim = bank_dim
        self.pool_size = pool_size
        self.debug = debug
        if len(in_channels_list) != 4:
            raise ValueError("ContextualCodebookBank expects four encoder stages.")
        c1, c2, c3, c4 = in_channels_list
        self.dwt12 = FixedHaarDWT2D()
        self.dwt23 = FixedHaarDWT2D()
        dwt_channels = max(bank_dim // 4, 16)
        dwt_out_channels = dwt_channels * 4
        c1_groups = 8 if c1 % 8 == 0 else 1
        c2_groups = 8 if c2 % 8 == 0 else 1
        c3_groups = 8 if c3 % 8 == 0 else 1
        self.s1_pre = nn.Sequential(
            nn.GroupNorm(c1_groups, c1),
            nn.GELU(),
            nn.Conv2d(c1, dwt_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.s2_norm = nn.GroupNorm(c2_groups, c2)
        self.s3_norm = nn.GroupNorm(c3_groups, c3)
        self.s2_fuse_pre = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(bank_dim + c2, dwt_channels, kernel_size=3, stride=1, padding=1, bias=True),
        )
        self.s3_fuse = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(bank_dim + c3, bank_dim, kernel_size=3, stride=1, padding=1, bias=True),
        )
        self.reduce12 = (
            nn.Identity() if dwt_out_channels == bank_dim
            else WaveletResidualReduce(dwt_out_channels, bank_dim)
        )
        self.reduce23 = (
            nn.Identity() if dwt_out_channels == bank_dim
            else WaveletResidualReduce(dwt_out_channels, bank_dim)
        )
        self.token_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))

    def _stats(self, name, x):
        x_detached = x.detach()
        return "{} shape={} mean={:.6f} std={:.6f} min={:.6f} max={:.6f}".format(
            name,
            tuple(x_detached.shape),
            float(x_detached.mean()),
            float(x_detached.std(unbiased=False)),
            float(x_detached.min()),
            float(x_detached.max()),
        )

    def forward(self, encoder_feats):
        if len(encoder_feats) != 4:
            raise ValueError("ContextualCodebookBank expected 4 encoder features, got {}.".format(
                len(encoder_feats)
            ))
        s1, s2, s3, _ = [torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0) for feat in encoder_feats]

        w1 = self.reduce12(self.dwt12(self.s1_pre(s1)))
        s2_n = self.s2_norm(s2)
        w2_pre = self.s2_fuse_pre(torch.cat([w1, s2_n], dim=1))
        w2 = self.reduce23(self.dwt23(w2_pre))
        s3_n = self.s3_norm(s3)
        bank_feat = self.s3_fuse(torch.cat([w2, s3_n], dim=1))

        bank_feat = self.token_pool(bank_feat)
        tokens = bank_feat.flatten(2).transpose(1, 2).contiguous()
        tokens = torch.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0)

        if self.debug:
            lines = ["[ContextualCodebookBank]"]
            lines.extend(self._stats("encoder_feat_{}".format(i), f) for i, f in enumerate(encoder_feats))
            lines.append(self._stats("bank_feat_w1", w1))
            lines.append(self._stats("bank_feat_w2", w2))
            lines.append(self._stats("bank_feat", bank_feat))
            lines.append(self._stats("E_global", tokens))
            if not torch.isfinite(tokens).all():
                lines.append("warning: non-finite E_global detected.")
            print("\n".join(lines))
        return tokens

class UncertaintyGuidedBankAttention(nn.Module):
    def __init__(
        self,
        dec_channels,
        bank_dim=64,
        attn_dim=64,
        bank_size=16,
        gamma_init=0.0,
        debug=False,
    ):
        super().__init__()
        self.attn_dim = attn_dim
        self.bank_size = bank_size
        self.debug = debug
        self.q_proj = nn.Sequential(
            nn.Conv2d(dec_channels + 1, attn_dim, 1),
            nn.GELU(),
            nn.Conv2d(attn_dim, attn_dim, 1),
        )
        self.k_proj = nn.Linear(bank_dim, attn_dim)
        self.v_proj = nn.Linear(bank_dim, attn_dim)
        self.out_proj = nn.Conv2d(attn_dim, dec_channels, 1)
        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))

    def _stats(self, name, x):
        x_detached = x.detach()
        return "{} shape={} mean={:.6f} std={:.6f} min={:.6f} max={:.6f}".format(
            name,
            tuple(x_detached.shape),
            float(x_detached.mean()),
            float(x_detached.std(unbiased=False)),
            float(x_detached.min()),
            float(x_detached.max()),
        )

    def forward(self, dec_feat, uncertainty, k_bank, v_bank):
        if k_bank is None or v_bank is None:
            return dec_feat
        dec_feat = torch.nan_to_num(dec_feat, nan=0.0, posinf=0.0, neginf=0.0)
        if uncertainty is None:
            uncertainty = dec_feat.new_zeros(dec_feat.shape[0], 1, dec_feat.shape[-2], dec_feat.shape[-1])
        if uncertainty.shape[1] != 1:
            uncertainty = uncertainty.mean(dim=1, keepdim=True)
        uncertainty = normalize_per_sample(uncertainty)

        if uncertainty.shape[-2:] != dec_feat.shape[-2:]:
            raise RuntimeError("UncertaintyGuidedBankAttention expects uncertainty to match decoder feature size.")
        q = self.q_proj(torch.cat([dec_feat, uncertainty.to(dtype=dec_feat.dtype)], dim=1))
        k = self.k_proj(torch.nan_to_num(k_bank, nan=0.0, posinf=0.0, neginf=0.0))
        v = self.v_proj(torch.nan_to_num(v_bank, nan=0.0, posinf=0.0, neginf=0.0))

        B, Cq, Hq, Wq = q.shape
        q_flat = q.flatten(2).transpose(1, 2)
        attn = torch.bmm(q_flat.float(), k.float().transpose(1, 2)) / math.sqrt(float(self.attn_dim))
        attn = torch.softmax(torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0), dim=-1).to(dtype=q.dtype)
        context = torch.bmm(attn, v.to(dtype=attn.dtype))
        context = context.transpose(1, 2).reshape(B, Cq, Hq, Wq)
        context = self.out_proj(context)
        uncertainty = torch.clamp(torch.nan_to_num(uncertainty, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        context = torch.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
        out = dec_feat + self.gamma.to(dtype=dec_feat.dtype) * uncertainty.to(dtype=dec_feat.dtype) * context
        if self.debug:
            lines = [
                "[UncertaintyGuidedBankAttention]",
                self._stats("dec_feat", dec_feat),
                self._stats("uncertainty", uncertainty),
                self._stats("attn", attn),
                self._stats("context", context),
                "gamma={:.6f}".format(float(self.gamma.detach())),
            ]
            if not torch.isfinite(context).all():
                lines.append("warning: non-finite context detected.")
            print("\n".join(lines))
        return out

class _Memory_Block(nn.Module):
    def __init__(self, hdim, kdim, moving_average_rate=0.999):
        super().__init__()

        self.c = hdim
        self.k = kdim

        self.moving_average_rate = moving_average_rate

        self.units = nn.Parameter(torch.rand(kdim, hdim), requires_grad=True)

    def update(self, x, score, m=None):
        '''
            x: (n, c)
            e: (k, c)
            score: (n, k)
        '''

        m = self.units
        x = x.detach()
        embed_ind = torch.max(score, dim=1)[1]
        embed_onehot = F.one_hot(embed_ind, self.k).type(x.dtype)
        embed_onehot_sum = embed_onehot.sum(0)
        embed_sum = x.transpose(0, 1) @ embed_onehot
        embed_mean = embed_sum / (embed_onehot_sum + 1e-6)
        new_data = m * self.moving_average_rate + embed_mean.t() * (1 - self.moving_average_rate)
        if self.training:

            self.units = nn.Parameter(new_data)
        return new_data

    def forward(self, x, update_flag=True):
        '''
          x: (b, c, h, w)
          embed: (k, c)
        '''

        b, c, h, w = x.size()
        assert c == self.c
        k, c = self.k, self.c

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, c)

        m = self.units

        xn = F.normalize(x, dim=1)
        mn = F.normalize(m, dim=1)
        score = torch.matmul(xn, mn.t())

        if update_flag:
            m = self.update(x, score, m)
            mn = F.normalize(m, dim=1)
            score = torch.matmul(xn, mn.t())

        soft_label = F.softmax(score, dim=1)
        out = torch.matmul(soft_label, m)
        out = out.view(b, h, w, c).permute(0, 3, 1, 2)

        return out, score

class _Memory_Block_prompt(nn.Module):
    def __init__(self, hdim, kdim, moving_average_rate=0.999):
        super().__init__()
        self.c = hdim
        self.k = kdim

        self.moving_average_rate = moving_average_rate
        self.units = nn.Parameter(torch.randn(kdim, hdim), requires_grad=True)
        self.prompts = nn.Parameter(torch.rand(kdim, hdim), requires_grad=True)

    def update(self, x, score, m=None):
        '''
            x: (n, c)
            e: (k, c)
            score: (n, k)
        '''

        m = self.units
        x = x.detach()
        embed_ind = torch.max(score, dim=1)[1]
        embed_onehot = F.one_hot(embed_ind, self.k).type(x.dtype)
        embed_onehot_sum = embed_onehot.sum(0)
        embed_sum = x.transpose(0, 1) @ embed_onehot
        embed_mean = embed_sum / (embed_onehot_sum + 1e-6)
        new_data = m * self.moving_average_rate + embed_mean.t() * (1 - self.moving_average_rate)
        if self.training:

            self.units = nn.Parameter(new_data)
        return new_data

    def forward(self, x, update_flag=True):
        '''
          x: (b, c, h, w)
          embed: (k, c)
        '''

        b, c, h, w = x.size()
        assert c == self.c
        k, c = self.k, self.c

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, c)

        m = self.units
        p = self.prompts

        xn = F.normalize(x, dim=1)
        mn = F.normalize(m, dim=1)
        score = torch.matmul(xn, mn.t())

        if update_flag:
            m = self.update(x, score, m)
            mn = F.normalize(m, dim=1)
            score = torch.matmul(xn, mn.t())

        soft_label = F.softmax(score, dim=1)

        out_m = torch.matmul(soft_label, m)
        out_m = out_m.view(b, h, w, c).permute(0, 3, 1, 2)
        out = torch.matmul(soft_label, p)
        out = out.view(b, h, w, c).permute(0, 3, 1, 2)
        soft_label = soft_label.view(b, h, w, -1).permute(0, 3, 1, 2)

        return out, out_m, soft_label

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),)

    def forward(self, x):
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class IG_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.dim = dim

    def forward(self, x, var):
        """
        x_in: [b,h,w,c]
        illu_fea: [b,h,w,c]
        return out: [b,h,w,c]
        """

        b, h, w, c = x.shape

        x = x

        x = x.reshape(b, h * w, c)
        var = var.reshape(b, h * w, c)

        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q_inp, k_inp, v_inp))

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out = out_c
        return out

class ContextRoutedAttention(nn.Module):
    def __init__(
        self,
        dim,
        bank_dim=None,
        dim_head=64,
        heads=8,
        down_kernel=3,
        refine_act=nn.ReLU,
    ):
        super().__init__()
        self.num = 256
        self.bank_dim = dim if bank_dim is None else bank_dim
        self.memory_key_proj = nn.Linear(self.bank_dim, dim)
        self.memory_prompt_proj = nn.Linear(self.bank_dim, dim)

        self.num_heads = heads
        self.dim_head = dim_head
        self.dim = dim

        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)

        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))

        self.proj = nn.Linear(dim_head * heads, dim, bias=True)

        self.down = nn.Conv2d(dim, dim, kernel_size=down_kernel, stride=2, padding=down_kernel//2, bias=False)
        self.up   = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2, bias=False)

        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            refine_act(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            refine_act(inplace=True),
        )

        self.act = nn.GELU()
        self.alpha = nn.Parameter(torch.zeros((1, 1, dim, 1)), requires_grad=True)

    def retrieve_dynamic_prompt(self, var_bhwc, e_global):
        """
        var_bhwc: [B, H, W, C]
        e_global: [B, N, bank_dim]
        return:
            z: [B, H, W, C]
            soft_label: [B, N, H, W]
        """
        B, H, W, C = var_bhwc.shape
        if e_global is None:
            raise ValueError("ContextRoutedAttention requires E_global from ContextualCodebookBank.")
        if e_global.shape[0] != B:
            raise ValueError("E_global batch size {} does not match var batch size {}.".format(e_global.shape[0], B))

        e_global = torch.nan_to_num(e_global, nan=0.0, posinf=0.0, neginf=0.0)
        e_m = self.memory_key_proj(e_global)
        e_p = self.memory_prompt_proj(e_global)

        q = var_bhwc.reshape(B, H * W, C)
        q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(e_m, dim=-1, p=2)
        score = torch.matmul(q, k.transpose(-2, -1))
        soft_label = F.softmax(score, dim=-1)
        z = torch.matmul(soft_label, e_p).view(B, H, W, C)
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        soft_label = soft_label.transpose(1, 2).view(B, e_global.shape[1], H, W)
        return z, soft_label

    def dual_prompt_attention_lowres(self, x_lr_bhwc, z_lr_bhwc, z_m_lr_bhwc):
        """
        x_lr_bhwc, z_lr_bhwc, z_m_lr_bhwc: [B, Hlr, Wlr, C]
        """
        B, Hlr, Wlr, C = x_lr_bhwc.shape

        q_h = self.to_q(x_lr_bhwc.reshape(B*Hlr, Wlr, C))
        k_h = self.to_k(z_lr_bhwc.reshape(B*Hlr, Wlr, C))
        v_h = self.to_v(z_m_lr_bhwc.reshape(B*Hlr, Wlr, C))

        q_h, k_h, v_h = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q_h, k_h, v_h))
        q_h = F.normalize(q_h, dim=-1); k_h = F.normalize(k_h, dim=-1)
        attn_h = torch.matmul(q_h, k_h.transpose(-2, -1)) * self.rescale
        attn_h = attn_h.softmax(dim=-1)
        out_h  = torch.matmul(attn_h, v_h)
        out_h  = rearrange(out_h, 'b h n d -> b n (h d)').reshape(B, Hlr, Wlr, C)

        q_v = self.to_q(x_lr_bhwc.permute(0, 2, 1, 3).reshape(B*Wlr, Hlr, C))
        k_v = self.to_k(z_lr_bhwc.permute(0, 2, 1, 3).reshape(B*Wlr, Hlr, C))
        v_v = self.to_v(z_m_lr_bhwc.permute(0, 2, 1, 3).reshape(B*Wlr, Hlr, C))

        q_v, k_v, v_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q_v, k_v, v_v))
        q_v = F.normalize(q_v, dim=-1); k_v = F.normalize(k_v, dim=-1)
        attn_v = torch.matmul(q_v, k_v.transpose(-2, -1)) * self.rescale
        attn_v = attn_v.softmax(dim=-1)
        out_v  = torch.matmul(attn_v, v_v)
        out_v  = rearrange(out_v, 'b h n d -> b n (h d)').reshape(B, Wlr, Hlr, C).permute(0, 2, 1, 3)

        out_lr = (out_h + out_v) * 0.5

        out_lr = self.proj(out_lr)
        return out_lr

    def forward(self, x, var, e_global):
        """
        x:   [B, H, W, C]
        var: [B, H, W, C]
        e_global: [B, N, bank_dim]
        return: [B, H, W, C]
        """
        z, soft_label = self.retrieve_dynamic_prompt(var, e_global)

        B, H, W, C = x.shape

        x_chw   = x.permute(0, 3, 1, 2)
        x_lr    = self.down(x_chw)
        Hlr, Wlr = x_lr.shape[2], x_lr.shape[3]

        z_lr   = F.interpolate(z.permute(0, 3, 1, 2), size=(Hlr, Wlr), mode='bilinear', align_corners=False)

        x_lr_bhwc   = x_lr.permute(0, 2, 3, 1)
        z_lr_bhwc   = z_lr.permute(0, 2, 3, 1)

        out_lr_bhwc = self.dual_prompt_attention_lowres(x_lr_bhwc, z_lr_bhwc, x_lr_bhwc)

        out_lr_chw = out_lr_bhwc.permute(0, 3, 1, 2)
        out_hr_chw = self.up(out_lr_chw)[:, :, :H, :W]

        out_chw = self.refine(out_hr_chw + x_chw)
        out = out_chw.permute(0, 2, 3, 1)

        return out, soft_label

class UncertaintyGuidedBankCrossAttention(nn.Module):
    """Full-resolution cross-attention from a degradation-conditioned query to the encoder Bank."""

    def __init__(
        self,
        dim,
        bank_dim=None,
        dim_head=64,
        heads=8,
        query_chunk_size=2048,
        gamma_init=0.0,
        return_scoremap=False,
    ):
        super().__init__()
        self.bank_dim = dim if bank_dim is None else bank_dim
        self.num_heads = heads
        self.dim_head = dim_head
        self.inner_dim = dim_head * heads
        self.query_chunk_size = max(1, int(query_chunk_size))
        self.return_scoremap = return_scoremap

        self.query_proj = nn.Linear(dim, self.inner_dim, bias=False)
        self.bank_norm = nn.LayerNorm(self.bank_dim)
        self.key_proj = nn.Linear(self.bank_dim, self.inner_dim, bias=False)
        self.value_proj = nn.Linear(self.bank_dim, self.inner_dim, bias=False)
        self.out_proj = nn.Linear(self.inner_dim, dim, bias=True)
        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))

    def forward(self, x, query_feature, e_global, query_gate):
        """
        x, query_feature, query_gate: [B, H, W, C]
        e_global: [B, N, bank_dim]
        """
        B, H, W, C = x.shape
        if e_global is None:
            raise ValueError("Bank cross-attention requires E_global.")
        if e_global.shape[0] != B:
            raise ValueError("E_global and decoder batch sizes do not match.")

        query_feature = torch.nan_to_num(query_feature, nan=0.0, posinf=0.0, neginf=0.0)
        bank = self.bank_norm(torch.nan_to_num(e_global, nan=0.0, posinf=0.0, neginf=0.0))
        q = self.query_proj(query_feature.reshape(B, H * W, C))
        k = self.key_proj(bank)
        v = self.value_proj(bank)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
            (q, k, v),
        )

        scale = self.dim_head ** -0.5
        k_t = k.transpose(-2, -1)
        output_chunks = []
        scoremap_chunks = [] if self.return_scoremap else None
        for start in range(0, H * W, self.query_chunk_size):
            q_chunk = q[:, :, start:start + self.query_chunk_size]
            score = torch.matmul(q_chunk, k_t) * scale
            attn = F.softmax(score, dim=-1)
            output_chunks.append(torch.matmul(attn, v))
            if scoremap_chunks is not None:
                scoremap_chunks.append(attn.mean(dim=1).detach())

        context = torch.cat(output_chunks, dim=2)
        context = rearrange(context, 'b h n d -> b n (h d)')
        context = self.out_proj(context).view(B, H, W, C)
        gate = torch.clamp(
            torch.nan_to_num(query_gate, nan=0.5, posinf=1.0, neginf=0.0),
            0.0,
            1.0,
        )
        delta = self.gamma.to(context.dtype) * gate * context

        scoremap = None
        if scoremap_chunks is not None:
            scoremap = torch.cat(scoremap_chunks, dim=1)
            scoremap = scoremap.transpose(1, 2).reshape(B, e_global.shape[1], H, W)
        return delta, scoremap

class IGAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, var):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        var = var.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x, var) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

class SpatialChannelInteraction(nn.Module):
    def __init__(
            self,
            dim,
            bank_dim=None,
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                ContextRoutedAttention(dim=dim, bank_dim=bank_dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))
            self.blocks.append(nn.ModuleList([
                IG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))
    def forward(self, x, query_feature, e_global):
        """
        x: [b,c,h,w]
        query_feature: [b,c,h,w]
        e_global: [b,n,c_bank]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        query_feature = query_feature.permute(0, 2, 3, 1)
        i = 0
        for (attn, ff) in self.blocks:
            if i==0:
                x_, s = attn(x, query_feature, e_global)
                x = x_ + x
                x = ff(x) + x
            else:
                x = attn(x, query_feature) + x
                x = ff(x) + x
            i += 1
        out = x.permute(0, 3, 1, 2)
        return out, s

class CACRNet(nn.Module):

    def __init__(self, img_channel=3,
                 width=32,
                 middle_blk_num_enc=2,
                 middle_blk_num_dec=2,
                 enc_blk_nums=[1, 2, 3],
                 dec_blk_nums=[3, 1, 1],
                 extra_depth_wise = True,
                 use_comp_uncertainty=False,
                 comp_uncertainty_mode="fuse",
                 lambda_dct=0.5,
                 lambda_lap=0.5,
                 lambda_block=0.5,
                 comp_uncertainty_gamma=0.5,
                 block_size=8,
                 comp_use_dct=True,
                 comp_use_lap=False,
                 comp_use_block=True,
                 debug_uncertainty=False,
                 uncertainty_debug_interval=200,
                 skip_fusion_mode="auto",
                 use_casr=False,
                 casr_use_ucomp=True,
                 casr_gate_type="spatial",
                 casr_use_softmax_selection=False,
                 casr_dilation=None,
                 casr_ucomp_mode="external",
                 casr_gate_mode="spatial",
                 casr_local_dilation=4,
                 casr_reduction=4,
                 casr_apply_layers="all",
                 casr_debug=False,
                 factorized_skip_k=3,
                 factorized_skip_use_prior=True,
                 factorized_skip_use_safety_gate=False,
                 factorized_skip_softmax_channel_kernel=False,
                 factorized_skip_use_abs_diff=True,
                 factorized_skip_gamma_init=0.0,
                 factorized_skip_debug=False,
                 use_dynamic_encoder_bank=False,
                 dynamic_bank_dim=64,
                 dynamic_bank_size=16,
                 dynamic_bank_attn_dim=64,
                 dynamic_bank_apply_layers="all",
                 dynamic_bank_use_ucomp_filter=False,
                 dynamic_bank_gamma_init=0.0,
                 dynamic_bank_encoder_layers="0,1,2",
                 dynamic_bank_debug=False):
        super(CACRNet, self).__init__()
        if comp_uncertainty_mode not in ("fuse", "replace"):
            raise ValueError("comp_uncertainty_mode must be 'fuse' or 'replace'.")
        self.use_comp_uncertainty = use_comp_uncertainty
        self.comp_uncertainty_mode = comp_uncertainty_mode
        self.comp_uncertainty_gamma = comp_uncertainty_gamma
        self.debug_uncertainty = debug_uncertainty
        self.uncertainty_debug_interval = uncertainty_debug_interval
        self._uncertainty_debug_counter = 0
        self.latest_uncertainty_debug = []
        self.skip_fusion_mode = str(skip_fusion_mode).lower()
        if self.skip_fusion_mode == "auto":
            self.skip_fusion_mode = "casr" if use_casr else "add"
        if self.skip_fusion_mode not in ("add", "casr", "agf", "cagf", "factorized_dynamic", "fdsw"):
            raise ValueError("skip_fusion_mode must be one of: auto, add, casr, agf, cagf, factorized_dynamic, fdsw.")
        self.use_casr = use_casr
        self.casr_use_ucomp = casr_use_ucomp
        self.casr_apply_layers = str(casr_apply_layers)
        self.casr_debug = casr_debug
        self.casr_local_dilation = casr_local_dilation if casr_dilation is None else casr_dilation
        self.factorized_skip_use_prior = factorized_skip_use_prior
        self.use_dynamic_encoder_bank = False
        self.dynamic_bank_apply_layers = str(dynamic_bank_apply_layers)
        self.dynamic_bank_encoder_layers = str(dynamic_bank_encoder_layers)
        self.dynamic_bank_debug = dynamic_bank_debug
        self.latest_dynamic_bank_debug = []
        if self.use_casr and self.casr_use_ucomp and not self.use_comp_uncertainty:
            print("[CASR] warning: casr_use_ucomp=True requires compression uncertainty; CASR will run without u_comp.")
            self.casr_use_ucomp = False
        self.comp_prior = CompressionUncertaintyPrior(
            lambda_dct=lambda_dct,
            lambda_lap=lambda_lap,
            lambda_block=lambda_block,
            block_size=block_size,
            use_dct=comp_use_dct,
            use_lap=comp_use_lap,
            use_block=comp_use_block,
        ) if use_comp_uncertainty else None
        self.register_buffer(
            "image_sobel_x",
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0,
            persistent=False,
        )
        self.register_buffer(
            "image_sobel_y",
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0,
            persistent=False,
        )

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=False)
        self.var = nn.Sequential(nn.Conv2d(width, width, 3, 1, 1), nn.ELU(), nn.Conv2d(width, 3, 1, 1, 0), nn.ELU(),)
        self.gaussian = torchvision.transforms.GaussianBlur(kernel_size=5, sigma=2)
        self.act = nn.GELU()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.transformer_blocks = nn.ModuleList()
        self.var_blocks = nn.ModuleList()
        self.ending_blocks = nn.ModuleList()
        self.casr_blocks = nn.ModuleList()
        self.agf_blocks = nn.ModuleList()
        self.factorized_skip_blocks = nn.ModuleList()
        self.cagf_blocks = nn.ModuleList()
        self.prior_query_gate_blocks = nn.ModuleList()
        self.dynamic_bank_attn_blocks = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.concat = nn.ModuleList()

        chan = width
        encoder_channels = []
        ratio = 1
        step_e = 0
        for num in enc_blk_nums:
            self.downs.append(
                nn.Sequential(
                    nn.Conv2d(chan, int(chan * ratio), kernel_size=2, stride=2, padding=0, bias=False),
                )
            )
            chan = int(chan * ratio)
            encoder_channels.append(chan)
            self.encoders.append(
                CustomSequential(
                    *[Gaussian_block(chan, extra_depth_wise=extra_depth_wise) for _ in range(num)]
                )
            )

        self.middle_blks_enc =\
            CustomSequential(
                *[Gaussian_block(chan, extra_depth_wise=extra_depth_wise) for _ in range(middle_blk_num_enc)]
            )
        self.middle_blks_dec =\
            CustomSequential(
                *[Gaussian_block(chan, extra_depth_wise=extra_depth_wise) for _ in range(middle_blk_num_dec)]
            )

        step_d = 0
        for decoder_layer_idx, num in enumerate(dec_blk_nums):
            if decoder_layer_idx == 0:
                spd_large_kernel = 3
            elif decoder_layer_idx == len(dec_blk_nums) - 1:
                spd_large_kernel = 9
            else:
                spd_large_kernel = 5
            self.decoders.append(
                CustomSequential(
                    *[Gaussian_block(chan, extra_depth_wise=extra_depth_wise) for _ in range(num)]
                )
            )
            self.casr_blocks.append(
                CompressionAdaptiveSkipRecalibration(
                    channels=chan,
                    reduction=casr_reduction,
                    dilation=self.casr_local_dilation,
                    use_ucomp=self.casr_use_ucomp,
                    gate_type=casr_gate_type,
                    use_softmax_selection=casr_use_softmax_selection,
                    debug=casr_debug,
                    ucomp_mode=casr_ucomp_mode,
                    gate_mode=casr_gate_mode,
                    local_branch_dilation=self.casr_local_dilation,
                ) if self.use_casr else nn.Identity()
            )
            self.agf_blocks.append(
                Adaptive_Gated_Fusion(chan, chan) if self.skip_fusion_mode == "agf" else nn.Identity()
            )
            self.factorized_skip_blocks.append(
                FactorizedDynamicSkipWeighting(
                    in_dim=chan,
                    out_dim=chan,
                    k=factorized_skip_k,
                    use_prior=factorized_skip_use_prior,
                    use_safety_gate=factorized_skip_use_safety_gate,
                    softmax_channel_kernel=factorized_skip_softmax_channel_kernel,
                    use_abs_diff=factorized_skip_use_abs_diff,
                    gamma_init=factorized_skip_gamma_init,
                    debug=factorized_skip_debug,
                ) if self.skip_fusion_mode in ("factorized_dynamic", "fdsw") else nn.Identity()
            )
            self.cagf_blocks.append(
                CompressionAwareGatedFusion(
                    in_dim=chan,
                    out_dim=chan,
                    use_prior=factorized_skip_use_prior,
                    gamma_init=factorized_skip_gamma_init,
                    debug=factorized_skip_debug,
                    spd_large_kernel=spd_large_kernel,
                ) if self.skip_fusion_mode == "cagf" else nn.Identity()
            )
            self.prior_query_gate_blocks.append(LocalDegradationQueryGate(chan))
            self.transformer_blocks.append(SpatialChannelInteraction(dim=chan, bank_dim=width, num_blocks=1, dim_head=chan, heads=1))
            self.dynamic_bank_attn_blocks.append(
                UncertaintyGuidedBankAttention(
                    dec_channels=chan,
                    bank_dim=dynamic_bank_dim,
                    attn_dim=dynamic_bank_attn_dim,
                    bank_size=dynamic_bank_size,
                    gamma_init=dynamic_bank_gamma_init,
                    debug=dynamic_bank_debug,
                ) if self.use_dynamic_encoder_bank else nn.Identity()
            )
            self.var_blocks.append(
                nn.Sequential(*[nn.Conv2d(chan, chan, 3, 1, 1),
                                nn.ELU(),
                                nn.Conv2d(chan, chan, 3, 1, 1),
                                nn.ELU(),
                                nn.Conv2d(chan, 3, 1, 1, 0),
                                nn.ELU(),]))
            self.ending_blocks.append(
                nn.Sequential(*[nn.Conv2d(chan, chan//2, 3, 1, 1),
                                nn.ELU(),
                                nn.Conv2d(chan//2, chan//4, 3, 1, 1),
                                nn.ELU(),
                                nn.Conv2d(chan//4, 3, 1, 1, 0),
                                nn.ELU(),]))
            self.ups.append(
                nn.Sequential(
                    nn.ConvTranspose2d(chan, int(chan // ratio), kernel_size=2, stride=2, padding=0, bias=False),
                )
            )
            chan = int(chan // ratio)
        bank_layer_ids = self._parse_dynamic_bank_encoder_layers(self.dynamic_bank_encoder_layers, len(encoder_channels))
        self.dynamic_bank_encoder_layer_ids = bank_layer_ids
        self.dynamic_encoder_bank = DynamicEncoderContextBank(
            [encoder_channels[i] for i in bank_layer_ids],
            bank_dim=dynamic_bank_dim,
            bank_size=dynamic_bank_size,
            use_ucomp_filter=dynamic_bank_use_ucomp_filter,
            debug=dynamic_bank_debug,
        ) if self.use_dynamic_encoder_bank else None
        self.dynamic_global_bank = ContextualCodebookBank(
            encoder_channels,
            bank_dim=width,
            pool_size=16,
            debug=dynamic_bank_debug,
        )
        self.padder_size = 2 ** len(self.encoders)

    def _parse_dynamic_bank_encoder_layers(self, spec, total_layers):
        tokens = [t.strip("'\"[] ") for t in str(spec).split(",") if t.strip("'\"[] ") != ""]
        if not tokens:
            tokens = ["0", "1", "2"]
        layer_ids = [int(t) for t in tokens[:3]]
        if len(layer_ids) != 3:
            raise ValueError("dynamic_bank_encoder_layers must contain exactly three indices, e.g. '0,1,2'.")
        if any(i < 0 or i >= total_layers for i in layer_ids):
            raise ValueError("dynamic_bank_encoder_layers contains an out-of-range encoder index.")
        return layer_ids

    def _casr_enabled_for_layer(self, layer_idx, total_layers):
        if self.skip_fusion_mode != "casr" or not self.use_casr:
            return False
        spec = self.casr_apply_layers.lower().replace(" ", "")
        if spec in ("", "all", "['all']", "[all]"):
            return True
        tokens = [t.strip("'\"[]") for t in spec.split(",") if t.strip("'\"[]") != ""]
        if str(layer_idx) in tokens:
            return True
        deep_end = max(1, total_layers // 3)
        shallow_start = max(0, total_layers - deep_end)
        is_deep = layer_idx < deep_end
        is_shallow = layer_idx >= shallow_start
        is_middle = not is_deep and not is_shallow
        return (
            ("deep" in tokens and is_deep)
            or ("middle" in tokens and is_middle)
            or ("mid" in tokens and is_middle)
            or ("shallow" in tokens and is_shallow)
        )

    def _dynamic_bank_enabled_for_layer(self, layer_idx, total_layers):
        if not self.use_dynamic_encoder_bank:
            return False
        spec = self.dynamic_bank_apply_layers.lower().replace(" ", "")
        if spec in ("", "all", "['all']", "[all]"):
            return True
        tokens = [t.strip("'\"[]") for t in spec.split(",") if t.strip("'\"[]") != ""]
        if str(layer_idx) in tokens:
            return True
        deep_end = max(1, total_layers // 3)
        shallow_start = max(0, total_layers - deep_end)
        is_deep = layer_idx < deep_end
        is_shallow = layer_idx >= shallow_start
        is_middle = not is_deep and not is_shallow
        return (
            ("deep" in tokens and is_deep)
            or ("middle" in tokens and is_middle)
            or ("mid" in tokens and is_middle)
            or ("shallow" in tokens and is_shallow)
        )

    def _fuse_uncertainty(self, raw, u_comp):
        if self.comp_uncertainty_mode == "replace":
            return u_comp.expand_as(raw)
        return raw * (1.0 + self.comp_uncertainty_gamma * u_comp.expand_as(raw))

    def _to_gray(self, img):
        img = torch.nan_to_num(img.float(), nan=0.0, posinf=1.0, neginf=0.0)
        if img.shape[1] == 1:
            return img
        return 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]

    def _safe_pad_image(self, x, pad):
        can_reflect = x.shape[-2] > pad and x.shape[-1] > pad
        mode = "reflect" if can_reflect else "replicate"
        return F.pad(x, (pad, pad, pad, pad), mode=mode)

    def _image_edge_prior(self, img):
        y = self._to_gray(img)
        sobel_x = self.image_sobel_x.to(device=y.device, dtype=y.dtype)
        sobel_y = self.image_sobel_y.to(device=y.device, dtype=y.dtype)
        y_pad = self._safe_pad_image(y, 1)
        grad_x = F.conv2d(y_pad, sobel_x).abs()
        grad_y = F.conv2d(y_pad, sobel_y).abs()
        edge = normalize_per_sample(grad_x + grad_y)
        return edge.to(dtype=img.dtype)

    def _image_local_variance_prior(self, img, kernel_size=3):
        y = self._to_gray(img)
        pad = kernel_size // 2
        y_pad = self._safe_pad_image(y, pad)
        mean = F.avg_pool2d(y_pad, kernel_size=kernel_size, stride=1, padding=0)
        mean2 = F.avg_pool2d(y_pad * y_pad, kernel_size=kernel_size, stride=1, padding=0)
        local_var = torch.clamp(mean2 - mean * mean, min=0.0)
        local_var = normalize_per_sample(local_var)
        return local_var.to(dtype=img.dtype)

    def _debug_tensor_stats(self, name, x):
        x_detached = x.detach()
        return "{} shape={} mean={:.6f} std={:.6f} min={:.6f} max={:.6f}".format(
            name,
            tuple(x_detached.shape),
            float(x_detached.mean()),
            float(x_detached.std(unbiased=False)),
            float(x_detached.min()),
            float(x_detached.max()),
        )

    def _maybe_debug_uncertainty(self, decoder_idx, feat, raw_feature, final_feature, raw_loss, final_loss, aux):
        if not self.debug_uncertainty:
            return
        self._uncertainty_debug_counter += 1
        if self._uncertainty_debug_counter % max(1, self.uncertainty_debug_interval) != 1:
            return
        stats = [
            "[CompUncertainty][decoder={}] mode={}".format(decoder_idx, self.comp_uncertainty_mode),
            self._debug_tensor_stats("feat", feat),
            self._debug_tensor_stats("F_U_raw", raw_feature),
            self._debug_tensor_stats("F_U", final_feature),
            self._debug_tensor_stats("var_raw", raw_loss),
            self._debug_tensor_stats("var_final", final_loss),
            self._debug_tensor_stats("u_std", aux["u_std"]),
            self._debug_tensor_stats("u_dct", aux["u_dct"]),
            self._debug_tensor_stats("u_lap", aux["u_lap"]),
            self._debug_tensor_stats("u_block", aux["u_block"]),
            self._debug_tensor_stats("u_comp", aux["u_comp"]),
        ]
        print("\n".join(stats))

    def forward(self, input, side_loss = False, use_adapter = None):
        out_list = []
        var_list = []
        uncertarinty_list = []
        scoremap_list = []
        self.latest_uncertainty_debug = []
        self.latest_dynamic_bank_debug = []

        _, _, H, W = input.shape

        input = self.check_image_size(input)
        comp_input = input
        edge_prior_full = self._image_edge_prior(comp_input) if (self.skip_fusion_mode == "cagf" or self.use_comp_uncertainty) else None
        edge_prior = edge_prior_full if self.skip_fusion_mode == "cagf" else None
        local_var_prior = self._image_local_variance_prior(comp_input) if self.use_comp_uncertainty else None
        x = self.intro(input)
        skip_first = x

        skips = []

        for encoder, down in zip(self.encoders, self.downs):
            x = down(x)
            x_hf = x
            y, x = encoder(x_hf)
            skips.append(x)

        x_e, x_high = self.middle_blks_enc(x)
        x, _ = self.middle_blks_dec(x_e + x)
        e_global = self.dynamic_global_bank(skips)
        if self.latest_dynamic_bank_debug is not None:
            self.latest_dynamic_bank_debug.append({
                "dynamic_global_bank_encoder_shapes": [tuple(f.shape) for f in skips],
                "E_global_shape": tuple(e_global.shape),
            })
        k_bank, v_bank = None, None
        if self.use_dynamic_encoder_bank:
            bank_feats = [skips[i] for i in self.dynamic_bank_encoder_layer_ids]
            bank_u_comp = None
            if self.dynamic_encoder_bank.use_ucomp_filter and self.use_comp_uncertainty:
                bank_u_comp, _ = self.comp_prior(bank_feats[-1], comp_input)
            k_bank, v_bank = self.dynamic_encoder_bank(bank_feats, u_comp=bank_u_comp)
            self.latest_dynamic_bank_debug.append({
                "encoder_shapes": [tuple(f.shape) for f in bank_feats],
                "k_bank_shape": tuple(k_bank.shape),
                "v_bank_shape": tuple(v_bank.shape),
            })

        total_decoder_layers = len(self.decoders)
        for decoder_idx, (decoder, up, skip, casr, agf, factorized_skip, cagf, prior_query_gate, transformer, dyn_bank_attn, var_block, ending_block) in\
        enumerate(zip(self.decoders, self.ups, skips[::-1], self.casr_blocks, self.agf_blocks, self.factorized_skip_blocks, self.cagf_blocks, self.prior_query_gate_blocks, self.transformer_blocks, self.dynamic_bank_attn_blocks, self.var_blocks, self.ending_blocks)):
            if self._casr_enabled_for_layer(decoder_idx, total_decoder_layers):
                u_comp_skip = None
                if self.casr_use_ucomp and self.use_comp_uncertainty:
                    u_comp_skip, casr_aux = self.comp_prior(x, comp_input)
                    casr_aux["casr_layer"] = decoder_idx
                    self.latest_uncertainty_debug.append(casr_aux)
                x = casr(skip, x, u_comp_skip)
            elif self.skip_fusion_mode == "agf":
                x = agf(skip, x)
            elif self.skip_fusion_mode == "cagf":
                u_prior_skip = None
                if self.factorized_skip_use_prior and self.use_comp_uncertainty:
                    u_prior_skip, factorized_aux = self.comp_prior(skip, comp_input)
                    factorized_aux["cagf_skip_layer"] = decoder_idx
                    self.latest_uncertainty_debug.append(factorized_aux)
                x = cagf(skip, x, u_prior_skip, edge_prior)
            elif self.skip_fusion_mode in ("factorized_dynamic", "fdsw"):
                u_prior_skip = None
                if self.factorized_skip_use_prior and self.use_comp_uncertainty:
                    u_prior_skip, factorized_aux = self.comp_prior(skip, comp_input)
                    factorized_aux["factorized_skip_layer"] = decoder_idx
                    self.latest_uncertainty_debug.append(factorized_aux)
                x = factorized_skip(skip, x, u_prior_skip)
            else:
                x = x + skip
            x_hf, _ = decoder(x)
            _, _, H_, W_ = x.shape
            out_side_resized = F.interpolate(input, (H_, W_), mode='bilinear', align_corners=False)
            var_raw_loss = var_block(x)
            var_raw_feature = var_block[:-2](x)
            if self.use_comp_uncertainty:
                u_comp, aux = self.comp_prior(x_hf, comp_input)
                var_feature = self._fuse_uncertainty(var_raw_feature, u_comp)
                var_loss = self._fuse_uncertainty(var_raw_loss, u_comp)
                query_feature, query_gate, query_gate_aux = prior_query_gate(
                    x_hf, edge_prior_full, local_var_prior
                )
                aux["F_U_raw"] = var_raw_feature.detach()
                aux["F_U_pre_query_gate"] = var_feature.detach()
                aux["F_U"] = var_feature.detach()
                aux["bank_query_feature"] = query_feature.detach()
                aux.update(query_gate_aux)
                self.latest_uncertainty_debug.append(aux)
                self._maybe_debug_uncertainty(decoder_idx, x_hf, var_raw_feature, var_feature, var_raw_loss, var_loss, aux)
            else:
                var_feature = var_raw_feature
                var_loss = var_raw_loss
                query_feature = x_hf
                query_gate = torch.ones_like(x_hf)
            var_list = [var_loss] + var_list
            out_decoder = ending_block(x) + out_side_resized
            out_list = [out_decoder] + out_list
            x, scoremap = transformer(x_hf, query_feature, e_global)
            if self._dynamic_bank_enabled_for_layer(decoder_idx, total_decoder_layers):
                uncertainty_l = var_feature.mean(dim=1, keepdim=True)
                x = dyn_bank_attn(x, uncertainty_l, k_bank, v_bank)
            x = up(x)

        out_side_resized = F.interpolate(input, (H, W), mode='bilinear', align_corners=False)
        final_var_raw = self.var(x)
        if self.use_comp_uncertainty:
            final_u_comp, final_aux = self.comp_prior(x, comp_input)
            final_var = self._fuse_uncertainty(final_var_raw, final_u_comp)
            final_aux["F_U_raw"] = final_var_raw.detach()
            final_aux["F_U"] = final_var.detach()
            self.latest_uncertainty_debug.append(final_aux)
            self._maybe_debug_uncertainty("final", x, final_var_raw, final_var, final_var_raw, final_var, final_aux)
        else:
            final_var = final_var_raw
        var_list = [final_var] + var_list
        x = self.ending(x) + input
        out_list = [x] + out_list

        return out_list, var_list

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), value = 0)
        return x

    def laplacian_hf(self, x):
        """Extract high-frequency component via Laplacian."""
        kernel = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],
                            dtype=x.dtype, device=x.device).view(1,1,3,3)
        kernel = kernel.repeat(x.size(1),1,1,1)
        return F.conv2d(x, kernel, padding=1, groups=x.size(1))
