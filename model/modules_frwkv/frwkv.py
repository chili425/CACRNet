# Copyright (c) Shanghai AI Lab. All rights reserved.
from typing import Sequence
import math, os, sys

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint as cp

from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.cnn.bricks.transformer import PatchEmbed
from mmcls.models.builder import BACKBONES
from mmcls.models.utils import resize_pos_embed
from mmcls.models.backbones.base_backbone import BaseBackbone
import numbers

from .drop import DropPath
from .DQshift import DQShift

logger = logging.getLogger(__name__)

T_MAX = 6553600

import os
python_bin = os.path.dirname(sys.executable)
os.environ["PATH"] = python_bin + os.pathsep + os.environ.get("PATH", "")
if "CUDA_HOME" not in os.environ:
    nvcc_path = os.path.join(python_bin, "nvcc")
    if os.path.exists(nvcc_path):
        os.environ["CUDA_HOME"] = os.path.dirname(python_bin)
import torch.utils.cpp_extension as cpp_extension
if "CUDA_HOME" in os.environ:
    cpp_extension.CUDA_HOME = os.environ["CUDA_HOME"]
load = cpp_extension.load

os.environ['TORCH_EXTENSIONS_DIR'] = os.environ.get(
    'TORCH_EXTENSIONS_DIR',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')
)

current_dir = os.path.dirname(os.path.abspath(__file__))

if not hasattr(torch, "_wkv_cuda_loaded"):
    wkv_cuda = load(
        name="wkv",
        sources=[
            os.path.join(current_dir, "cuda", "wkv_op.cpp"),
            os.path.join(current_dir, "cuda", "wkv_cuda.cu")
        ],
        verbose=False,
        extra_cuda_cflags=[
            '-res-usage', '--maxrregcount 60', '--use_fast_math',
            '-O3', '-Xptxas -O3', f'-DTmax={T_MAX}'
        ]
    )
    torch._wkv_cuda_loaded = wkv_cuda
else:
    wkv_cuda = torch._wkv_cuda_loaded

class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0

        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C

        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        wkv_cuda.backward(B, T, C,
                          w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous(),
                          gw, gu, gk, gv)
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            gw = torch.sum(gw.bfloat16(), dim=0)
            gu = torch.sum(gu.bfloat16(), dim=0)
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)

def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())

def q_shift(input, shift_pixel=1, gamma=1/4, patch_resolution=None):
    assert gamma <= 1/4
    B, T, C= input.shape
    input = input.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])

    B, C, H, W = input.shape
    output = torch.zeros_like(input)

    output[:, 0:int(C*gamma), :, shift_pixel:W] = input[:, 0:int(C*gamma), :, 0:W-shift_pixel]
    output[:, int(C*gamma):int(C*gamma*2), :, 0:W-shift_pixel] = input[:, int(C*gamma):int(C*gamma*2), :, shift_pixel:W]
    output[:, int(C*gamma*2):int(C*gamma*3), shift_pixel:H, :] = input[:, int(C*gamma*2):int(C*gamma*3), 0:H-shift_pixel, :]
    output[:, int(C*gamma*3):int(C*gamma*4), 0:H-shift_pixel, :] = input[:, int(C*gamma*3):int(C*gamma*4), shift_pixel:H, :]
    output[:, int(C*gamma*4):, ...] = input[:, int(C*gamma*4):, ...]

    return output.flatten(2).transpose(1, 2)

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class PhaseDQShiftConv(nn.Module):
    def __init__(self, dim, shift_pixel=1, groups=4, kernel_size=3, dilation=1):
        super().__init__()
        if dim % groups != 0:
            groups = 1
        self.dim = dim
        self.groups = groups
        self.offset = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=(kernel_size // 2) * dilation,
                      dilation=dilation, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, groups * 2, 1)
        )
        self.weight = nn.Sequential(
            nn.Conv2d(dim, groups * 2, 1),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.LeakyReLU(0.1, inplace=True),
        )
        normal_init(self.offset[-1], std=0.001)
        constant_init(self.weight[0], val=0.)

        base = torch.tensor([
            [0., -shift_pixel],
            [0., +shift_pixel],
            [-shift_pixel, 0.],
            [+shift_pixel, 0.],
        ]).transpose(0, 1).view(1, 2, 4, 1, 1)
        if groups == 4:
            shifts = base
        else:
            shifts = torch.zeros(1, 2, groups, 1, 1)
            for i in range(groups):
                shifts[:, :, i:i+1] = base[:, :, i % 4:i % 4 + 1]
        self.register_buffer("shifts", shifts, persistent=False)
        self._coords = None
        self._coords_key = None

    def get_coords(self, H, W, device, dtype):
        key = (H, W, device, dtype)
        if self._coords is None or self._coords_key != key:
            coords_h = torch.arange(H, device=device, dtype=dtype) + 0.5
            coords_w = torch.arange(W, device=device, dtype=dtype) + 0.5
            coords = torch.stack(torch.meshgrid([coords_w, coords_h])).transpose(1, 2)
            coords = coords.unsqueeze(1).unsqueeze(0)
            self._coords = coords
            self._coords_key = key
        return self._coords

    def forward(self, x):
        B, C, H, W = x.shape
        offset = self.offset(x)
        weight = self.weight(x)
        offset = (offset * weight).view(B, 2, self.groups, H, W)
        offset = offset + self.shifts.to(dtype=x.dtype, device=x.device)

        coords = self.get_coords(H, W, x.device, x.dtype)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = coords.permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)

        sampled = F.grid_sample(
            x.reshape(B * self.groups, C // self.groups, H, W),
            coords,
            mode='bilinear',
            align_corners=False,
            padding_mode='border'
        ).view(B, C, H, W)
        return self.refine(sampled)

class VRWKV_FourierMix(BaseModule):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=0, init_mode='fancy',
                 groups=4, key_norm=False, with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.u_embd = n_embd * 2
        self.groups = groups
        self.device = None
        attn_sz = n_embd

        self.idx_dict = {}
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode

        if shift_mode == 'q_shift':
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        elif shift_mode == 'dq_shift':
            self.shift_func = DQShift(n_embd, layer_id, shift_pixel, groups, use_sim=False)
        else:
            self.spatial_mix_k = None
            self.spatial_mix_v = None
            self.spatial_mix_r = None

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance_spa = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance_fft = nn.Linear(attn_sz * 2, attn_sz * 2, bias=False)

        if key_norm:
            self.key_norm = nn.LayerNorm(attn_sz * 2)
        else:
            self.key_norm = None

        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance_spa.scale_init = 0
        self.receptance_fft.scale_init = 0
        self.output.scale_init = 0

        self.with_cp = with_cp

    def _init_weights(self, init_mode):
        if init_mode=='fancy':
            with torch.no_grad():
                ratio_0_to_1 = (self.layer_id / (self.n_layer - 1))
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer))

                decay_speed_half = torch.ones(self.n_embd)
                for h in range(self.n_embd):
                    decay_speed_half[h] = -5 + 8 * (h / (self.n_embd-1)) ** (0.7 + 1.3 * ratio_0_to_1)
                decay_speed = decay_speed_half.repeat(2)
                self.spatial_decay = nn.Parameter(decay_speed)

                zigzag_half = (torch.tensor([(i+1)%3 - 1 for i in range(self.n_embd)]) * 0.5)
                zigzag = zigzag_half.repeat(2)
                self.spatial_first = nn.Parameter(torch.ones(self.u_embd) * math.log(0.3) + zigzag)

                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
                self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))

        elif init_mode=='local':
            self.spatial_decay = nn.Parameter(torch.ones(self.u_embd))
            self.spatial_first = nn.Parameter(torch.ones(self.u_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))

        elif init_mode=='global':
            self.spatial_decay = nn.Parameter(torch.zeros(self.u_embd))
            self.spatial_first = nn.Parameter(torch.zeros(self.u_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)

        else:
            raise NotImplementedError

    def jit_func(self, x, offset_list, patch_resolution):

        B, T, C = x.size()
        if self.shift_mode == 'q_shift':
            xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        elif self.shift_mode == 'dq_shift':
            xx, offset_list = self.shift_func(x, offset_list, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xv = x
            xr = x

        k = self.key(xk)
        v = self.value(xv)
        r_spa = self.receptance_spa(xr)

        return k, v, r_spa, offset_list

    def comp2real(self, x):

        return torch.cat([x.real, x.imag], 1)

    def real2comp(self, x):
        xr, xi = x.chunk(2, dim=1)
        return torch.complex(xr, xi)

    def get_idx_map(self, h, w, device):
        key = (h, w, str(device))
        if key in self.idx_dict:
            return self.idx_dict[key]
        l1_u = torch.arange(h//2, device=device).view(1, 1, -1, 1)
        l2_u = torch.arange(w, device=device).view(1, 1, 1, -1)
        half_map_u = l1_u * l2_u
        l1_d = torch.arange(h - h//2, device=device).flip(0).view(1, 1, -1, 1)
        l2_d = torch.arange(w, device=device).view(1, 1, 1, -1)
        half_map_d = l1_d * l2_d
        freq_map = torch.cat([half_map_u, half_map_d], 2).view(-1)

        idx = freq_map.argsort().to(device).detach()
        inv_idx = torch.empty_like(idx)
        inv_idx[idx] = torch.arange(idx.size(0), device=device)

        self.idx_dict[key] = (idx, inv_idx)
        return idx, inv_idx

    def forward(self, x, offset_list, patch_resolution=None):
        def _inner_forward(x, offset_list):
            B, T, C = x.size()
            H, W = patch_resolution
            self.device = x.device

            k, v, r_spa, offset_list = self.jit_func(x, offset_list, patch_resolution)

            k_fft = torch.fft.rfft2(k.transpose(1, 2).view(B, C, H, W), norm="ortho")
            v_fft = torch.fft.rfft2(v.transpose(1, 2).view(B, C, H, W), norm="ortho")
            k_fft = self.comp2real(k_fft).permute(0, 2, 3, 1).reshape(B, -1, 2 * C)
            v_fft = self.comp2real(v_fft).permute(0, 2, 3, 1).reshape(B, -1, 2 * C)

            idx, inv_idx = self.get_idx_map(H, W//2 + 1, self.device)
            k_fft = torch.index_select(k_fft, dim=1, index=idx)
            v_fft = torch.index_select(v_fft, dim=1, index=idx)
            sr_fft = torch.sigmoid(self.receptance_fft(v_fft))

            T_used = v_fft.size(1)
            rwkv_fft = RUN_CUDA(B, T_used, 2 * C, self.spatial_decay / T_used, self.spatial_first / T_used, k_fft, v_fft)

            if self.key_norm is not None:
                rwkv_fft = rwkv_fft.to(self.device)
                rwkv_fft = self.key_norm(rwkv_fft)
            rwkv_fft = sr_fft * rwkv_fft

            rwkv_fft = torch.index_select(rwkv_fft, dim=1, index=inv_idx)
            rwkv_fft = rwkv_fft.permute(0, 2, 1).reshape(B, 2 * C, H, W//2 + 1)
            rwkv_fft = self.real2comp(rwkv_fft)
            rwkv = torch.fft.irfft2(rwkv_fft, s=patch_resolution, norm='ortho')
            rwkv = rwkv.permute(0, 2, 3, 1).reshape(B, T, C)
            x = rwkv * r_spa
            x = self.output(x)

            return x, offset_list

        if self.with_cp and x.requires_grad:
            x, offset_list = cp.checkpoint(lambda x: _inner_forward(x, offset_list), x)
        else:
            x, offset_list = _inner_forward(x, offset_list)
        return x, offset_list

class VRWKV_SpatialMix(BaseModule):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=0, init_mode='fancy',
                 groups=4, key_norm=False, with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.groups = groups
        self.device = None
        attn_sz = n_embd

        self.idx_dict = {}
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_mode == 'q_shift':
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        elif shift_mode == 'dq_shift':
                self.shift_func = DQShift(n_embd, layer_id, shift_pixel, groups, use_sim=False)
        else:
            self.spatial_mix_k = None
            self.spatial_mix_v = None
            self.spatial_mix_r = None

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)

        if key_norm:
            self.key_norm = nn.LayerNorm(attn_sz)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

        self.with_cp = with_cp

    def _init_weights(self, init_mode):
        if init_mode=='fancy':
            with torch.no_grad():
                ratio_0_to_1 = (self.layer_id / (self.n_layer - 1))
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer))

                decay_speed = torch.ones(self.n_embd)
                for h in range(self.n_embd):
                    decay_speed[h] = -5 + 8 * (h / (self.n_embd-1)) ** (0.7 + 1.3 * ratio_0_to_1)
                self.spatial_decay = nn.Parameter(decay_speed)

                zigzag = (torch.tensor([(i+1)%3 - 1 for i in range(self.n_embd)]) * 0.5)
                self.spatial_first = nn.Parameter(torch.ones(self.n_embd) * math.log(0.3) + zigzag)

                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
                self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))

        elif init_mode=='local':
            self.spatial_decay = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_first = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))

        elif init_mode=='global':
            self.spatial_decay = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_first = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)

        else:
            raise NotImplementedError

    def jit_func(self, x, offset_list, patch_resolution):

        B, T, C = x.size()
        if self.shift_mode == 'q_shift':
            xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        elif self.shift_mode == 'dq_shift':
            xx, offset_list = self.shift_func(x, offset_list, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xv = x
            xr = x

        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        return k, v, sr, offset_list

    def forward(self, x, offset_list, patch_resolution=None):
        def _inner_forward(x, offset_list):
            B, T, C = x.size()

            self.device = x.device

            k, v, sr, offset_list = self.jit_func(x, offset_list, patch_resolution)
            x = RUN_CUDA(B, T, C, self.spatial_decay / T, self.spatial_first / T, k, v)

            if self.key_norm is not None:
                x = x.to(self.device)
                x = self.key_norm(x)
            x = sr * x
            x = self.output(x)
            return x, offset_list

        if self.with_cp and x.requires_grad:
            x, offset_list = cp.checkpoint(lambda x: _inner_forward(x, offset_list), x)
        else:
            x, offset_list = _inner_forward(x, offset_list)
        return x, offset_list

class VRWKV_ChannelMix(BaseModule):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=0, hidden_rate=4, init_mode='fancy',
                 groups=4, key_norm=False, with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.groups = groups
        self.with_cp = with_cp
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_mode == 'q_shift':
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        elif shift_mode == 'dq_shift':
                self.shift_func = DQShift(n_embd, layer_id, shift_pixel, groups, use_sim=False)
        else:
            self.spatial_mix_k = None
            self.spatial_mix_r = None

        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad():
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer))
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        elif init_mode == 'local':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode == 'global':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError

    def forward(self, x, offset_list, patch_resolution=None):
        def _inner_forward(x, offset_list):
            if self.shift_mode == 'q_shift':
                xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, patch_resolution)
                xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
                xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
            elif self.shift_mode == 'dq_shift':
                xx, offset_list = self.shift_func(x, offset_list, patch_resolution)
                xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
                xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
            else:
                xk = x
                xr = x

            k = self.key(xk)
            k = torch.square(torch.relu(k))
            if self.key_norm is not None:
                k = self.key_norm(k)
            kv = self.value(k)
            x = torch.sigmoid(self.receptance(xr)) * kv
            return x, offset_list
        if self.with_cp and x.requires_grad:
            x, offset_list = cp.checkpoint(lambda x: _inner_forward(x, offset_list), x)
        else:
            x, offset_list = _inner_forward(x, offset_list)
        return x, offset_list

class Block(BaseModule):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=0, drop_path=0., hidden_rate=4,
                 init_mode='fancy', init_values=None, post_norm=False, key_norm=False,
                 with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.att = VRWKV_FourierMix(n_embd, n_layer, layer_id, shift_mode,
                                   channel_gamma, shift_pixel, init_mode,
                                   key_norm=key_norm)

        self.ffn = VRWKV_ChannelMix(n_embd, n_layer, layer_id, shift_mode,
                                   channel_gamma, shift_pixel, hidden_rate,
                                   init_mode, key_norm=key_norm)

        self.layer_scale = (init_values is not None)
        self.post_norm = post_norm
        if self.layer_scale:
            self.gamma1 = nn.Parameter(init_values * torch.ones((n_embd)), requires_grad=True)
            self.gamma2 = nn.Parameter(init_values * torch.ones((n_embd)), requires_grad=True)
        self.with_cp = with_cp

    def forward(self, x, offset_list, patch_resolution=None):
        def _inner_forward(x, offset_list):
            if self.post_norm:
                if self.layer_scale:

                    att, offset_list = self.att(x, offset_list, patch_resolution)
                    x = x + self.drop_path(self.gamma1 * self.ln1(att))

                    ffn, offset_list = self.ffn(x, offset_list, patch_resolution)
                    x = x + self.drop_path(self.gamma2 * self.ln2(ffn))
                else:

                    att, offset_list = self.att(x, offset_list, patch_resolution)
                    x = x + self.drop_path(self.ln1(att))

                    ffn, offset_list = self.ffn(x, offset_list, patch_resolution)
                    x = x + self.drop_path(self.ln2(ffn))
            else:
                if self.layer_scale:

                    x_ln1 = self.ln1(x)
                    att, offset_list = self.att(x_ln1, offset_list, patch_resolution)
                    x = x + self.drop_path(self.gamma1 * att)

                    x_ln2 = self.ln2(x)
                    ffn, offset_list = self.ffn(x_ln2, offset_list, patch_resolution)
                    x = x + self.drop_path(self.gamma2 * ffn)
                else:

                    x_ln1 = self.ln1(x)
                    att, offset_list = self.att(x_ln1, offset_list, patch_resolution)
                    x = x + self.drop_path(att)

                    x_ln2 = self.ln2(x)
                    ffn, offset_list = self.ffn(x_ln2, offset_list, patch_resolution)
                    x = x + self.drop_path(ffn)
            return x, offset_list
        if self.with_cp and x.requires_grad:
            x, offset_list = cp.checkpoint(lambda x: _inner_forward(x, offset_list), x)
        else:
            x, offset_list = _inner_forward(x, offset_list)
        return x, offset_list

@BACKBONES.register_module()
class FRWKV(BaseBackbone):
    def __init__(self,
                 img_size=32,
                 patch_size=3,
                 in_channels=3,
                 out_indices=-1,
                 drop_rate=0.,
                 embed_dims=32,
                 depth=2,
                 drop_path_rate=0.,
                 channel_gamma=1/4,
                 shift_pixel=1,
                 shift_mode='q_shift',
                 init_mode='fancy',
                 post_norm=False,
                 key_norm=True,
                 init_values=None,
                 hidden_rate=4,
                 final_norm=False,
                 interpolate_mode='bicubic',
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_extra_tokens = 0
        self.num_layers = depth
        self.drop_path_rate = drop_path_rate

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=1,
            bias=True)

        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        self.interpolate_mode = interpolate_mode
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, self.embed_dims))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence),\
            f'"out_indices" must by a sequence or int, '\
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers,\
                f'Invalid out_indices {index}'
        self.out_indices = out_indices
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.layers = ModuleList()
        for i in range(self.num_layers):
            self.layers.append(Block(
                n_embd=embed_dims,
                n_layer=depth,
                layer_id=i,
                channel_gamma=channel_gamma,
                shift_pixel=shift_pixel,
                shift_mode=shift_mode,
                hidden_rate=hidden_rate,
                drop_path=dpr[i],
                init_mode=init_mode,
                post_norm=post_norm,
                key_norm=key_norm,
                init_values=init_values,
                with_cp=with_cp
            ))

        self.final_norm = final_norm
        if final_norm:
            self.ln1 = nn.LayerNorm(self.embed_dims)

    def forward(self, x):
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)

        x = self.drop_after_pos(x)

        outs = []
        offset_list = []
        for i, layer in enumerate(self.layers):
            x, offset_list = layer(x, offset_list, patch_resolution)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.ln1(x)

            if i in self.out_indices:
                B, _, C = x.shape
                patch_token = x.reshape(B, *patch_resolution, C)
                patch_token = patch_token.permute(0, 3, 1, 2)

                out = patch_token
                outs.append(out)

        return outs[0]
