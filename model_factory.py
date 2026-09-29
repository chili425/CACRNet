from model.cacrnet import CACRNet


def build_model(width=144):
    return CACRNet(
        img_channel=3,
        width=width,
        middle_blk_num_enc=1,
        middle_blk_num_dec=1,
        enc_blk_nums=[1, 1, 1, 1],
        dec_blk_nums=[1, 1, 1, 1],
        extra_depth_wise=True,
        use_comp_uncertainty=True,
        skip_fusion_mode="cagf",
        factorized_skip_use_prior=True,
        factorized_skip_gamma_init=0.2,
    )
