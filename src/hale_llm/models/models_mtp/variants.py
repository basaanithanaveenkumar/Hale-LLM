"""Names of the model variants this package registers."""

MTP_VARIANT = "autoregressive_mtp"  # dense or mixture-of-experts, as model.ffn_type says
MOE_MTP_VARIANT = "autoregressive_moe_mtp"  # always a mixture-of-experts trunk
ALL_VARIANTS = (MTP_VARIANT, MOE_MTP_VARIANT)
