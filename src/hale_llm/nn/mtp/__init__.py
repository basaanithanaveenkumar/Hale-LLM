"""Multi-token prediction (DeepSeek-V3 style), plus an offset-conditioned variant of it."""

from hale_llm.nn.mtp.build_predictor import build_mtp_predictor
from hale_llm.nn.mtp.mtp_module import MTPModule
from hale_llm.nn.mtp.multi_token_predictor import MultiTokenPredictor
from hale_llm.nn.mtp.offset_conditioned_predictor import OffsetConditionedPredictor
from hale_llm.nn.mtp.offset_share import OffsetShare
from hale_llm.nn.mtp.stratified_offset_assigner import StratifiedOffsetAssigner

__all__ = [
    "MTPModule",
    "MultiTokenPredictor",
    "OffsetConditionedPredictor",
    "OffsetShare",
    "StratifiedOffsetAssigner",
    "build_mtp_predictor",
]
