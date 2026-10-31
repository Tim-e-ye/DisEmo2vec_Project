"""DisEmo2vec Pluggable Hammer Slots & Model Interfaces."""

from .tbd_hammer_slots import (
    HammerRedLineViolationError,
    HammerSlotH1_ViewTransform,
    HammerSlotH2_Disentangler,
    HammerSlotH3_TemporalAggregator,
    PluggableDisEmo2vecFrontend,
    build_hammer_suite_from_config,
    register_hammer_card,
    run_hammer_slots_self_test,
)

__all__ = [
    "HammerRedLineViolationError",
    "HammerSlotH1_ViewTransform",
    "HammerSlotH2_Disentangler",
    "HammerSlotH3_TemporalAggregator",
    "PluggableDisEmo2vecFrontend",
    "build_hammer_suite_from_config",
    "register_hammer_card",
    "run_hammer_slots_self_test",
]
