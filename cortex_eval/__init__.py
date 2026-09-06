"""cortex_eval — evaluation harnesses for the C2 ablation protocol (ADR-006).

Slice B: MQAR (associative recall, by difficulty tiers) and the serial-position
curve. Evaluation-only; validated on the control before any ablation run.
"""
from cortex_eval.mqar import (
    MQARTier, MQARResult, make_batch, standard_curriculum, accuracy, SEP,
)

__all__ = ["MQARTier", "MQARResult", "make_batch", "standard_curriculum",
           "accuracy", "SEP"]
