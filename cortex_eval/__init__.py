"""cortex_eval — evaluation harnesses for the C2 ablation protocol (ADR-006).

Slice B: MQAR (associative recall, by difficulty tiers) and the serial-position
curve (primacy/recency). Evaluation-only; validated on the control before any
ablation run. The founder's principle: never measure at a single point that may
be flat — sweep a scale and read the curve.
"""
from cortex_eval.mqar import (
    MQARTier, MQARResult, make_batch, standard_curriculum, accuracy, SEP,
)
from cortex_eval.serial_position import (
    SerialPositionTask, SerialPositionResult, aggregate_by_position,
    make_batch as make_serial_batch, PROBE,
)

__all__ = [
    "MQARTier", "MQARResult", "make_batch", "standard_curriculum", "accuracy", "SEP",
    "SerialPositionTask", "SerialPositionResult", "aggregate_by_position",
    "make_serial_batch", "PROBE",
]
