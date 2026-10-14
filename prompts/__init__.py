"""Prompt constants for SkillGrad."""

from .executor import EXECUTOR_PROMPT
from .diagnoser import FAILURE_DIAGNOSER_PROMPT, CONTRASTIVE_DIAGNOSER_PROMPT
from .momentum import MOMENTUM_PROMPT
from .patcher import PATCHER_PROMPT

__all__ = [
    "EXECUTOR_PROMPT",
    "FAILURE_DIAGNOSER_PROMPT",
    "CONTRASTIVE_DIAGNOSER_PROMPT",
    "MOMENTUM_PROMPT",
    "PATCHER_PROMPT",
]
