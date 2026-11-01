#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared tensor/numpy compatibility & red-line exception base for `models/hammer_slots/`."""

from __future__ import annotations

from typing import Any
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
    BaseModule = nn.Module
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    HAS_TORCH = False

    class BaseModule:  # type: ignore
        """Lightweight nn.Module shim when running local self-test without PyTorch installed."""

        def __init__(self) -> None:
            self.training = True

        def train(self, mode: bool = True) -> "BaseModule":
            self.training = mode
            return self

        def eval(self) -> "BaseModule":
            self.training = False
            return self

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.forward(*args, **kwargs)


class HammerRedLineViolationError(RuntimeError):
    """Raised when a candidate literature hammer card violates a locked reviewer red-line constraint."""


def to_numpy(x: Any) -> np.ndarray:
    if HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def scalar_value(x: Any) -> float:
    if HAS_TORCH and isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(np.asarray(x).item())
