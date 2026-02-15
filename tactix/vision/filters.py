"""
Project: Tactix
File Created: 2026-02-15 12:00:00
Author: Xingnan Zhu
File Name: filters.py
Description:
    Implements the One Euro Filter for adaptive low-pass filtering.
    Used to smooth the homography matrix across frames — providing high
    smoothness during slow camera movements and low latency during fast pans.
    Reference: Casiez et al., "1€ Filter: A Simple Speed-based Low-pass Filter
    for Noisy Input in Interactive Systems", CHI 2012.
"""

import math
import numpy as np
from typing import Optional


class LowPassFilter:
    """Simple first-order low-pass (exponential moving average) filter."""

    def __init__(self) -> None:
        self._y: Optional[float] = None
        self._s: Optional[float] = None

    def last_value(self) -> Optional[float]:
        return self._s

    def filter(self, value: float, alpha: float) -> float:
        if self._s is None:
            self._s = value
        else:
            self._s = alpha * value + (1.0 - alpha) * self._s
        return self._s


class OneEuroFilterScalar:
    """
    One Euro Filter for a single scalar value.

    Parameters:
        rate:       Sampling frequency (Hz). E.g. video at 30fps → rate=30.
        min_cutoff: Minimum cutoff frequency (Hz). Lower = smoother when still.
        beta:       Speed coefficient. Higher = less lag when moving fast.
        d_cutoff:   Cutoff frequency for the derivative signal (Hz).
    """

    def __init__(
        self,
        rate: float = 30.0,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        self._rate = rate
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._x_filter = LowPassFilter()
        self._dx_filter = LowPassFilter()
        self._last_time: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, rate: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / rate
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: float, timestamp: Optional[float] = None) -> float:
        # Compute effective rate from timestamps if provided
        if self._last_time is not None and timestamp is not None:
            dt = timestamp - self._last_time
            if dt > 0:
                self._rate = 1.0 / dt
        self._last_time = timestamp

        # Estimate derivative (speed)
        prev = self._x_filter.last_value()
        dx = 0.0 if prev is None else (x - prev) * self._rate

        # Filter the derivative
        edx = self._dx_filter.filter(dx, self._alpha(self._d_cutoff, self._rate))

        # Adaptive cutoff: faster movement → higher cutoff → less smoothing
        cutoff = self._min_cutoff + self._beta * abs(edx)

        # Filter the signal
        return self._x_filter.filter(x, self._alpha(cutoff, self._rate))


class OneEuroFilter:
    """
    One Euro Filter for N-dimensional vectors (e.g. flattened 3×3 homography = 9-D).

    Usage:
        filt = OneEuroFilter(ndim=9, rate=30.0, min_cutoff=1.0, beta=0.007)
        smoothed = filt.filter(matrix.flatten())   # returns np.ndarray of shape (9,)
    """

    def __init__(
        self,
        ndim: int,
        rate: float = 30.0,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        self._filters = [
            OneEuroFilterScalar(rate, min_cutoff, beta, d_cutoff)
            for _ in range(ndim)
        ]
        self._ndim = ndim

    def filter(self, values: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        """
        Filter an N-dimensional vector.
        
        Args:
            values: Input array of shape (ndim,).
            timestamp: Optional timestamp in seconds.
            
        Returns:
            Smoothed array of shape (ndim,).
        """
        assert len(values) == self._ndim, (
            f"Expected {self._ndim} values, got {len(values)}"
        )
        result = np.array(
            [f.filter(float(v), timestamp) for f, v in zip(self._filters, values)]
        )
        return result
