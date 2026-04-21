"""Shared numpy array type aliases.

A single alias in a single place keeps every module consistent and makes the future Rust
port obvious: ``FloatArray`` maps to ``ndarray::Array<f64, _>`` / ``nalgebra::DMatrix<f64>``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
