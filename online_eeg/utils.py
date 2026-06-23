"""Shared utility functions for paths, arrays, serialization, and reproducibility."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

try:
    from .config import PathLike
except ImportError:
    from config import PathLike


def ensure_dir(path: PathLike) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def set_seed(seed: int) -> None:
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return value

def save_json(data: Dict[str, Any], path: PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(data), f, indent=2)
    return path

def _now_string() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _as_2d_samples_channels(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D data, got shape {arr.shape}.")
    return arr

def channel_index(channels: Sequence[int], hardware_channel: int) -> int:
    channels = [int(ch) for ch in channels]
    hardware_channel = int(hardware_channel)
    if hardware_channel not in channels:
        raise ValueError(
            f"Channel {hardware_channel} is not in acquired channel list {channels}."
        )
    return channels.index(hardware_channel)

def extract_hardware_channels(
    data: np.ndarray,
    acquired_channels: Sequence[int],
    wanted_channels: Sequence[int],
) -> np.ndarray:
    arr = _as_2d_samples_channels(data)
    idx = [channel_index(acquired_channels, ch) for ch in wanted_channels]
    return arr[:, idx].astype(np.float32)

def _duration_to_sample_count(duration_sec: float, samplerate: int, name: str = "duration_sec") -> int:
    samplerate = int(samplerate)
    if samplerate <= 0:
        raise ValueError("samplerate must be positive.")
    if float(duration_sec) <= 0:
        raise ValueError(f"{name} must be positive.")
    return max(1, int(round(float(duration_sec) * samplerate)))

def _checked_acquisition_chunk(chunk: np.ndarray, expected_channels: int) -> np.ndarray:
    arr = _as_2d_samples_channels(chunk)
    if arr.shape[1] != int(expected_channels):
        raise ValueError(
            f"Expected acquired chunk with {expected_channels} channels, got shape {arr.shape}."
        )
    if arr.shape[0] == 0:
        raise RuntimeError("MP150 returned an empty chunk; cannot continue acquisition.")
    return arr

def drop_single_value_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Drop metadata/provenance columns that are constant in a saved table."""

    out = df.copy()
    for column in columns:
        if column in out.columns and out[column].nunique(dropna=False) <= 1:
            out = out.drop(columns=[column])
    return out

def probability_column_map(
    probabilities: np.ndarray,
    class_names: Sequence[str],
) -> Dict[str, np.ndarray]:
    """Return one named probability vector per classifier output."""

    prob = np.asarray(probabilities)
    columns: Dict[str, np.ndarray] = {}
    if prob.ndim != 2:
        raise ValueError(f"Expected probabilities with shape (rows, classes), got {prob.shape}.")
    for label_idx in range(prob.shape[1]):
        name = class_names[label_idx] if label_idx < len(class_names) else f"class_{label_idx}"
        columns[f"prob_{name}"] = prob[:, label_idx]
    return columns

def add_probability_columns(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    class_names: Sequence[str],
) -> pd.DataFrame:
    """Append one probability column per classifier output to ``df`` in place."""

    for column, values in probability_column_map(probabilities, class_names).items():
        df[column] = values
    return df


__all__ = [
    "ensure_dir",
    "set_seed",
    "save_json",
    "channel_index",
    "extract_hardware_channels",
    "drop_single_value_columns",
    "probability_column_map",
    "add_probability_columns",
]
