"""Configuration dataclasses shared across the online EEG pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Sequence, TYPE_CHECKING, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

ArrayLike = Union[np.ndarray, Sequence[float]]
PathLike = Union[str, os.PathLike]


@dataclass
class AcquisitionConfig:
    """BIOPAC MP150 collection settings."""

    samplerate: int = 200
    channels: Tuple[int, ...] = (1, 2, 3, 4, 16)
    chunk_sec: float = 0.10

@dataclass
class AudioLabelConfig:
    """Settings for deriving sample labels from the audio cue channel.

    The default is a binary state-transition task: label 0 before the
    first cue, then switch to 1 at the first cue, back to 0 at the second cue,
    and so on. For explicit cued multi-class tasks, set ``cue_label_sequence``
    to the expected labels in onset order, for example ``[1, 2, 1, 2]``.
    """

    class_names: Tuple[str, ...] = ("Idle", "Task")
    baseline_label: int = 0
    active_label: int = 1
    cue_label_sequence: Optional[Tuple[Union[int, str], ...]] = None
    cycle_cue_sequence: bool = True
    alternate_binary_labels: bool = True
    label_duration_sec: Optional[float] = None
    label_start_offset_sec: float = 0.0
    infer_labels_from_audio_peaks: bool = False
    n_peak_label_classes: Optional[int] = None

    envelope_window_sec: float = 0.025
    onset_threshold: Optional[float] = None
    onset_threshold_mad_multiplier: float = 8.0
    onset_min_interval_sec: float = 0.50
    onset_peak_window_sec: float = 0.20

@dataclass
class PreprocessConfig:
    """Signal preprocessing and channel layout."""

    eeg_channels: Tuple[int, ...] = (1, 2, 3, 4)
    audio_channel: int = 16
    apply_software_filters: bool = True
    bandpass_low_hz: Optional[float] = 1.0
    bandpass_high_hz: Optional[float] = 40.0
    notch_hz: Optional[Tuple[float, ...]] = (60.0,)
    notch_quality_factor: float = 30.0
    filter_order: int = 4
    demean_channels: bool = True

@dataclass
class WindowConfig:
    """Sliding-window settings for time-domain EEG windows."""

    feature_mode: str = "filtered_signal"
    window_sec: float = 1.0
    stride_sec: float = 0.10
    label_mode: str = "endpoint"

@dataclass
class TrainingConfig:
    """LSTM training settings."""

    train_fraction: float = 0.80
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.0
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    seed: int = 888
    device: Optional[str] = None

@dataclass
class DatasetBundle:
    """Windowed train/validation arrays plus metadata."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    train_windows: pd.DataFrame
    val_windows: pd.DataFrame
    normalizer_mean: np.ndarray
    normalizer_std: np.ndarray
    fs: int
    class_names: Tuple[str, ...]
    window_config: WindowConfig
    source_files: Tuple[str, ...] = field(default_factory=tuple)


__all__ = [
    "ArrayLike",
    "PathLike",
    "AcquisitionConfig",
    "AudioLabelConfig",
    "PreprocessConfig",
    "WindowConfig",
    "TrainingConfig",
    "DatasetBundle",
]
