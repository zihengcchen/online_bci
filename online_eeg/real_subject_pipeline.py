# -*- coding: utf-8 -*-
"""Real-subject MP150 collection, labeling, training, and testing pipeline.

This module consolidates the notebook logic in ``signal_generator`` into
functions that can be called from one orchestration notebook. The copied
``mpy150_chunk.py`` file is imported lazily and is not modified here.
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


ArrayLike = Union[np.ndarray, Sequence[float]]
PathLike = Union[str, os.PathLike]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AcquisitionConfig:
    """BIOPAC MP150 collection settings."""

    samplerate: int = 200
    channels: Tuple[int, ...] = (1, 2, 3, 4, 5)
    chunk_sec: float = 0.10


@dataclass
class AudioLabelConfig:
    """Settings for deriving sample labels from the audio cue channel.

    The default is a binary task where each detected cue marks class 1 for
    ``label_duration_sec`` seconds and everything else is class 0. For cued
    multi-class tasks, set ``cue_label_sequence`` to the expected labels in
    onset order, for example ``[1, 2, 1, 2]``.
    """

    class_names: Tuple[str, ...] = ("Idle", "Task")
    baseline_label: int = 0
    active_label: int = 1
    cue_label_sequence: Optional[Tuple[Union[int, str], ...]] = None
    cycle_cue_sequence: bool = True
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
    audio_channel: int = 5
    bandpass_low_hz: Optional[float] = 1.0
    bandpass_high_hz: Optional[float] = 40.0
    notch_hz: Optional[Tuple[float, ...]] = (60.0,)
    notch_quality_factor: float = 30.0
    filter_order: int = 4
    demean_channels: bool = True


@dataclass
class WindowConfig:
    """Sliding-window and feature extraction settings."""

    feature_mode: str = "raw_signal"  # "raw_signal" or "fft_bandpower"
    window_sec: float = 1.0
    stride_sec: float = 0.10
    label_mode: str = "endpoint"  # "endpoint" or "majority"
    bandpower_hz: Tuple[float, float] = (18.0, 22.0)


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


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def ensure_dir(path: PathLike) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
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


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def _import_mp150_class():
    try:
        from .mpy150_chunk import MP150  # type: ignore
    except Exception:
        from mpy150_chunk import MP150  # type: ignore
    return MP150


def collect_mp150_recording(
    output_npz: PathLike,
    duration_sec: float,
    config: AcquisitionConfig,
    segment_name: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Collect one MP150 recording and save it as a raw NPZ.

    Saved key ``data`` has shape ``(samples, channels)`` and uses the hardware
    channel order in ``config.channels``.
    """

    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive.")
    if config.chunk_sec <= 0:
        raise ValueError("config.chunk_sec must be positive.")

    MP150 = _import_mp150_class()
    mp = MP150(samplerate=int(config.samplerate), channels=list(config.channels))

    chunks: List[np.ndarray] = []
    chunk_start_wall: List[float] = []
    chunk_end_wall: List[float] = []

    start_wall = time.time()
    try:
        while True:
            elapsed = time.time() - start_wall
            remaining = float(duration_sec) - elapsed
            if remaining <= 0:
                break

            request_sec = min(float(config.chunk_sec), max(remaining, 1.0 / config.samplerate))
            chunk_start_wall.append(time.time() - start_wall)
            chunk = mp.get_chunk(request_sec)
            chunk_end_wall.append(time.time() - start_wall)
            chunks.append(_as_2d_samples_channels(chunk))
    finally:
        mp.close()

    if chunks:
        data = np.concatenate(chunks, axis=0).astype(np.float32)
    else:
        data = np.empty((0, len(config.channels)), dtype=np.float32)

    time_sec = np.arange(data.shape[0], dtype=np.float64) / float(config.samplerate)
    np.savez(
        output_npz,
        data=data,
        samplerate=np.array(int(config.samplerate), dtype=np.int64),
        channels=np.asarray(config.channels, dtype=np.int64),
        time_sec=time_sec,
        segment_name=np.array(str(segment_name)),
        chunk_start_wall_sec=np.asarray(chunk_start_wall, dtype=np.float64),
        chunk_end_wall_sec=np.asarray(chunk_end_wall, dtype=np.float64),
        requested_duration_sec=np.array(float(duration_sec), dtype=np.float64),
        created_at=np.array(_now_string()),
        metadata_json=np.array(json.dumps(_json_safe(metadata or {}))),
    )
    return output_npz


def collect_training_segments(
    output_dir: PathLike,
    segment_names: Sequence[str],
    config: AcquisitionConfig,
    duration_sec: float = 300.0,
) -> List[Path]:
    """Collect several fixed-duration training recordings."""

    output_dir = ensure_dir(output_dir)
    paths = []
    for name in segment_names:
        safe_name = str(name).strip().replace(" ", "_") or f"segment_{len(paths) + 1:02d}"
        out = output_dir / f"{safe_name}.npz"
        paths.append(
            collect_mp150_recording(
                output_npz=out,
                duration_sec=duration_sec,
                config=config,
                segment_name=safe_name,
            )
        )
    return paths


def load_raw_recording(path: PathLike) -> Dict[str, Any]:
    raw = np.load(Path(path), allow_pickle=True)
    if "data" not in raw.files:
        raise KeyError(f"Raw recording {path} is missing key 'data'.")
    if "samplerate" not in raw.files:
        raise KeyError(f"Raw recording {path} is missing key 'samplerate'.")
    if "channels" not in raw.files:
        raise KeyError(f"Raw recording {path} is missing key 'channels'.")
    return {
        "data": _as_2d_samples_channels(raw["data"]),
        "samplerate": int(np.asarray(raw["samplerate"]).item()),
        "channels": tuple(int(x) for x in np.asarray(raw["channels"]).reshape(-1)),
        "path": str(path),
        "segment_name": str(np.asarray(raw["segment_name"]).item()) if "segment_name" in raw.files else Path(path).stem,
    }


# ---------------------------------------------------------------------------
# Audio cue labeling and preprocessing
# ---------------------------------------------------------------------------


def rolling_rms_causal(x: ArrayLike, window_samples: int) -> np.ndarray:
    """Causal rolling RMS with the same output length as ``x``."""

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    window_samples = int(window_samples)
    if window_samples < 1:
        raise ValueError("window_samples must be >= 1.")
    if len(x) == 0:
        return np.empty(0, dtype=np.float64)

    x2 = x ** 2
    csum = np.concatenate([[0.0], np.cumsum(x2)])
    end = np.arange(1, len(x2) + 1)
    start = np.maximum(0, end - window_samples)
    window_sums = csum[end] - csum[start]
    window_counts = end - start
    return np.sqrt(window_sums / window_counts)


def _auto_audio_threshold(envelope: np.ndarray, mad_multiplier: float) -> float:
    finite = envelope[np.isfinite(envelope)]
    if finite.size == 0:
        return 0.0
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    robust_sigma = 1.4826 * mad
    threshold = med + float(mad_multiplier) * robust_sigma
    if not np.isfinite(threshold) or threshold <= med:
        p90 = float(np.percentile(finite, 90))
        p99 = float(np.percentile(finite, 99))
        threshold = med + 0.5 * (p99 - med)
        if threshold <= med:
            threshold = p90
    return float(threshold)


def detect_audio_onsets(
    audio: ArrayLike,
    fs: int,
    config: AudioLabelConfig,
) -> Dict[str, Any]:
    """Detect cue onsets from the audio channel."""

    audio = np.asarray(audio, dtype=np.float64).reshape(-1)
    fs = int(fs)
    if fs <= 0:
        raise ValueError("fs must be positive.")

    centered = audio - np.nanmedian(audio)
    env_samples = max(1, int(round(float(config.envelope_window_sec) * fs)))
    envelope = rolling_rms_causal(centered, env_samples)
    threshold = (
        float(config.onset_threshold)
        if config.onset_threshold is not None
        else _auto_audio_threshold(envelope, config.onset_threshold_mad_multiplier)
    )

    active = envelope >= threshold
    rising = np.flatnonzero(active & np.r_[True, ~active[:-1]])

    min_interval_samples = max(1, int(round(float(config.onset_min_interval_sec) * fs)))
    kept: List[int] = []
    for idx in rising:
        idx = int(idx)
        if not kept or idx - kept[-1] >= min_interval_samples:
            kept.append(idx)
        else:
            # Keep the earlier onset but update it if the later edge has a stronger envelope.
            prev = kept[-1]
            if envelope[idx] > envelope[prev]:
                kept[-1] = idx

    peak_window = max(1, int(round(float(config.onset_peak_window_sec) * fs)))
    peaks = []
    for onset in kept:
        end = min(len(envelope), int(onset) + peak_window)
        peaks.append(float(np.max(envelope[int(onset):end])) if end > onset else float(envelope[onset]))

    return {
        "onset_samples": np.asarray(kept, dtype=np.int64),
        "onset_times_sec": np.asarray(kept, dtype=np.float64) / float(fs),
        "peak_values": np.asarray(peaks, dtype=np.float64),
        "envelope": envelope.astype(np.float64),
        "threshold": float(threshold),
        "envelope_window_samples": int(env_samples),
    }


def _label_to_int(label: Union[int, str], class_names: Sequence[str]) -> int:
    if isinstance(label, (int, np.integer)):
        return int(label)
    label_text = str(label)
    for idx, name in enumerate(class_names):
        if label_text.lower() == str(name).lower():
            return idx
    raise ValueError(f"Unknown label {label!r}; class_names={tuple(class_names)!r}.")


def _infer_peak_labels(peaks: np.ndarray, n_classes: int, baseline_label: int) -> np.ndarray:
    """Small dependency-free 1D peak clustering for distinct cue amplitudes."""

    peaks = np.asarray(peaks, dtype=np.float64).reshape(-1)
    if len(peaks) == 0:
        return np.empty(0, dtype=np.int64)
    if n_classes <= 1:
        return np.full(len(peaks), int(baseline_label), dtype=np.int64)

    # Exclude the baseline from cue classes. With class_names=("Idle", "Left", "Right"),
    # peaks map to labels 1 and 2.
    cue_labels = [i for i in range(n_classes) if i != int(baseline_label)]
    if not cue_labels:
        cue_labels = list(range(n_classes))

    n_clusters = len(cue_labels)
    if n_clusters == 1:
        return np.full(len(peaks), cue_labels[0], dtype=np.int64)

    quantiles = np.linspace(0, 100, n_clusters + 2)[1:-1]
    centers = np.percentile(peaks, quantiles).astype(np.float64)
    for _ in range(20):
        distances = np.abs(peaks[:, None] - centers[None, :])
        cluster_idx = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        for k in range(n_clusters):
            if np.any(cluster_idx == k):
                new_centers[k] = peaks[cluster_idx == k].mean()
        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    order = np.argsort(centers)
    sorted_cue_labels = np.asarray(cue_labels, dtype=np.int64)
    cluster_to_label = {int(cluster): int(sorted_cue_labels[rank]) for rank, cluster in enumerate(order)}
    return np.asarray([cluster_to_label[int(k)] for k in cluster_idx], dtype=np.int64)


def labels_from_audio_onsets(
    n_samples: int,
    fs: int,
    onset_info: Dict[str, Any],
    config: AudioLabelConfig,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Create sample-level labels and a cue table from detected audio onsets."""

    n_samples = int(n_samples)
    fs = int(fs)
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative.")

    class_names = tuple(config.class_names)
    labels = np.full(n_samples, int(config.baseline_label), dtype=np.int64)
    onsets = np.asarray(onset_info["onset_samples"], dtype=np.int64).reshape(-1)
    peaks = np.asarray(onset_info.get("peak_values", np.full(len(onsets), np.nan)), dtype=np.float64)

    if config.infer_labels_from_audio_peaks:
        n_classes = int(config.n_peak_label_classes or len(class_names))
        cue_labels = _infer_peak_labels(peaks, n_classes=n_classes, baseline_label=config.baseline_label)
    elif config.cue_label_sequence is not None:
        seq = [_label_to_int(item, class_names) for item in config.cue_label_sequence]
        if not seq:
            raise ValueError("cue_label_sequence cannot be empty.")
        cue_labels = []
        for i in range(len(onsets)):
            if i < len(seq):
                cue_labels.append(seq[i])
            elif config.cycle_cue_sequence:
                cue_labels.append(seq[i % len(seq)])
            else:
                raise ValueError(
                    f"Detected {len(onsets)} cues but cue_label_sequence only has {len(seq)} labels."
                )
        cue_labels = np.asarray(cue_labels, dtype=np.int64)
    else:
        cue_labels = np.full(len(onsets), int(config.active_label), dtype=np.int64)

    start_offset = int(round(float(config.label_start_offset_sec) * fs))
    fixed_duration = None
    if config.label_duration_sec is not None:
        fixed_duration = max(1, int(round(float(config.label_duration_sec) * fs)))

    rows: List[Dict[str, Any]] = []
    for i, onset in enumerate(onsets):
        start = int(np.clip(int(onset) + start_offset, 0, n_samples))
        if fixed_duration is None:
            next_onset = int(onsets[i + 1]) + start_offset if i + 1 < len(onsets) else n_samples
            end = int(np.clip(next_onset, start, n_samples))
        else:
            end = int(np.clip(start + fixed_duration, start, n_samples))

        label = int(cue_labels[i])
        labels[start:end] = label
        rows.append(
            {
                "cue_index": i,
                "onset_sample": int(onset),
                "onset_time_sec": float(onset) / float(fs),
                "label_start_sample": start,
                "label_end_sample": end,
                "label_start_time_sec": start / float(fs),
                "label_end_time_sec": end / float(fs),
                "label": label,
                "label_name": class_names[label] if 0 <= label < len(class_names) else str(label),
                "peak_value": float(peaks[i]) if i < len(peaks) else np.nan,
            }
        )

    return labels, pd.DataFrame(rows)


def bandpass_filter(
    data: np.ndarray,
    fs: int,
    low_hz: Optional[float],
    high_hz: Optional[float],
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth filter. Returns the input unchanged if disabled."""

    arr = _as_2d_samples_channels(data).astype(np.float32)
    if low_hz is None and high_hz is None:
        return arr

    try:
        from scipy.signal import butter, sosfilt, sosfiltfilt
    except Exception as exc:
        raise ImportError("scipy is required for bandpass filtering.") from exc

    fs = int(fs)
    nyq = fs / 2.0
    if low_hz is not None and high_hz is not None:
        wn = [float(low_hz) / nyq, float(high_hz) / nyq]
        btype = "bandpass"
    elif low_hz is not None:
        wn = float(low_hz) / nyq
        btype = "highpass"
    else:
        wn = float(high_hz) / nyq
        btype = "lowpass"

    sos = butter(int(order), wn, btype=btype, output="sos")
    try:
        filtered = sosfiltfilt(sos, arr, axis=0)
    except ValueError:
        filtered = sosfilt(sos, arr, axis=0)
    return filtered.astype(np.float32)


def _notch_values(notch_hz: Optional[Union[float, Sequence[float]]]) -> Tuple[float, ...]:
    if notch_hz is None:
        return tuple()
    if np.isscalar(notch_hz):
        return (float(notch_hz),)
    return tuple(float(x) for x in notch_hz)


def notch_filter(
    data: np.ndarray,
    fs: int,
    notch_hz: Optional[Union[float, Sequence[float]]] = (60.0,),
    quality_factor: float = 30.0,
) -> np.ndarray:
    """Apply one or more zero-phase IIR notch filters."""

    arr = _as_2d_samples_channels(data).astype(np.float32)
    values = _notch_values(notch_hz)
    if not values:
        return arr

    try:
        from scipy.signal import filtfilt, iirnotch, lfilter
    except Exception as exc:
        raise ImportError("scipy is required for notch filtering.") from exc

    fs = int(fs)
    nyq = fs / 2.0
    filtered = arr.astype(np.float64)
    for hz in values:
        if hz <= 0 or hz >= nyq:
            continue
        b, a = iirnotch(w0=float(hz), Q=float(quality_factor), fs=fs)
        try:
            filtered = filtfilt(b, a, filtered, axis=0)
        except ValueError:
            filtered = lfilter(b, a, filtered, axis=0)
    return filtered.astype(np.float32)


def preprocess_eeg_signal(
    eeg_raw: np.ndarray,
    fs: int,
    preprocess_config: PreprocessConfig,
) -> np.ndarray:
    """Apply the EEG preprocessing used before feature extraction."""

    eeg = _as_2d_samples_channels(eeg_raw).astype(np.float32)
    if preprocess_config.demean_channels:
        eeg = eeg - np.nanmean(eeg, axis=0, keepdims=True)
    eeg = notch_filter(
        eeg,
        fs=fs,
        notch_hz=preprocess_config.notch_hz,
        quality_factor=preprocess_config.notch_quality_factor,
    )
    eeg = bandpass_filter(
        eeg,
        fs=fs,
        low_hz=preprocess_config.bandpass_low_hz,
        high_hz=preprocess_config.bandpass_high_hz,
        order=preprocess_config.filter_order,
    )
    return eeg.astype(np.float32)


def preprocess_recording(
    raw_npz: PathLike,
    output_npz: PathLike,
    preprocess_config: PreprocessConfig,
    label_config: AudioLabelConfig,
) -> Tuple[Path, pd.DataFrame]:
    """Extract EEG/audio, detect cue onsets, create labels, and save a labeled NPZ."""

    raw = load_raw_recording(raw_npz)
    data = raw["data"]
    fs = int(raw["samplerate"])
    channels = tuple(raw["channels"])

    eeg_raw = extract_hardware_channels(data, channels, preprocess_config.eeg_channels)
    audio = extract_hardware_channels(data, channels, (preprocess_config.audio_channel,))[:, 0]

    eeg = preprocess_eeg_signal(eeg_raw, fs=fs, preprocess_config=preprocess_config)

    onset_info = detect_audio_onsets(audio, fs, label_config)
    sample_labels, cue_table = labels_from_audio_onsets(
        n_samples=eeg.shape[0],
        fs=fs,
        onset_info=onset_info,
        config=label_config,
    )

    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    cue_csv = output_npz.with_suffix(".cue_table.csv")
    cue_table.to_csv(cue_csv, index=False)

    np.savez(
        output_npz,
        eeg=eeg.astype(np.float32),
        eeg_raw=eeg_raw.astype(np.float32),
        audio=audio.astype(np.float32),
        audio_envelope=np.asarray(onset_info["envelope"], dtype=np.float64),
        audio_threshold=np.array(float(onset_info["threshold"]), dtype=np.float64),
        cue_onset_samples=np.asarray(onset_info["onset_samples"], dtype=np.int64),
        cue_peak_values=np.asarray(onset_info["peak_values"], dtype=np.float64),
        sample_labels=sample_labels.astype(np.int64),
        samplerate=np.array(fs, dtype=np.int64),
        acquired_channels=np.asarray(channels, dtype=np.int64),
        eeg_channels=np.asarray(preprocess_config.eeg_channels, dtype=np.int64),
        audio_channel=np.array(int(preprocess_config.audio_channel), dtype=np.int64),
        class_names=np.asarray(label_config.class_names),
        source_raw_npz=np.array(str(raw_npz)),
        segment_name=np.array(raw["segment_name"]),
        preprocess_config_json=np.array(json.dumps(_json_safe(asdict(preprocess_config)))),
        label_config_json=np.array(json.dumps(_json_safe(asdict(label_config)))),
        cue_table_csv=np.array(str(cue_csv)),
        label_convention=np.array("sample_labels indexes class_names"),
    )
    return output_npz, cue_table


def preprocess_many_recordings(
    raw_npz_paths: Sequence[PathLike],
    output_dir: PathLike,
    preprocess_config: PreprocessConfig,
    label_config: AudioLabelConfig,
) -> List[Path]:
    output_dir = ensure_dir(output_dir)
    out_paths: List[Path] = []
    for raw_path in raw_npz_paths:
        raw_path = Path(raw_path)
        out_path = output_dir / f"{raw_path.stem}_labeled.npz"
        preprocess_recording(raw_path, out_path, preprocess_config, label_config)
        out_paths.append(out_path)
    return out_paths


def load_labeled_recording(path: PathLike) -> Dict[str, Any]:
    labeled = np.load(Path(path), allow_pickle=True)
    required = {"eeg", "samplerate", "sample_labels"}
    missing = required.difference(set(labeled.files))
    if missing:
        raise KeyError(f"Labeled recording {path} is missing {sorted(missing)}.")
    eeg = _as_2d_samples_channels(labeled["eeg"])
    labels = np.asarray(labeled["sample_labels"], dtype=np.int64).reshape(-1)
    n = min(eeg.shape[0], len(labels))
    class_names = tuple(str(x) for x in labeled["class_names"]) if "class_names" in labeled.files else tuple()
    return {
        "eeg": eeg[:n].astype(np.float32),
        "sample_labels": labels[:n].astype(np.int64),
        "samplerate": int(np.asarray(labeled["samplerate"]).item()),
        "class_names": class_names,
        "path": str(path),
        "segment_name": str(np.asarray(labeled["segment_name"]).item()) if "segment_name" in labeled.files else Path(path).stem,
        "audio": np.asarray(labeled["audio"], dtype=np.float32).reshape(-1)[:n] if "audio" in labeled.files else None,
        "audio_envelope": np.asarray(labeled["audio_envelope"], dtype=np.float64).reshape(-1)[:n] if "audio_envelope" in labeled.files else None,
        "audio_threshold": float(np.asarray(labeled["audio_threshold"]).item()) if "audio_threshold" in labeled.files else None,
    }


# ---------------------------------------------------------------------------
# Windowing and features
# ---------------------------------------------------------------------------


def window_label(labels_window: np.ndarray, mode: str = "endpoint") -> int:
    labels_window = np.asarray(labels_window, dtype=np.int64).reshape(-1)
    if labels_window.size == 0:
        raise ValueError("labels_window cannot be empty.")
    mode = str(mode).lower()
    if mode == "endpoint":
        return int(labels_window[-1])
    if mode == "majority":
        counts = np.bincount(labels_window)
        return int(np.argmax(counts))
    raise ValueError("label_mode must be 'endpoint' or 'majority'.")


def fft_log_bandpower(window: np.ndarray, fs: int, band: Tuple[float, float], eps: float = 1e-12) -> np.ndarray:
    """Return one log-bandpower feature per channel."""

    x = _as_2d_samples_channels(window).astype(np.float32)
    if x.shape[0] < 2:
        raise ValueError("FFT bandpower requires at least two samples per window.")
    x = x - np.mean(x, axis=0, keepdims=True)
    taper = np.hanning(x.shape[0]).astype(np.float32)[:, None]
    xw = x * taper
    freqs = np.fft.rfftfreq(xw.shape[0], d=1.0 / int(fs))
    fft_vals = np.fft.rfft(xw, axis=0)
    power = np.abs(fft_vals) ** 2
    low, high = float(band[0]), float(band[1])
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        raise ValueError(f"No FFT bins inside band={band} for fs={fs} and window length={x.shape[0]}.")
    return np.log10(np.sum(power[mask, :], axis=0) + eps).astype(np.float32)


def extract_window_features(window: np.ndarray, fs: int, config: WindowConfig) -> np.ndarray:
    mode = str(config.feature_mode).lower()
    if mode == "raw_signal":
        return _as_2d_samples_channels(window).astype(np.float32)
    if mode == "fft_bandpower":
        bp = fft_log_bandpower(window, fs=fs, band=config.bandpower_hz)
        return bp[None, :].astype(np.float32)
    raise ValueError("feature_mode must be 'raw_signal' or 'fft_bandpower'.")


def make_labeled_windows(
    eeg: np.ndarray,
    sample_labels: np.ndarray,
    fs: int,
    config: WindowConfig,
    recording_id: str = "",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    eeg = _as_2d_samples_channels(eeg)
    labels = np.asarray(sample_labels, dtype=np.int64).reshape(-1)
    n = min(eeg.shape[0], labels.shape[0])
    eeg = eeg[:n]
    labels = labels[:n]

    win = int(round(float(config.window_sec) * int(fs)))
    stride = int(round(float(config.stride_sec) * int(fs)))
    if win <= 0 or stride <= 0:
        raise ValueError("window_sec and stride_sec must produce positive sample counts.")
    if n < win:
        raise ValueError(f"Not enough samples ({n}) for window length ({win}).")

    X: List[np.ndarray] = []
    y: List[int] = []
    rows: List[Dict[str, Any]] = []
    for start in range(0, n - win + 1, stride):
        end = start + win
        features = extract_window_features(eeg[start:end], int(fs), config)
        label = window_label(labels[start:end], mode=config.label_mode)
        X.append(features.astype(np.float32))
        y.append(label)
        rows.append(
            {
                "recording_id": recording_id,
                "start_sample": int(start),
                "end_sample": int(end),
                "start_time_sec": float(start) / float(fs),
                "end_time_sec": float(end) / float(fs),
                "label": int(label),
            }
        )

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), pd.DataFrame(rows)


def fit_normalizer(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_normalizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = np.where(np.asarray(std) < 1e-6, 1.0, std)
    return ((X - mean) / std).astype(np.float32)


def build_train_val_dataset(
    labeled_npz_paths: Sequence[PathLike],
    window_config: WindowConfig,
    training_config: TrainingConfig,
) -> DatasetBundle:
    """Make windows from one or more labeled 5-minute recordings."""

    if not labeled_npz_paths:
        raise ValueError("labeled_npz_paths cannot be empty.")

    X_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []
    tables: List[pd.DataFrame] = []
    fs_values = set()
    class_names: Tuple[str, ...] = tuple()

    for path in labeled_npz_paths:
        rec = load_labeled_recording(path)
        fs_values.add(int(rec["samplerate"]))
        if rec["class_names"]:
            class_names = tuple(rec["class_names"])
        X, y, table = make_labeled_windows(
            rec["eeg"],
            rec["sample_labels"],
            rec["samplerate"],
            window_config,
            recording_id=rec["segment_name"],
        )
        table["source_file"] = str(path)
        X_all.append(X)
        y_all.append(y)
        tables.append(table)

    if len(fs_values) != 1:
        raise ValueError(f"All recordings must use the same samplerate; found {sorted(fs_values)}.")

    X_cat = np.concatenate(X_all, axis=0)
    y_cat = np.concatenate(y_all, axis=0)
    table_cat = pd.concat(tables, ignore_index=True)

    n = len(y_cat)
    split = int(round(n * float(training_config.train_fraction)))
    split = max(1, min(n - 1, split))

    X_train_raw, y_train = X_cat[:split], y_cat[:split]
    X_val_raw, y_val = X_cat[split:], y_cat[split:]
    train_table = table_cat.iloc[:split].copy().reset_index(drop=True)
    val_table = table_cat.iloc[split:].copy().reset_index(drop=True)

    mean, std = fit_normalizer(X_train_raw)
    X_train = apply_normalizer(X_train_raw, mean, std)
    X_val = apply_normalizer(X_val_raw, mean, std)

    if not class_names:
        max_label = int(np.max(y_cat)) if len(y_cat) else 1
        class_names = tuple(f"class_{i}" for i in range(max_label + 1))

    return DatasetBundle(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        train_windows=train_table,
        val_windows=val_table,
        normalizer_mean=mean,
        normalizer_std=std,
        fs=int(next(iter(fs_values))),
        class_names=class_names,
        window_config=window_config,
        source_files=tuple(str(p) for p in labeled_npz_paths),
    )


# ---------------------------------------------------------------------------
# Model, training, validation
# ---------------------------------------------------------------------------


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class LSTMClassifier(nn.Module):
    """Minimal LSTM classifier for window-level class prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        effective_dropout = float(dropout) if int(num_layers) > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=int(input_size),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=effective_dropout,
        )
        self.fc = nn.Linear(int(hidden_size), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _device(training_config: TrainingConfig) -> torch.device:
    if training_config.device:
        return torch.device(training_config.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_array(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    loader = DataLoader(torch.as_tensor(X, dtype=torch.float32), batch_size=int(batch_size), shuffle=False)
    probs: List[np.ndarray] = []
    for xb in loader:
        xb = xb.to(device)
        logits = model(xb)
        probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    prob = np.concatenate(probs, axis=0) if probs else np.empty((0, 0), dtype=np.float32)
    pred = np.argmax(prob, axis=1).astype(np.int64) if len(prob) else np.empty(0, dtype=np.int64)
    return pred, prob.astype(np.float32)


def classification_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    n_classes = max(len(class_names), int(max(np.max(y_true), np.max(y_pred))) + 1 if len(y_true) else 0)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        if 0 <= truth < n_classes and 0 <= pred < n_classes:
            confusion[truth, pred] += 1

    rows = []
    recalls = []
    for label in range(n_classes):
        tp = int(confusion[label, label])
        support = int(confusion[label, :].sum())
        predicted = int(confusion[:, label].sum())
        recall = tp / support if support else np.nan
        precision = tp / predicted if predicted else np.nan
        recalls.append(recall)
        rows.append(
            {
                "label": label,
                "class_name": class_names[label] if label < len(class_names) else f"class_{label}",
                "support": support,
                "predicted": predicted,
                "precision": precision,
                "recall": recall,
            }
        )

    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) else np.nan
    finite_recalls = [r for r in recalls if np.isfinite(r)]
    balanced_accuracy = float(np.mean(finite_recalls)) if finite_recalls else np.nan

    summary = pd.DataFrame(
        [
            {
                "n_windows": int(len(y_true)),
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
            }
        ]
    )
    per_class = pd.DataFrame(rows)
    return summary, per_class


def train_lstm(
    bundle: DatasetBundle,
    training_config: TrainingConfig,
    output_dir: PathLike,
) -> Dict[str, Any]:
    """Train and validate the LSTM, then save checkpoint and CSV artifacts."""

    set_seed(int(training_config.seed))
    output_dir = ensure_dir(output_dir)
    device = _device(training_config)

    input_size = int(bundle.X_train.shape[-1])
    num_classes = int(max(len(bundle.class_names), np.max(bundle.y_train) + 1, np.max(bundle.y_val) + 1))
    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=int(training_config.hidden_size),
        num_layers=int(training_config.num_layers),
        num_classes=num_classes,
        dropout=float(training_config.dropout),
    ).to(device)

    train_loader = DataLoader(
        WindowDataset(bundle.X_train, bundle.y_train),
        batch_size=int(training_config.batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(training_config.seed)),
        num_workers=0,
    )
    val_loader = DataLoader(
        WindowDataset(bundle.X_val, bundle.y_val),
        batch_size=int(training_config.batch_size),
        shuffle=False,
        num_workers=0,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_config.lr))

    best_state = None
    best_val_acc = -np.inf
    history: List[Dict[str, Any]] = []

    for epoch in range(1, int(training_config.epochs) + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * int(yb.numel())
            total_correct += int((logits.argmax(dim=1) == yb).sum().item())
            total += int(yb.numel())

        train_loss = total_loss / max(total, 1)
        train_acc = total_correct / max(total, 1)

        val_loss, val_acc = _evaluate_loss_acc(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    val_pred, val_prob = predict_array(model, bundle.X_val, training_config.batch_size, device)
    val_summary, val_per_class = classification_summary(bundle.y_val, val_pred, bundle.class_names)

    history_df = pd.DataFrame(history)
    val_predictions = bundle.val_windows.copy()
    val_predictions["true_label"] = bundle.y_val
    val_predictions["pred_label"] = val_pred
    for label_idx in range(val_prob.shape[1]):
        name = bundle.class_names[label_idx] if label_idx < len(bundle.class_names) else f"class_{label_idx}"
        val_predictions[f"prob_{name}"] = val_prob[:, label_idx]

    checkpoint_path = output_dir / "lstm_checkpoint.pt"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_size": input_size,
            "hidden_size": int(training_config.hidden_size),
            "num_layers": int(training_config.num_layers),
            "num_classes": num_classes,
            "dropout": float(training_config.dropout),
        },
        "window_config": asdict(bundle.window_config),
        "training_config": asdict(training_config),
        "normalizer_mean": bundle.normalizer_mean,
        "normalizer_std": bundle.normalizer_std,
        "fs": int(bundle.fs),
        "class_names": tuple(bundle.class_names),
        "source_files": tuple(bundle.source_files),
        "final_val_summary": val_summary.to_dict(orient="records"),
        "history": history,
    }
    torch.save(checkpoint, checkpoint_path)

    history_csv = output_dir / "training_history.csv"
    val_predictions_csv = output_dir / "validation_predictions.csv"
    val_summary_csv = output_dir / "validation_summary.csv"
    val_per_class_csv = output_dir / "validation_per_class.csv"
    metadata_json = output_dir / "checkpoint_metadata.json"

    history_df.to_csv(history_csv, index=False)
    val_predictions.to_csv(val_predictions_csv, index=False)
    val_summary.to_csv(val_summary_csv, index=False)
    val_per_class.to_csv(val_per_class_csv, index=False)
    save_json({k: v for k, v in checkpoint.items() if k != "model_state_dict"}, metadata_json)

    return {
        "model": model,
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "history": history_df,
        "validation_predictions": val_predictions,
        "validation_summary": val_summary,
        "validation_per_class": val_per_class,
    }


@torch.no_grad()
def _evaluate_loss_acc(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += float(loss.item()) * int(yb.numel())
        total_correct += int((logits.argmax(dim=1) == yb).sum().item())
        total += int(yb.numel())
    return total_loss / max(total, 1), total_correct / max(total, 1)


def train_validate_pipeline(
    labeled_npz_paths: Sequence[PathLike],
    output_dir: PathLike,
    window_config: WindowConfig,
    training_config: TrainingConfig,
) -> Dict[str, Any]:
    bundle = build_train_val_dataset(labeled_npz_paths, window_config, training_config)
    result = train_lstm(bundle, training_config, output_dir)
    result["dataset_bundle"] = bundle
    return result


def load_checkpoint(checkpoint_path: PathLike, device: Optional[Union[str, torch.device]] = None) -> Tuple[nn.Module, Dict[str, Any], torch.device]:
    checkpoint_path = Path(checkpoint_path)
    device_obj = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device_obj)
    model = LSTMClassifier(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device_obj)
    model.eval()
    return model, checkpoint, device_obj


def window_config_from_checkpoint(checkpoint: Dict[str, Any]) -> WindowConfig:
    cfg = dict(checkpoint["window_config"])
    cfg["bandpower_hz"] = tuple(cfg.get("bandpower_hz", (18.0, 22.0)))
    return WindowConfig(**cfg)


def predict_labeled_recording(
    labeled_npz: PathLike,
    checkpoint_path: PathLike,
    output_dir: PathLike,
    batch_size: int = 256,
) -> Dict[str, Any]:
    """Run a saved checkpoint on one labeled arbitrary-length trial."""

    output_dir = ensure_dir(output_dir)
    rec = load_labeled_recording(labeled_npz)
    model, checkpoint, device = load_checkpoint(checkpoint_path)
    win_cfg = window_config_from_checkpoint(checkpoint)

    if int(rec["samplerate"]) != int(checkpoint["fs"]):
        raise ValueError(
            f"Samplerate mismatch: recording fs={rec['samplerate']}, checkpoint fs={checkpoint['fs']}."
        )

    X_raw, y_true, windows = make_labeled_windows(
        rec["eeg"],
        rec["sample_labels"],
        rec["samplerate"],
        win_cfg,
        recording_id=rec["segment_name"],
    )
    X = apply_normalizer(X_raw, checkpoint["normalizer_mean"], checkpoint["normalizer_std"])
    pred, prob = predict_array(model, X, batch_size=batch_size, device=device)

    class_names = tuple(str(x) for x in checkpoint.get("class_names", rec["class_names"]))
    summary, per_class = classification_summary(y_true, pred, class_names)

    pred_df = windows.copy()
    pred_df["true_label"] = y_true
    pred_df["pred_label"] = pred
    pred_df["correct"] = pred_df["true_label"].astype(int) == pred_df["pred_label"].astype(int)
    for label_idx in range(prob.shape[1]):
        name = class_names[label_idx] if label_idx < len(class_names) else f"class_{label_idx}"
        pred_df[f"prob_{name}"] = prob[:, label_idx]

    delay_df = estimate_transition_delay(
        pred_df,
        sample_labels=rec["sample_labels"],
        fs=int(rec["samplerate"]),
    )
    delay_summary = summarize_transition_delay(delay_df)

    stem = Path(labeled_npz).stem
    pred_csv = output_dir / f"{stem}_test_predictions.csv"
    summary_csv = output_dir / f"{stem}_test_summary.csv"
    per_class_csv = output_dir / f"{stem}_test_per_class.csv"
    delay_csv = output_dir / f"{stem}_test_delay_by_transition.csv"
    delay_summary_csv = output_dir / f"{stem}_test_delay_summary.csv"

    pred_df.to_csv(pred_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    per_class.to_csv(per_class_csv, index=False)
    delay_df.to_csv(delay_csv, index=False)
    delay_summary.to_csv(delay_summary_csv, index=False)

    return {
        "predictions": pred_df,
        "summary": summary,
        "per_class": per_class,
        "delay": delay_df,
        "delay_summary": delay_summary,
        "prediction_csv": pred_csv,
        "summary_csv": summary_csv,
        "per_class_csv": per_class_csv,
        "delay_csv": delay_csv,
        "delay_summary_csv": delay_summary_csv,
    }


def collect_preprocess_and_test_trial(
    output_dir: PathLike,
    checkpoint_path: PathLike,
    acquisition_config: AcquisitionConfig,
    preprocess_config: PreprocessConfig,
    label_config: AudioLabelConfig,
    duration_sec: float = 60.0,
    trial_name: str = "test_trial",
) -> Dict[str, Any]:
    """Collect an arbitrary-length trial, label it from audio, and test it."""

    output_dir = ensure_dir(output_dir)
    raw_path = output_dir / f"{trial_name}_raw.npz"
    labeled_path = output_dir / f"{trial_name}_labeled.npz"
    collect_mp150_recording(raw_path, duration_sec, acquisition_config, segment_name=trial_name)
    preprocess_recording(raw_path, labeled_path, preprocess_config, label_config)
    result = predict_labeled_recording(labeled_path, checkpoint_path, output_dir=output_dir)
    result["raw_path"] = raw_path
    result["labeled_path"] = labeled_path
    return result


# ---------------------------------------------------------------------------
# Transition-delay metrics
# ---------------------------------------------------------------------------


def estimate_transition_delay(
    prediction_df: pd.DataFrame,
    sample_labels: np.ndarray,
    fs: int,
    max_delay_sec: Optional[float] = 5.0,
) -> pd.DataFrame:
    labels = np.asarray(sample_labels, dtype=np.int64).reshape(-1)
    if len(labels) < 2 or len(prediction_df) == 0:
        return pd.DataFrame()

    true_change_samples = np.flatnonzero(np.diff(labels) != 0) + 1
    true_change_times = true_change_samples / float(fs)
    pred_df = prediction_df.sort_values("end_time_sec").reset_index(drop=True)
    pred_times = pred_df["end_time_sec"].to_numpy(dtype=float)
    pred_labels = pred_df["pred_label"].astype(int).to_numpy()

    pred_change_idx = np.flatnonzero(np.diff(pred_labels) != 0) + 1
    pred_change_times = pred_times[pred_change_idx]
    pred_change_to_labels = pred_labels[pred_change_idx]

    rows = []
    for i, sample_idx in enumerate(true_change_samples):
        true_t = float(true_change_times[i])
        from_label = int(labels[sample_idx - 1])
        to_label = int(labels[sample_idx])
        next_true_t = float(true_change_times[i + 1]) if i + 1 < len(true_change_times) else np.inf
        latest_t = next_true_t
        if max_delay_sec is not None:
            latest_t = min(latest_t, true_t + float(max_delay_sec))

        post_mask = (pred_times >= true_t) & (pred_times <= latest_t)
        post_idx = np.flatnonzero(post_mask)
        correct_idx = post_idx[pred_labels[post_idx] == to_label] if len(post_idx) else np.array([], dtype=int)

        if len(correct_idx):
            first_correct_time = float(pred_times[int(correct_idx[0])])
            delay_first_correct = first_correct_time - true_t
        else:
            first_correct_time = np.nan
            delay_first_correct = np.nan

        transition_mask = (
            (pred_change_times >= true_t)
            & (pred_change_times <= latest_t)
            & (pred_change_to_labels == to_label)
        )
        transition_idx = np.flatnonzero(transition_mask)
        if len(transition_idx):
            predicted_transition_time = float(pred_change_times[int(transition_idx[0])])
            delay_pred_transition = predicted_transition_time - true_t
        else:
            predicted_transition_time = np.nan
            delay_pred_transition = np.nan

        rows.append(
            {
                "true_transition_sample": int(sample_idx),
                "true_transition_time_sec": true_t,
                "from_label": from_label,
                "to_label": to_label,
                "transition_type": f"{from_label}->{to_label}",
                "first_correct_prediction_time_sec": first_correct_time,
                "delay_to_first_correct_prediction_sec": delay_first_correct,
                "predicted_transition_time_sec": predicted_transition_time,
                "delay_to_predicted_transition_sec": delay_pred_transition,
                "matched_first_correct_prediction": bool(np.isfinite(delay_first_correct)),
                "matched_predicted_transition": bool(np.isfinite(delay_pred_transition)),
            }
        )

    return pd.DataFrame(rows)


def summarize_transition_delay(delay_df: pd.DataFrame) -> pd.DataFrame:
    if delay_df is None or len(delay_df) == 0:
        return pd.DataFrame(
            [
                {
                    "n_true_transitions": 0,
                    "n_matched_first_correct": 0,
                    "mean_delay_to_first_correct_sec": np.nan,
                    "median_delay_to_first_correct_sec": np.nan,
                    "n_matched_predicted_transition": 0,
                    "mean_delay_to_predicted_transition_sec": np.nan,
                    "median_delay_to_predicted_transition_sec": np.nan,
                }
            ]
        )

    first = delay_df["delay_to_first_correct_prediction_sec"].dropna().to_numpy(dtype=float)
    transition = delay_df["delay_to_predicted_transition_sec"].dropna().to_numpy(dtype=float)
    return pd.DataFrame(
        [
            {
                "n_true_transitions": int(len(delay_df)),
                "n_matched_first_correct": int(delay_df["matched_first_correct_prediction"].sum()),
                "mean_delay_to_first_correct_sec": float(np.mean(first)) if len(first) else np.nan,
                "median_delay_to_first_correct_sec": float(np.median(first)) if len(first) else np.nan,
                "n_matched_predicted_transition": int(delay_df["matched_predicted_transition"].sum()),
                "mean_delay_to_predicted_transition_sec": float(np.mean(transition)) if len(transition) else np.nan,
                "median_delay_to_predicted_transition_sec": float(np.median(transition)) if len(transition) else np.nan,
            }
        ]
    )


# ---------------------------------------------------------------------------
# Plotting helpers used by the notebook
# ---------------------------------------------------------------------------


def plot_labeled_recording(labeled_npz: PathLike, max_duration_sec: Optional[float] = 30.0):
    import matplotlib.pyplot as plt

    rec = load_labeled_recording(labeled_npz)
    eeg = rec["eeg"]
    labels = rec["sample_labels"]
    fs = int(rec["samplerate"])
    n = len(labels)
    if max_duration_sec is not None:
        n = min(n, int(round(float(max_duration_sec) * fs)))
    t = np.arange(n) / float(fs)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, eeg[:n, 0], linewidth=0.8, label="EEG channel 1")
    ax.step(t, labels[:n], where="post", linewidth=1.0, label="sample label")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude / label")
    ax.set_title(f"Labeled recording: {Path(labeled_npz).name}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    return fig, ax


def plot_predictions_overlay(
    labeled_npz: PathLike,
    predictions: Union[pd.DataFrame, PathLike],
    max_duration_sec: Optional[float] = None,
):
    import matplotlib.pyplot as plt

    rec = load_labeled_recording(labeled_npz)
    pred_df = pd.read_csv(predictions) if not isinstance(predictions, pd.DataFrame) else predictions.copy()
    eeg = rec["eeg"]
    labels = rec["sample_labels"]
    fs = int(rec["samplerate"])

    n = len(labels)
    if max_duration_sec is not None:
        n = min(n, int(round(float(max_duration_sec) * fs)))
        pred_df = pred_df[pred_df["end_time_sec"] <= float(max_duration_sec)].copy()

    t = np.arange(n) / float(fs)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, eeg[:n, 0], linewidth=0.7, label="EEG channel 1")
    ax.step(t, labels[:n], where="post", linewidth=1.0, label="true label")
    if len(pred_df):
        t_pred = pred_df["end_time_sec"].to_numpy(dtype=float)
        pred = pred_df["pred_label"].to_numpy(dtype=float)
        ax.step(t_pred, pred, where="post", linewidth=1.8, label="predicted label")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude / label")
    ax.set_title(f"Predictions: {Path(labeled_npz).name}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    return fig, ax
