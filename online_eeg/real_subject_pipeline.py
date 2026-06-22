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
import sys
import threading
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
    """Sliding-window and feature extraction settings."""

    feature_mode: str = "filtered_signal"  # filtered_signal = non-FFT filtered windows; fft_bandpower = FFT features
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


def _print_recording_elapsed(
    start_wall: float,
    duration_sec: float,
    stop_event: threading.Event,
    interval_sec: float = 1.0,
) -> None:
    while not stop_event.wait(float(interval_sec)):
        elapsed = max(0.0, time.time() - float(start_wall))
        shown_elapsed = min(elapsed, float(duration_sec))
        sys.stdout.write(
            f"\rElapsed since recording start: {shown_elapsed:6.1f}s / {float(duration_sec):.1f}s"
        )
        sys.stdout.flush()


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
    channel order in ``config.channels``. Fixed-duration recordings are one
    hardware chunk whose length is ``duration_sec``.
    """

    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    sample_rate = int(config.samplerate)
    channels = tuple(int(ch) for ch in config.channels)
    if not channels:
        raise ValueError("config.channels must contain at least one channel.")
    target_samples = _duration_to_sample_count(duration_sec, sample_rate)

    MP150 = _import_mp150_class()
    mp = MP150(samplerate=sample_rate, channels=list(channels))

    chunk_start_wall: List[float] = []
    chunk_end_wall: List[float] = []

    start_wall = time.time()
    segment_label = str(segment_name).strip() or output_npz.stem
    print(
        f"Recording started: {segment_label} at {_now_string()} "
        f"({float(duration_sec):.1f}s, channels={channels})",
        flush=True,
    )
    stop_progress = threading.Event()
    progress_thread = threading.Thread(
        target=_print_recording_elapsed,
        args=(start_wall, float(duration_sec), stop_progress),
        daemon=True,
    )
    progress_thread.start()
    try:
        chunk_start_wall.append(time.time() - start_wall)
        data = _checked_acquisition_chunk(mp.get_chunk(float(duration_sec)), len(channels))
        chunk_end_wall.append(time.time() - start_wall)
    finally:
        stop_progress.set()
        progress_thread.join(timeout=2.0)
        elapsed = max(0.0, time.time() - start_wall)
        sys.stdout.write(
            f"\rElapsed since recording start: {min(elapsed, float(duration_sec)):6.1f}s / "
            f"{float(duration_sec):.1f}s\n"
        )
        sys.stdout.flush()
        mp.close()
        print(
            f"Recording ended: {segment_label} at {_now_string()} "
            f"(elapsed {elapsed:.1f}s)",
            flush=True,
        )

    if data.shape[0] < target_samples:
        raise RuntimeError(
            f"Expected one {duration_sec:.3f}s chunk with {target_samples} samples, "
            f"got {data.shape[0]} samples."
        )
    if data.shape[0] > target_samples:
        data = data[:target_samples]
    data = data.astype(np.float32)

    time_sec = np.arange(data.shape[0], dtype=np.float64) / float(sample_rate)
    np.savez(
        output_npz,
        data=data,
        samplerate=np.array(sample_rate, dtype=np.int64),
        channels=np.asarray(channels, dtype=np.int64),
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
        print(
            f"Collecting training segment {len(paths) + 1}/{len(segment_names)}: {safe_name}",
            flush=True,
        )
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
    elif config.alternate_binary_labels:
        seq = [int(config.active_label), int(config.baseline_label)]
        cue_labels = np.asarray([seq[i % 2] for i in range(len(onsets))], dtype=np.int64)
    else:
        cue_labels = np.full(len(onsets), int(config.active_label), dtype=np.int64)

    start_offset = int(round(float(config.label_start_offset_sec) * fs))
    transition_mode = config.label_duration_sec is None
    fixed_duration = None
    if not transition_mode:
        fixed_duration = max(1, int(round(float(config.label_duration_sec) * fs)))

    rows: List[Dict[str, Any]] = []
    for i, onset in enumerate(onsets):
        start = int(np.clip(int(onset) + start_offset, 0, n_samples))
        if transition_mode:
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
    if not bool(getattr(preprocess_config, "apply_software_filters", True)):
        return eeg.astype(np.float32)
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
        label_convention=np.array("sample_labels indexes class_names; cue onsets are state transitions when label_duration_sec is None"),
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
    fs = int(np.asarray(labeled["samplerate"]).item())
    cue_onset_samples = (
        np.asarray(labeled["cue_onset_samples"], dtype=np.int64).reshape(-1)
        if "cue_onset_samples" in labeled.files
        else np.empty(0, dtype=np.int64)
    )
    cue_onset_samples = cue_onset_samples[
        (cue_onset_samples >= 0) & (cue_onset_samples < n)
    ]
    class_names = tuple(str(x) for x in labeled["class_names"]) if "class_names" in labeled.files else tuple()
    return {
        "eeg": eeg[:n].astype(np.float32),
        "eeg_raw": (
            _as_2d_samples_channels(labeled["eeg_raw"])[:n].astype(np.float32)
            if "eeg_raw" in labeled.files
            else None
        ),
        "sample_labels": labels[:n].astype(np.int64),
        "samplerate": fs,
        "class_names": class_names,
        "path": str(path),
        "segment_name": str(np.asarray(labeled["segment_name"]).item()) if "segment_name" in labeled.files else Path(path).stem,
        "audio": np.asarray(labeled["audio"], dtype=np.float32).reshape(-1)[:n] if "audio" in labeled.files else None,
        "audio_envelope": np.asarray(labeled["audio_envelope"], dtype=np.float64).reshape(-1)[:n] if "audio_envelope" in labeled.files else None,
        "audio_threshold": float(np.asarray(labeled["audio_threshold"]).item()) if "audio_threshold" in labeled.files else None,
        "cue_onset_samples": cue_onset_samples,
        "cue_onset_times_sec": cue_onset_samples.astype(np.float64) / float(fs),
        "acquired_channels": (
            tuple(int(x) for x in np.asarray(labeled["acquired_channels"]).reshape(-1))
            if "acquired_channels" in labeled.files
            else tuple()
        ),
        "eeg_channels": (
            tuple(int(x) for x in np.asarray(labeled["eeg_channels"]).reshape(-1))
            if "eeg_channels" in labeled.files
            else tuple()
        ),
        "audio_channel": int(np.asarray(labeled["audio_channel"]).item()) if "audio_channel" in labeled.files else None,
    }


def labeled_preprocess_summary(
    labeled_npz_paths: Union[PathLike, Sequence[PathLike], Dict[str, PathLike]],
) -> pd.DataFrame:
    """Summarize preprocessing metadata saved inside labeled NPZ files."""

    if isinstance(labeled_npz_paths, dict):
        items = [(str(name), Path(path)) for name, path in labeled_npz_paths.items()]
    elif isinstance(labeled_npz_paths, (str, os.PathLike)):
        path = Path(labeled_npz_paths)
        items = [(path.stem, path)]
    else:
        items = [(Path(path).stem, Path(path)) for path in labeled_npz_paths]

    rows: List[Dict[str, Any]] = []
    for name, path in items:
        labeled = np.load(path, allow_pickle=True)
        try:
            files = set(labeled.files)
            preprocess_config: Dict[str, Any] = {}
            if "preprocess_config_json" in files:
                raw_json = str(np.asarray(labeled["preprocess_config_json"]).item())
                preprocess_config = json.loads(raw_json) if raw_json else {}

            fs = int(np.asarray(labeled["samplerate"]).item()) if "samplerate" in files else np.nan
            n_samples = int(labeled["eeg"].shape[0]) if "eeg" in files else np.nan
            duration_sec = float(n_samples) / float(fs) if np.isfinite(fs) and fs else np.nan
            has_apply_flag = "apply_software_filters" in preprocess_config
            apply_software_filters = preprocess_config.get(
                "apply_software_filters",
                True if preprocess_config else np.nan,
            )
            rows.append(
                {
                    "name": name,
                    "path": str(path),
                    "samplerate": fs,
                    "duration_sec": duration_sec,
                    "has_preprocess_config": bool(preprocess_config),
                    "has_apply_software_filters_flag": bool(has_apply_flag),
                    "apply_software_filters": apply_software_filters,
                    "demean_channels": preprocess_config.get("demean_channels", np.nan),
                    "bandpass_low_hz": preprocess_config.get("bandpass_low_hz", np.nan),
                    "bandpass_high_hz": preprocess_config.get("bandpass_high_hz", np.nan),
                    "notch_hz": preprocess_config.get("notch_hz", np.nan),
                    "eeg_channels": preprocess_config.get("eeg_channels", np.nan),
                    "audio_channel": preprocess_config.get("audio_channel", np.nan),
                    "source_raw_npz": (
                        str(np.asarray(labeled["source_raw_npz"]).item())
                        if "source_raw_npz" in files
                        else ""
                    ),
                }
            )
        finally:
            labeled.close()

    return pd.DataFrame(rows)

def _cue_onset_times_for_plot(rec: Dict[str, Any], max_duration_sec: Optional[float]) -> np.ndarray:
    fs = int(rec["samplerate"])
    samples = np.asarray(rec.get("cue_onset_samples", []), dtype=np.int64).reshape(-1)
    cue_times = samples.astype(np.float64) / float(fs)
    if max_duration_sec is not None:
        cue_times = cue_times[cue_times <= float(max_duration_sec)]
    return cue_times


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


def canonical_feature_mode(feature_mode: str) -> str:
    """Return the canonical feature-mode name.

    ``raw_signal`` is accepted only as a backward-compatible alias for older
    checkpoints and sweep outputs. The data are preprocessed before windowing.
    """

    mode = str(feature_mode).lower()
    if mode == "raw_signal":
        return "filtered_signal"
    if mode in {"filtered_signal", "fft_bandpower"}:
        return mode
    raise ValueError("feature_mode must be 'filtered_signal' or 'fft_bandpower'.")


def extract_window_features(window: np.ndarray, fs: int, config: WindowConfig) -> np.ndarray:
    """Extract features from an already preprocessed EEG window.

    ``filtered_signal`` keeps the filtered time-domain samples and skips FFT
    feature extraction.
    """

    mode = canonical_feature_mode(config.feature_mode)
    if mode == "filtered_signal":
        return _as_2d_samples_channels(window).astype(np.float32)
    if mode == "fft_bandpower":
        bp = fft_log_bandpower(window, fs=fs, band=config.bandpower_hz)
        return bp[None, :].astype(np.float32)
    raise ValueError("feature_mode must be 'filtered_signal' or 'fft_bandpower'.")


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




@torch.no_grad()
def predict_single_window(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict one realtime window without DataLoader overhead."""

    model.eval()
    xb = torch.as_tensor(X, dtype=torch.float32).to(device)
    if xb.ndim == 2:
        xb = xb.unsqueeze(0)
    logits = model(xb)
    prob = torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32)
    pred = np.argmax(prob, axis=1).astype(np.int64)
    return pred, prob


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


def slugify_config_value(value: Any) -> str:
    text = str(value).strip().lower()
    replacements = {
        " ": "_",
        ".": "p",
        "-": "m",
        "(": "",
        ")": "",
        "[": "",
        "]": "",
        ",": "_",
        ":": "_",
        "/": "_",
        "\\": "_",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or "value"


def offline_train_test_sweep(
    train_labeled_npz: PathLike,
    test_labeled_npz: PathLike,
    output_dir: PathLike,
    feature_modes: Sequence[str] = ("filtered_signal", "fft_bandpower"),
    window_secs: Sequence[float] = (1.0, 1.5, 2.0),
    stride_secs: Sequence[float] = (0.2,),
    bandpower_hz_values: Sequence[Tuple[float, float]] = ((8.0, 35.0),),
    training_config: Optional[TrainingConfig] = None,
    label_mode: str = "endpoint",
) -> Dict[str, Any]:
    """Train offline model variants and test each on a labeled realtime trial."""

    output_dir = ensure_dir(output_dir)
    training_config = training_config or TrainingConfig()
    label_mode = str(label_mode).lower()
    if label_mode not in {"endpoint", "majority"}:
        raise ValueError("label_mode must be 'endpoint' or 'majority'.")
    rows: List[Dict[str, Any]] = []
    result_dirs: List[Path] = []

    for feature_mode in feature_modes:
        feature_mode = canonical_feature_mode(str(feature_mode))
        bands = bandpower_hz_values if feature_mode == "fft_bandpower" else ((0.0, 0.0),)
        for window_sec in window_secs:
            for stride_sec in stride_secs:
                for bandpower_hz in bands:
                    win_cfg = WindowConfig(
                        feature_mode=feature_mode,
                        window_sec=float(window_sec),
                        stride_sec=float(stride_sec),
                        label_mode=label_mode,
                        bandpower_hz=tuple(float(x) for x in bandpower_hz),
                    )
                    parts = [
                        feature_mode,
                        f"win_{slugify_config_value(window_sec)}s",
                        f"stride_{slugify_config_value(stride_sec)}s",
                        f"labels_{slugify_config_value(label_mode)}",
                    ]
                    if feature_mode == "fft_bandpower":
                        parts.append(f"band_{slugify_config_value(win_cfg.bandpower_hz[0])}_{slugify_config_value(win_cfg.bandpower_hz[1])}hz")
                    variant_name = "__".join(parts)
                    variant_dir = ensure_dir(output_dir / variant_name)

                    train_result = train_validate_pipeline(
                        labeled_npz_paths=[train_labeled_npz],
                        output_dir=variant_dir,
                        window_config=win_cfg,
                        training_config=training_config,
                    )
                    test_result = predict_labeled_recording(
                        labeled_npz=test_labeled_npz,
                        checkpoint_path=train_result["checkpoint_path"],
                        output_dir=variant_dir,
                        batch_size=training_config.batch_size,
                    )

                    val_summary = train_result["validation_summary"].iloc[0].to_dict()
                    test_summary = test_result["summary"].iloc[0].to_dict()
                    cue_delay_summary = test_result["cue_delay_summary"].iloc[0].to_dict()
                    xcov_delay_summary = test_result["xcov_delay_summary"].iloc[0].to_dict()
                    row = {
                        "variant": variant_name,
                        "feature_mode": win_cfg.feature_mode,
                        "window_sec": win_cfg.window_sec,
                        "stride_sec": win_cfg.stride_sec,
                        "bandpower_low_hz": win_cfg.bandpower_hz[0] if win_cfg.feature_mode == "fft_bandpower" else np.nan,
                        "bandpower_high_hz": win_cfg.bandpower_hz[1] if win_cfg.feature_mode == "fft_bandpower" else np.nan,
                        "checkpoint_path": str(train_result["checkpoint_path"]),
                        "variant_dir": str(variant_dir),
                        "val_accuracy": val_summary.get("accuracy", np.nan),
                        "val_balanced_accuracy": val_summary.get("balanced_accuracy", np.nan),
                        "test_accuracy": test_summary.get("accuracy", np.nan),
                        "test_balanced_accuracy": test_summary.get("balanced_accuracy", np.nan),
                        "test_n_windows": test_summary.get("n_windows", np.nan),
                        "test_mean_cue_to_first_correct_sec": cue_delay_summary.get("mean_cue_to_first_correct_sec", np.nan),
                        "test_median_cue_to_first_correct_sec": cue_delay_summary.get("median_cue_to_first_correct_sec", np.nan),
                        "test_mean_cue_to_predicted_transition_sec": cue_delay_summary.get("mean_cue_to_predicted_transition_sec", np.nan),
                        "test_median_cue_to_predicted_transition_sec": cue_delay_summary.get("median_cue_to_predicted_transition_sec", np.nan),
                        "test_mean_cue_to_sustained_prediction_sec": cue_delay_summary.get("mean_cue_to_sustained_prediction_sec", np.nan),
                        "test_median_cue_to_sustained_prediction_sec": cue_delay_summary.get("median_cue_to_sustained_prediction_sec", np.nan),
                        "test_xcov_delay_sec": xcov_delay_summary.get("xcov_delay_sec", np.nan),
                        "test_xcov_peak_coeff": xcov_delay_summary.get("xcov_peak_coeff", np.nan),
                        "test_xcov_signal_column": xcov_delay_summary.get("prediction_signal_column", ""),
                    }
                    rows.append(row)
                    result_dirs.append(variant_dir)

    summary_df = pd.DataFrame(rows)
    summary_csv = output_dir / "offline_sweep_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    return {
        "summary": summary_df,
        "summary_csv": summary_csv,
        "result_dirs": result_dirs,
    }



def load_checkpoint(
    checkpoint_path: PathLike,
    device: Optional[str] = None,
) -> Tuple[LSTMClassifier, Dict[str, Any], torch.device]:
    """Load a saved LSTM checkpoint and return the model, metadata, and device."""

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

def class_names_from_checkpoint(
    checkpoint: Dict[str, Any],
    fallback: Sequence[str] = (),
) -> Tuple[str, ...]:
    """Return class names from new or original signal_generator checkpoints."""

    raw = checkpoint.get("class_names", None)
    if isinstance(raw, dict):
        def _sort_key(item):
            key, _ = item
            try:
                return int(key)
            except Exception:
                return str(key)

        names = tuple(str(value) for _, value in sorted(raw.items(), key=_sort_key))
    elif raw is not None:
        names = tuple(str(x) for x in np.asarray(raw).reshape(-1).tolist())
    else:
        names = tuple(str(x) for x in fallback)

    num_classes = int(checkpoint.get("model_config", {}).get("num_classes", len(names)))
    if len(names) < num_classes:
        names = names + tuple(f"class_{i}" for i in range(len(names), num_classes))
    return names

def window_config_from_checkpoint(checkpoint: Dict[str, Any]) -> WindowConfig:
    """Return window settings from new or original signal_generator checkpoints."""

    if "window_config" in checkpoint:
        cfg = dict(checkpoint["window_config"])
        cfg["feature_mode"] = canonical_feature_mode(cfg.get("feature_mode", "filtered_signal"))
        cfg["bandpower_hz"] = tuple(cfg.get("bandpower_hz", (18.0, 22.0)))
        return WindowConfig(**cfg)

    saved_cfg = dict(checkpoint.get("config", {}))
    fs = int(checkpoint.get("fs", saved_cfg.get("fs", 200)))
    feature_mode = canonical_feature_mode(checkpoint.get("feature_mode", saved_cfg.get("feature_mode", "filtered_signal")))
    bandpower_hz = tuple(checkpoint.get("bandpower_hz", saved_cfg.get("bandpower_hz", (18.0, 22.0))))

    if "window_sec" in saved_cfg:
        window_sec = float(saved_cfg["window_sec"])
    elif "window_samples" in checkpoint:
        window_sec = float(checkpoint["window_samples"]) / float(fs)
    else:
        window_sec = 1.0

    if "stride_sec" in saved_cfg:
        stride_sec = float(saved_cfg["stride_sec"])
    elif "stride_samples" in checkpoint:
        stride_sec = float(checkpoint["stride_samples"]) / float(fs)
    else:
        stride_sec = 0.25

    label_mode = checkpoint.get("label_mode", saved_cfg.get("label_mode", "endpoint"))
    return WindowConfig(
        feature_mode=str(feature_mode),
        window_sec=window_sec,
        stride_sec=stride_sec,
        label_mode=str(label_mode),
        bandpower_hz=bandpower_hz,
    )


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

    class_names = class_names_from_checkpoint(checkpoint, fallback=rec["class_names"])
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
    cue_delay_df = estimate_cue_prediction_delay(
        pred_df,
        sample_labels=rec["sample_labels"],
        cue_onset_samples=rec.get("cue_onset_samples", np.empty(0, dtype=np.int64)),
        fs=int(rec["samplerate"]),
        class_names=class_names,
    )
    cue_delay_summary = summarize_cue_prediction_delay(cue_delay_df)
    xcov_delay_summary, xcov_curve = estimate_prediction_xcov_delay(
        pred_df,
        sample_labels=rec["sample_labels"],
        fs=int(rec["samplerate"]),
        class_names=class_names,
        target_label=1,
    )

    stem = Path(labeled_npz).stem
    pred_csv = output_dir / f"{stem}_test_predictions.csv"
    summary_csv = output_dir / f"{stem}_test_summary.csv"
    per_class_csv = output_dir / f"{stem}_test_per_class.csv"
    delay_csv = output_dir / f"{stem}_test_delay_by_transition.csv"
    delay_summary_csv = output_dir / f"{stem}_test_delay_summary.csv"
    cue_delay_csv = output_dir / f"{stem}_test_cue_delay_by_cue.csv"
    cue_delay_summary_csv = output_dir / f"{stem}_test_cue_delay_summary.csv"
    xcov_delay_summary_csv = output_dir / f"{stem}_test_xcov_delay_summary.csv"
    xcov_curve_csv = output_dir / f"{stem}_test_xcov_curve.csv"

    pred_df.to_csv(pred_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    per_class.to_csv(per_class_csv, index=False)
    delay_df.to_csv(delay_csv, index=False)
    delay_summary.to_csv(delay_summary_csv, index=False)
    cue_delay_df.to_csv(cue_delay_csv, index=False)
    cue_delay_summary.to_csv(cue_delay_summary_csv, index=False)
    xcov_delay_summary.to_csv(xcov_delay_summary_csv, index=False)
    xcov_curve.to_csv(xcov_curve_csv, index=False)

    return {
        "predictions": pred_df,
        "summary": summary,
        "per_class": per_class,
        "delay": delay_df,
        "delay_summary": delay_summary,
        "cue_delay": cue_delay_df,
        "cue_delay_summary": cue_delay_summary,
        "xcov_delay_summary": xcov_delay_summary,
        "xcov_curve": xcov_curve,
        "prediction_csv": pred_csv,
        "summary_csv": summary_csv,
        "per_class_csv": per_class_csv,
        "delay_csv": delay_csv,
        "delay_summary_csv": delay_summary_csv,
        "cue_delay_csv": cue_delay_csv,
        "cue_delay_summary_csv": cue_delay_summary_csv,
        "xcov_delay_summary_csv": xcov_delay_summary_csv,
        "xcov_curve_csv": xcov_curve_csv,
    }


def _prediction_probability_columns(prob: np.ndarray, class_names: Sequence[str]) -> Dict[str, np.ndarray]:
    columns: Dict[str, np.ndarray] = {}
    for label_idx in range(prob.shape[1]):
        name = class_names[label_idx] if label_idx < len(class_names) else f"class_{label_idx}"
        columns[f"prob_{name}"] = prob[:, label_idx]
    return columns


def _format_realtime_prediction_status(row: Dict[str, Any], class_names: Sequence[str]) -> str:
    pred_label = int(row["pred_label"])
    pred_name = class_names[pred_label] if 0 <= pred_label < len(class_names) else f"class_{pred_label}"
    probability_parts = []
    for label_idx, name in enumerate(class_names):
        key = f"prob_{name}"
        if key in row:
            probability_parts.append(f"{name}={float(row[key]):.3f}")
        else:
            probability_parts.append(f"class_{label_idx}=nan")
    return (
        f"t={float(row['end_time_sec']):7.2f}s "
        f"pred={pred_label} ({pred_name}) "
        f"probabilities: {', '.join(probability_parts)}"
    )


def evaluate_prediction_log_against_labeled_recording(
    prediction_csv: PathLike,
    labeled_npz: PathLike,
    output_dir: PathLike,
    checkpoint_path: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """Score an existing real-time prediction CSV against audio-derived labels."""

    output_dir = ensure_dir(output_dir)
    prediction_csv = Path(prediction_csv)
    rec = load_labeled_recording(labeled_npz)
    pred_df = pd.read_csv(prediction_csv)

    if len(pred_df) == 0:
        pred_df["true_label"] = []
        pred_df["correct"] = []
        class_names = rec["class_names"]
        summary = pd.DataFrame([{"n_windows": 0, "accuracy": np.nan, "balanced_accuracy": np.nan}])
        per_class = pd.DataFrame()
    else:
        fs = int(rec["samplerate"])
        labels = np.asarray(rec["sample_labels"], dtype=np.int64).reshape(-1)
        if checkpoint_path is not None:
            _, checkpoint, _ = load_checkpoint(checkpoint_path)
            class_names = class_names_from_checkpoint(checkpoint, fallback=rec["class_names"])
            win_cfg = window_config_from_checkpoint(checkpoint)
            default_window_samples = int(round(win_cfg.window_sec * fs))
            label_mode = win_cfg.label_mode
        else:
            class_names = tuple(rec["class_names"])
            default_window_samples = int(round(1.0 * fs))
            label_mode = "endpoint"

        true_values = []
        valid_rows = []
        for _, row in pred_df.iterrows():
            end_sample = int(row["end_sample"])
            window_samples = int(row.get("window_samples", default_window_samples))
            start_sample = max(0, end_sample - window_samples)
            if end_sample < 1 or end_sample > len(labels):
                true_values.append(np.nan)
                valid_rows.append(False)
                continue
            window = labels[start_sample:end_sample]
            if len(window) == 0:
                true_values.append(np.nan)
                valid_rows.append(False)
            else:
                true_values.append(window_label(window, mode=label_mode))
                valid_rows.append(True)

        pred_df = pred_df.copy()
        pred_df["true_label"] = true_values
        pred_df["valid_true_label"] = valid_rows
        valid = pred_df[pred_df["valid_true_label"].astype(bool)].copy()
        if len(valid):
            y_true = valid["true_label"].astype(int).to_numpy()
            y_pred = valid["pred_label"].astype(int).to_numpy()
            summary, per_class = classification_summary(y_true, y_pred, class_names)
            pred_df["correct"] = np.nan
            pred_df.loc[valid.index, "correct"] = y_true == y_pred
        else:
            summary = pd.DataFrame([{"n_windows": int(len(pred_df)), "accuracy": np.nan, "balanced_accuracy": np.nan}])
            per_class = pd.DataFrame()
            pred_df["correct"] = np.nan

    valid_pred_df = pred_df[pred_df.get("valid_true_label", True).astype(bool)] if "valid_true_label" in pred_df else pred_df
    delay_df = estimate_transition_delay(
        valid_pred_df,
        sample_labels=rec["sample_labels"],
        fs=int(rec["samplerate"]),
    )
    delay_summary = summarize_transition_delay(delay_df)
    cue_delay_df = estimate_cue_prediction_delay(
        valid_pred_df,
        sample_labels=rec["sample_labels"],
        cue_onset_samples=rec.get("cue_onset_samples", np.empty(0, dtype=np.int64)),
        fs=int(rec["samplerate"]),
        class_names=class_names,
    )
    cue_delay_summary = summarize_cue_prediction_delay(cue_delay_df)
    xcov_delay_summary, xcov_curve = estimate_prediction_xcov_delay(
        valid_pred_df,
        sample_labels=rec["sample_labels"],
        fs=int(rec["samplerate"]),
        class_names=class_names,
        target_label=1,
    )

    stem = Path(labeled_npz).stem
    evaluated_csv = output_dir / f"{stem}_realtime_predictions_evaluated.csv"
    summary_csv = output_dir / f"{stem}_realtime_summary.csv"
    per_class_csv = output_dir / f"{stem}_realtime_per_class.csv"
    delay_csv = output_dir / f"{stem}_realtime_delay_by_transition.csv"
    delay_summary_csv = output_dir / f"{stem}_realtime_delay_summary.csv"
    cue_delay_csv = output_dir / f"{stem}_realtime_cue_delay_by_cue.csv"
    cue_delay_summary_csv = output_dir / f"{stem}_realtime_cue_delay_summary.csv"
    xcov_delay_summary_csv = output_dir / f"{stem}_realtime_xcov_delay_summary.csv"
    xcov_curve_csv = output_dir / f"{stem}_realtime_xcov_curve.csv"

    pred_df.to_csv(evaluated_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    per_class.to_csv(per_class_csv, index=False)
    delay_df.to_csv(delay_csv, index=False)
    delay_summary.to_csv(delay_summary_csv, index=False)
    cue_delay_df.to_csv(cue_delay_csv, index=False)
    cue_delay_summary.to_csv(cue_delay_summary_csv, index=False)
    xcov_delay_summary.to_csv(xcov_delay_summary_csv, index=False)
    xcov_curve.to_csv(xcov_curve_csv, index=False)

    return {
        "evaluated_predictions": pred_df,
        "summary": summary,
        "per_class": per_class,
        "delay": delay_df,
        "delay_summary": delay_summary,
        "cue_delay": cue_delay_df,
        "cue_delay_summary": cue_delay_summary,
        "xcov_delay_summary": xcov_delay_summary,
        "xcov_curve": xcov_curve,
        "evaluated_csv": evaluated_csv,
        "summary_csv": summary_csv,
        "per_class_csv": per_class_csv,
        "delay_csv": delay_csv,
        "delay_summary_csv": delay_summary_csv,
        "cue_delay_csv": cue_delay_csv,
        "cue_delay_summary_csv": cue_delay_summary_csv,
        "xcov_delay_summary_csv": xcov_delay_summary_csv,
        "xcov_curve_csv": xcov_curve_csv,
    }

def _setup_realtime_plot(
    acquired_channels: Sequence[int],
    class_names: Sequence[str],
) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    plt.ion()
    n_channels = len(acquired_channels)
    fig, axes = plt.subplots(
        n_channels + 1,
        1,
        sharex=True,
        figsize=(14, max(4.0, 1.45 * (n_channels + 1))),
        squeeze=False,
    )
    axes_flat = axes.reshape(-1)

    channel_lines = []
    for ax, channel in zip(axes_flat[:-1], acquired_channels):
        line, = ax.plot([], [], linewidth=0.7)
        channel_lines.append(line)
        ax.set_ylabel(f"Ch {int(channel)}")
        ax.grid(True, alpha=0.3)

    pred_line, = axes_flat[-1].step([], [], where="post", linewidth=1.8, color="tab:red")
    axes_flat[-1].set_ylabel("Prediction")
    axes_flat[-1].set_xlabel("Time (s)")
    axes_flat[-1].grid(True, alpha=0.3)
    if class_names:
        axes_flat[-1].set_yticks(np.arange(len(class_names)))
        axes_flat[-1].set_yticklabels([str(name) for name in class_names])
        axes_flat[-1].set_ylim(-0.5, max(0.5, len(class_names) - 0.5))

    axes_flat[0].set_title("Realtime channels and prediction")
    plt.tight_layout()

    display_handle = None
    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            from IPython.display import display

            display_handle = display(fig, display_id=True)
    except Exception:
        display_handle = None

    if display_handle is None:
        try:
            backend = str(plt.get_backend()).lower()
            if "agg" not in backend:
                fig.show()
        except Exception:
            pass

    return {
        "fig": fig,
        "axes": axes_flat,
        "channel_lines": channel_lines,
        "prediction_line": pred_line,
        "display_handle": display_handle,
        "plt": plt,
    }


def _set_signal_axis_limits(ax: Any, trace: np.ndarray) -> None:
    finite = np.asarray(trace, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return
    ymin = float(np.min(finite))
    ymax = float(np.max(finite))
    if np.isclose(ymin, ymax):
        pad = max(1.0, abs(ymin) * 0.1)
    else:
        pad = (ymax - ymin) * 0.1
    ax.set_ylim(ymin - pad, ymax + pad)


def _update_realtime_plot(
    plot_state: Optional[Dict[str, Any]],
    raw_data: np.ndarray,
    fs: int,
    prediction_rows: Sequence[Dict[str, Any]],
    class_names: Sequence[str],
    live_plot_window_sec: Optional[float],
) -> None:
    if plot_state is None:
        return

    raw = _as_2d_samples_channels(raw_data)
    if raw.shape[0] == 0:
        return

    axes = plot_state["axes"]
    end_time = raw.shape[0] / float(fs)
    if live_plot_window_sec is None:
        start_time = 0.0
    else:
        start_time = max(0.0, end_time - float(live_plot_window_sec))
    start_sample = max(0, int(np.floor(start_time * int(fs))))
    t = np.arange(start_sample, raw.shape[0], dtype=np.float64) / float(fs)
    visible = raw[start_sample:]

    for idx, line in enumerate(plot_state["channel_lines"]):
        if idx >= visible.shape[1]:
            line.set_data([], [])
            continue
        line.set_data(t, visible[:, idx])
        _set_signal_axis_limits(axes[idx], visible[:, idx])

    pred_times = []
    pred_labels = []
    previous_label = None
    for row in prediction_rows:
        row_time = float(row["end_time_sec"])
        row_label = int(row["pred_label"])
        if row_time < start_time:
            previous_label = row_label
            continue
        pred_times.append(row_time)
        pred_labels.append(row_label)

    if previous_label is not None:
        pred_times.insert(0, start_time)
        pred_labels.insert(0, previous_label)

    pred_line = plot_state["prediction_line"]
    pred_line.set_data(np.asarray(pred_times, dtype=np.float64), np.asarray(pred_labels, dtype=np.float64))
    if class_names:
        axes[-1].set_ylim(-0.5, max(0.5, len(class_names) - 0.5))
    elif pred_labels:
        ymin = min(pred_labels)
        ymax = max(pred_labels)
        axes[-1].set_ylim(float(ymin) - 0.5, float(ymax) + 0.5)

    if end_time <= start_time:
        end_time = start_time + 1.0
    for ax in axes:
        ax.set_xlim(start_time, end_time)

    fig = plot_state["fig"]
    fig.canvas.draw_idle()
    display_handle = plot_state.get("display_handle")
    if display_handle is not None:
        try:
            display_handle.update(fig)
        except Exception:
            pass
    if display_handle is None:
        try:
            backend = str(plot_state["plt"].get_backend()).lower()
            if "agg" not in backend:
                plot_state["plt"].pause(0.001)
        except Exception:
            pass


def run_realtime_mp150_prediction(
    output_dir: PathLike,
    checkpoint_path: PathLike,
    acquisition_config: AcquisitionConfig,
    preprocess_config: PreprocessConfig,
    duration_sec: Optional[float] = 60.0,
    trial_name: str = "realtime_trial",
    print_every_prediction: bool = True,
    live_plot: bool = False,
    live_plot_window_sec: Optional[float] = 15.0,
    live_plot_update_sec: Optional[float] = None,
    prediction_stride_sec: Optional[float] = None,
    prediction_flush_every: Optional[int] = 10,
) -> Dict[str, Any]:
    """Stream MP150 data and emit causal model predictions in real time.

    The full raw trial is saved continuously. Model predictions use the
    checkpoint's window length and feature settings. ``prediction_stride_sec``
    can override how often the window advances at test time. Prediction CSV
    writes are flushed every ``prediction_flush_every`` rows; use 0 or None to
    rely on normal file buffering until the trial ends.
    """

    output_dir = ensure_dir(output_dir)
    raw_path = output_dir / f"{trial_name}_raw.npz"
    prediction_csv = output_dir / f"{trial_name}_realtime_predictions.csv"

    model, checkpoint, device = load_checkpoint(checkpoint_path)
    win_cfg = window_config_from_checkpoint(checkpoint)
    fs = int(checkpoint["fs"])
    if int(acquisition_config.samplerate) != fs:
        raise ValueError(
            f"Samplerate mismatch: acquisition fs={acquisition_config.samplerate}, checkpoint fs={fs}."
        )
    acquired_channels = tuple(int(ch) for ch in acquisition_config.channels)
    if not acquired_channels:
        raise ValueError("acquisition_config.channels must contain at least one channel.")
    chunk_samples = _duration_to_sample_count(
        acquisition_config.chunk_sec, fs, "acquisition_config.chunk_sec"
    )
    target_samples = (
        _duration_to_sample_count(duration_sec, fs)
        if duration_sec is not None
        else None
    )

    window_samples = int(round(float(win_cfg.window_sec) * fs))
    stride_sec = float(win_cfg.stride_sec if prediction_stride_sec is None else prediction_stride_sec)
    stride_samples = int(round(stride_sec * fs))
    class_names = class_names_from_checkpoint(checkpoint)
    if stride_samples <= 0 or window_samples <= 0:
        raise ValueError("window_sec and prediction stride must produce positive sample counts.")
    plot_update_sec = float(stride_sec if live_plot_update_sec is None else live_plot_update_sec)
    if live_plot and plot_update_sec <= 0:
        raise ValueError("live_plot_update_sec must be positive.")
    flush_every = int(prediction_flush_every or 0)
    if flush_every < 0:
        raise ValueError("prediction_flush_every must be non-negative or None.")

    MP150 = _import_mp150_class()
    mp = MP150(samplerate=fs, channels=list(acquired_channels))

    fieldnames = [
        "wall_time_sec",
        "end_sample",
        "end_time_sec",
        "pred_label",
        "window_samples",
        "stride_samples",
        "feature_mode",
    ] + [f"prob_{name}" for name in class_names]

    raw_chunks: List[np.ndarray] = []
    chunk_start_wall: List[float] = []
    chunk_end_wall: List[float] = []
    prediction_rows: List[Dict[str, Any]] = []
    eeg_buffer = np.empty((0, len(preprocess_config.eeg_channels)), dtype=np.float32)
    buffer_start_sample = 0
    total_samples_seen = 0
    next_prediction_end = window_samples
    start_wall = time.time()
    live_plot_state = _setup_realtime_plot(acquired_channels, class_names) if live_plot else None
    last_live_plot_wall = 0.0
    printed_prediction_line = False

    print(
        f"Realtime prediction: feature_mode={win_cfg.feature_mode}, "
        f"window={float(win_cfg.window_sec):.3f}s, stride={float(stride_sec):.3f}s",
        flush=True,
    )

    prediction_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(prediction_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        predictions_since_flush = 0

        try:
            while target_samples is None or total_samples_seen < target_samples:
                if target_samples is not None:
                    remaining_samples = target_samples - total_samples_seen
                    if remaining_samples <= 0:
                        break
                    request_samples = min(chunk_samples, remaining_samples)
                    request_sec = request_samples / float(fs)
                else:
                    request_sec = float(acquisition_config.chunk_sec)

                chunk_start_wall.append(time.time() - start_wall)
                chunk = _checked_acquisition_chunk(mp.get_chunk(request_sec), len(acquired_channels))
                chunk_end_wall.append(time.time() - start_wall)
                if target_samples is not None:
                    remaining_samples = target_samples - total_samples_seen
                    if chunk.shape[0] > remaining_samples:
                        chunk = chunk[:remaining_samples]
                raw_chunks.append(chunk.astype(np.float32))

                eeg_raw = extract_hardware_channels(
                    chunk, acquired_channels, preprocess_config.eeg_channels
                )
                eeg_buffer = np.concatenate([eeg_buffer, eeg_raw.astype(np.float32)], axis=0)
                total_samples_seen += int(chunk.shape[0])

                while next_prediction_end <= total_samples_seen:
                    window_start_global = next_prediction_end - window_samples
                    local_start = window_start_global - buffer_start_sample
                    local_end = next_prediction_end - buffer_start_sample
                    if local_start < 0 or local_end > len(eeg_buffer):
                        break

                    raw_window = eeg_buffer[local_start:local_end]
                    model_window = preprocess_eeg_signal(
                        raw_window,
                        fs=fs,
                        preprocess_config=preprocess_config,
                    )
                    X_raw = extract_window_features(model_window, fs=fs, config=win_cfg)[None, ...]
                    X = apply_normalizer(X_raw, checkpoint["normalizer_mean"], checkpoint["normalizer_std"])
                    pred, prob = predict_single_window(model, X, device=device)

                    row = {
                        "wall_time_sec": time.time() - start_wall,
                        "end_sample": int(next_prediction_end),
                        "end_time_sec": float(next_prediction_end) / float(fs),
                        "pred_label": int(pred[0]),
                        "window_samples": int(window_samples),
                        "stride_samples": int(stride_samples),
                        "feature_mode": win_cfg.feature_mode,
                    }
                    for key, value in _prediction_probability_columns(prob, class_names).items():
                        row[key] = float(value[0])
                    writer.writerow(row)
                    predictions_since_flush += 1
                    if flush_every > 0 and predictions_since_flush >= flush_every:
                        f.flush()
                        predictions_since_flush = 0
                    prediction_rows.append(row)

                    if print_every_prediction:
                        status = _format_realtime_prediction_status(row, class_names)
                        sys.stdout.write("\r" + status.ljust(160))
                        sys.stdout.flush()
                        printed_prediction_line = True
                    next_prediction_end += stride_samples

                keep_from_global = max(0, next_prediction_end - window_samples)
                drop = keep_from_global - buffer_start_sample
                if drop > 0:
                    eeg_buffer = eeg_buffer[drop:]
                    buffer_start_sample = keep_from_global

                if live_plot_state is not None:
                    now_wall = time.time()
                    if now_wall - last_live_plot_wall >= plot_update_sec:
                        _update_realtime_plot(
                            live_plot_state,
                            np.concatenate(raw_chunks, axis=0),
                            fs=fs,
                            prediction_rows=prediction_rows,
                            class_names=class_names,
                            live_plot_window_sec=live_plot_window_sec,
                        )
                        last_live_plot_wall = now_wall
        finally:
            try:
                f.flush()
            except Exception:
                pass
            if printed_prediction_line:
                sys.stdout.write("\n")
                sys.stdout.flush()
            mp.close()
            if live_plot_state is not None and raw_chunks:
                _update_realtime_plot(
                    live_plot_state,
                    np.concatenate(raw_chunks, axis=0),
                    fs=fs,
                    prediction_rows=prediction_rows,
                    class_names=class_names,
                    live_plot_window_sec=live_plot_window_sec,
                )

    raw_data = np.concatenate(raw_chunks, axis=0).astype(np.float32) if raw_chunks else np.empty(
        (0, len(acquired_channels)), dtype=np.float32
    )
    np.savez(
        raw_path,
        data=raw_data,
        samplerate=np.array(fs, dtype=np.int64),
        channels=np.asarray(acquired_channels, dtype=np.int64),
        time_sec=np.arange(raw_data.shape[0], dtype=np.float64) / float(fs),
        segment_name=np.array(str(trial_name)),
        chunk_start_wall_sec=np.asarray(chunk_start_wall, dtype=np.float64),
        chunk_end_wall_sec=np.asarray(chunk_end_wall, dtype=np.float64),
        requested_duration_sec=np.array(np.nan if duration_sec is None else float(duration_sec), dtype=np.float64),
        created_at=np.array(_now_string()),
        metadata_json=np.array(json.dumps({
            "mode": "realtime_prediction",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_stride_sec": float(win_cfg.stride_sec),
            "prediction_stride_sec": float(stride_sec),
            "prediction_flush_every": int(flush_every),
        })),
    )

    return {
        "raw_path": raw_path,
        "prediction_csv": prediction_csv,
        "predictions": pd.DataFrame(prediction_rows),
    }


def posthoc_analyze_realtime_trial(
    raw_npz: PathLike,
    prediction_csv: PathLike,
    checkpoint_path: PathLike,
    output_dir: PathLike,
    preprocess_config: PreprocessConfig,
    label_config: AudioLabelConfig,
) -> Dict[str, Any]:
    """Label a streamed raw trial from audio and score the real-time predictions."""

    output_dir = ensure_dir(output_dir)
    raw_npz = Path(raw_npz)
    labeled_npz = output_dir / f"{raw_npz.stem}_labeled.npz"
    preprocess_recording(raw_npz, labeled_npz, preprocess_config, label_config)
    result = evaluate_prediction_log_against_labeled_recording(
        prediction_csv=prediction_csv,
        labeled_npz=labeled_npz,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
    )
    result["labeled_path"] = labeled_npz
    result["raw_path"] = raw_npz
    result["prediction_csv"] = Path(prediction_csv)
    return result


def collect_preprocess_and_test_trial(
    output_dir: PathLike,
    checkpoint_path: PathLike,
    acquisition_config: AcquisitionConfig,
    preprocess_config: PreprocessConfig,
    label_config: AudioLabelConfig,
    duration_sec: float = 300.0,
    trial_name: str = "test_trial",
    prediction_stride_sec: Optional[float] = None,
    prediction_flush_every: Optional[int] = 10,
) -> Dict[str, Any]:
    """Collect one continuous trial, predict on sliding windows, then score posthoc."""

    realtime = run_realtime_mp150_prediction(
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        acquisition_config=acquisition_config,
        preprocess_config=preprocess_config,
        duration_sec=duration_sec,
        trial_name=trial_name,
        print_every_prediction=True,
        prediction_stride_sec=prediction_stride_sec,
        prediction_flush_every=prediction_flush_every,
    )
    analysis = posthoc_analyze_realtime_trial(
        raw_npz=realtime["raw_path"],
        prediction_csv=realtime["prediction_csv"],
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        preprocess_config=preprocess_config,
        label_config=label_config,
    )
    analysis["realtime_predictions"] = realtime["predictions"]
    return analysis


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


def _label_name(label: int, class_names: Sequence[str]) -> str:
    label = int(label)
    return str(class_names[label]) if 0 <= label < len(class_names) else f"class_{label}"


def _prediction_probability_column(
    prediction_df: pd.DataFrame,
    target_label: int,
    class_names: Sequence[str],
) -> Optional[str]:
    candidates: List[str] = []
    if 0 <= int(target_label) < len(class_names):
        candidates.append(f"prob_{class_names[int(target_label)]}")
    candidates.append(f"prob_class_{int(target_label)}")
    candidates.append(f"prob_{int(target_label)}")
    for column in candidates:
        if column in prediction_df.columns:
            return column
    return None


def estimate_cue_prediction_delay(
    prediction_df: pd.DataFrame,
    sample_labels: np.ndarray,
    cue_onset_samples: np.ndarray,
    fs: int,
    class_names: Sequence[str] = (),
    max_delay_sec: Optional[float] = 10.0,
    sustained_windows: int = 3,
) -> pd.DataFrame:
    """Measure delay from each audio cue onset to the corresponding prediction change."""

    labels = np.asarray(sample_labels, dtype=np.int64).reshape(-1)
    cues = np.asarray(cue_onset_samples, dtype=np.int64).reshape(-1)
    cues = cues[(cues >= 0) & (cues < len(labels))]
    if len(labels) == 0 or len(cues) == 0 or len(prediction_df) == 0:
        return pd.DataFrame()

    pred_df = prediction_df.sort_values("end_time_sec").reset_index(drop=True)
    pred_times = pred_df["end_time_sec"].to_numpy(dtype=float)
    pred_labels = pred_df["pred_label"].astype(int).to_numpy()
    if len(pred_times) == 0:
        return pd.DataFrame()

    pred_change_idx = np.flatnonzero(np.diff(pred_labels) != 0) + 1
    pred_change_times = pred_times[pred_change_idx]
    pred_change_to_labels = pred_labels[pred_change_idx]
    hold = max(1, int(sustained_windows))

    rows: List[Dict[str, Any]] = []
    for cue_idx, cue_sample in enumerate(cues):
        cue_sample = int(cue_sample)
        cue_t = cue_sample / float(fs)
        next_cue_sample = int(cues[cue_idx + 1]) if cue_idx + 1 < len(cues) else len(labels)
        next_cue_t = next_cue_sample / float(fs) if next_cue_sample < len(labels) else np.inf

        from_label = int(labels[cue_sample - 1]) if cue_sample > 0 else int(labels[cue_sample])
        target_sample = cue_sample
        if cue_sample > 0:
            search_stop = max(cue_sample + 1, min(next_cue_sample, len(labels)))
            changed = np.flatnonzero(labels[cue_sample:search_stop] != from_label)
            if len(changed):
                target_sample = cue_sample + int(changed[0])
        target_label = int(labels[target_sample])
        label_transition_t = target_sample / float(fs)

        latest_t = next_cue_t
        if max_delay_sec is not None:
            latest_t = min(latest_t, cue_t + float(max_delay_sec))

        post_idx = np.flatnonzero((pred_times >= cue_t) & (pred_times <= latest_t))
        correct_idx = post_idx[pred_labels[post_idx] == target_label] if len(post_idx) else np.array([], dtype=int)
        if len(correct_idx):
            first_correct_time = float(pred_times[int(correct_idx[0])])
            delay_first_correct = first_correct_time - cue_t
        else:
            first_correct_time = np.nan
            delay_first_correct = np.nan

        transition_mask = (
            (pred_change_times >= cue_t)
            & (pred_change_times <= latest_t)
            & (pred_change_to_labels == target_label)
        )
        transition_idx = np.flatnonzero(transition_mask)
        if len(transition_idx):
            predicted_transition_time = float(pred_change_times[int(transition_idx[0])])
            delay_pred_transition = predicted_transition_time - cue_t
        else:
            predicted_transition_time = np.nan
            delay_pred_transition = np.nan

        sustained_time = np.nan
        sustained_confirm_time = np.nan
        if len(post_idx):
            for idx in post_idx:
                idx = int(idx)
                end_idx = idx + hold
                if end_idx > len(pred_labels):
                    break
                if pred_times[end_idx - 1] > latest_t:
                    break
                if np.all(pred_labels[idx:end_idx] == target_label):
                    sustained_time = float(pred_times[idx])
                    sustained_confirm_time = float(pred_times[end_idx - 1])
                    break
        delay_sustained = sustained_time - cue_t if np.isfinite(sustained_time) else np.nan

        rows.append(
            {
                "cue_index": int(cue_idx),
                "cue_onset_sample": cue_sample,
                "cue_onset_time_sec": cue_t,
                "label_transition_sample": int(target_sample),
                "label_transition_time_sec": label_transition_t,
                "from_label": from_label,
                "from_label_name": _label_name(from_label, class_names),
                "target_label": target_label,
                "target_label_name": _label_name(target_label, class_names),
                "transition_type": f"{from_label}->{target_label}",
                "first_correct_prediction_time_sec": first_correct_time,
                "cue_to_first_correct_prediction_sec": delay_first_correct,
                "predicted_transition_time_sec": predicted_transition_time,
                "cue_to_predicted_transition_sec": delay_pred_transition,
                "sustained_prediction_time_sec": sustained_time,
                "sustained_prediction_confirm_time_sec": sustained_confirm_time,
                "cue_to_sustained_prediction_sec": delay_sustained,
                "sustained_windows": hold,
                "matched_first_correct_prediction": bool(np.isfinite(delay_first_correct)),
                "matched_predicted_transition": bool(np.isfinite(delay_pred_transition)),
                "matched_sustained_prediction": bool(np.isfinite(delay_sustained)),
            }
        )

    return pd.DataFrame(rows)


def summarize_cue_prediction_delay(cue_delay_df: pd.DataFrame) -> pd.DataFrame:
    if cue_delay_df is None or len(cue_delay_df) == 0:
        return pd.DataFrame(
            [
                {
                    "n_cues": 0,
                    "n_matched_first_correct": 0,
                    "mean_cue_to_first_correct_sec": np.nan,
                    "median_cue_to_first_correct_sec": np.nan,
                    "n_matched_predicted_transition": 0,
                    "mean_cue_to_predicted_transition_sec": np.nan,
                    "median_cue_to_predicted_transition_sec": np.nan,
                    "n_matched_sustained": 0,
                    "mean_cue_to_sustained_prediction_sec": np.nan,
                    "median_cue_to_sustained_prediction_sec": np.nan,
                }
            ]
        )

    first = cue_delay_df["cue_to_first_correct_prediction_sec"].dropna().to_numpy(dtype=float)
    transition = cue_delay_df["cue_to_predicted_transition_sec"].dropna().to_numpy(dtype=float)
    sustained = cue_delay_df["cue_to_sustained_prediction_sec"].dropna().to_numpy(dtype=float)
    return pd.DataFrame(
        [
            {
                "n_cues": int(len(cue_delay_df)),
                "n_matched_first_correct": int(cue_delay_df["matched_first_correct_prediction"].sum()),
                "mean_cue_to_first_correct_sec": float(np.mean(first)) if len(first) else np.nan,
                "median_cue_to_first_correct_sec": float(np.median(first)) if len(first) else np.nan,
                "n_matched_predicted_transition": int(cue_delay_df["matched_predicted_transition"].sum()),
                "mean_cue_to_predicted_transition_sec": float(np.mean(transition)) if len(transition) else np.nan,
                "median_cue_to_predicted_transition_sec": float(np.median(transition)) if len(transition) else np.nan,
                "n_matched_sustained": int(cue_delay_df["matched_sustained_prediction"].sum()),
                "mean_cue_to_sustained_prediction_sec": float(np.mean(sustained)) if len(sustained) else np.nan,
                "median_cue_to_sustained_prediction_sec": float(np.median(sustained)) if len(sustained) else np.nan,
            }
        ]
    )


def _prediction_signal_on_samples(
    prediction_df: pd.DataFrame,
    n_samples: int,
    fs: int,
    value_column: Optional[str],
    target_label: int,
) -> Tuple[np.ndarray, str]:
    pred_df = prediction_df.sort_values("end_time_sec").reset_index(drop=True)
    signal = np.full(int(n_samples), np.nan, dtype=np.float64)
    if len(pred_df) == 0 or n_samples <= 0:
        return signal, value_column or "pred_label"

    if value_column is not None and value_column in pred_df.columns:
        values = pred_df[value_column].to_numpy(dtype=float)
        source = value_column
    else:
        values = (pred_df["pred_label"].astype(int).to_numpy() == int(target_label)).astype(float)
        source = f"pred_label_equals_{int(target_label)}"

    if "end_sample" in pred_df.columns:
        end_samples = pred_df["end_sample"].to_numpy(dtype=float)
    else:
        end_samples = pred_df["end_time_sec"].to_numpy(dtype=float) * float(fs)
    end_samples = np.asarray(np.round(end_samples), dtype=np.int64)

    for i, sample in enumerate(end_samples):
        start = int(np.clip(sample, 0, n_samples))
        stop = int(np.clip(end_samples[i + 1], 0, n_samples)) if i + 1 < len(end_samples) else int(n_samples)
        if stop > start:
            signal[start:stop] = float(values[i])

    return signal, source


def normalized_xcov_coefficients(
    reference: np.ndarray,
    response: np.ndarray,
    fs: int,
    max_lag_sec: float = 10.0,
    min_overlap_samples: Optional[int] = None,
) -> pd.DataFrame:
    """Python equivalent of normalized xcov with positive lag meaning response lags reference."""

    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    resp = np.asarray(response, dtype=np.float64).reshape(-1)
    n = min(len(ref), len(resp))
    ref = ref[:n]
    resp = resp[:n]
    if n == 0:
        return pd.DataFrame(columns=["lag_samples", "lag_sec", "xcov_coeff", "n_overlap"])

    max_lag = min(int(round(float(max_lag_sec) * int(fs))), max(0, n - 1))
    min_overlap = int(min_overlap_samples) if min_overlap_samples is not None else max(3, int(round(0.5 * int(fs))))
    rows: List[Dict[str, Any]] = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a = ref[: n - lag]
            b = resp[lag:n]
        else:
            a = ref[-lag:n]
            b = resp[: n + lag]

        valid = np.isfinite(a) & np.isfinite(b)
        n_overlap = int(np.sum(valid))
        if n_overlap < min_overlap:
            coeff = np.nan
        else:
            av = a[valid] - np.mean(a[valid])
            bv = b[valid] - np.mean(b[valid])
            denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
            coeff = float(np.dot(av, bv) / denom) if denom > 0 else np.nan
        rows.append(
            {
                "lag_samples": int(lag),
                "lag_sec": float(lag) / float(fs),
                "xcov_coeff": coeff,
                "n_overlap": n_overlap,
            }
        )

    return pd.DataFrame(rows)


def estimate_prediction_xcov_delay(
    prediction_df: pd.DataFrame,
    sample_labels: np.ndarray,
    fs: int,
    class_names: Sequence[str] = (),
    target_label: int = 1,
    max_lag_sec: float = 10.0,
    signal_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate global prediction lag using normalized cross-covariance."""

    labels = np.asarray(sample_labels, dtype=np.int64).reshape(-1)
    if len(labels) == 0 or len(prediction_df) == 0:
        empty_curve = pd.DataFrame(columns=["lag_samples", "lag_sec", "xcov_coeff", "n_overlap"])
        summary = pd.DataFrame(
            [
                {
                    "target_label": int(target_label),
                    "target_label_name": _label_name(target_label, class_names),
                    "prediction_signal_column": signal_column or "",
                    "xcov_delay_sec": np.nan,
                    "xcov_lag_samples": np.nan,
                    "xcov_peak_coeff": np.nan,
                    "max_lag_sec": float(max_lag_sec),
                    "n_valid_samples": 0,
                }
            ]
        )
        return summary, empty_curve

    prob_column = signal_column or _prediction_probability_column(prediction_df, target_label, class_names)
    pred_signal, source = _prediction_signal_on_samples(
        prediction_df,
        n_samples=len(labels),
        fs=fs,
        value_column=prob_column,
        target_label=target_label,
    )
    true_signal = (labels == int(target_label)).astype(np.float64)
    valid_samples = int(np.sum(np.isfinite(pred_signal)))
    curve = normalized_xcov_coefficients(
        true_signal,
        pred_signal,
        fs=fs,
        max_lag_sec=max_lag_sec,
    )

    valid_curve = curve[np.isfinite(curve["xcov_coeff"].to_numpy(dtype=float))]
    if len(valid_curve):
        best_idx = int(valid_curve["xcov_coeff"].astype(float).idxmax())
        best = curve.loc[best_idx]
        delay_sec = float(best["lag_sec"])
        lag_samples = int(best["lag_samples"])
        coeff = float(best["xcov_coeff"])
    else:
        delay_sec = np.nan
        lag_samples = np.nan
        coeff = np.nan

    summary = pd.DataFrame(
        [
            {
                "target_label": int(target_label),
                "target_label_name": _label_name(target_label, class_names),
                "prediction_signal_column": source,
                "xcov_delay_sec": delay_sec,
                "xcov_lag_samples": lag_samples,
                "xcov_peak_coeff": coeff,
                "max_lag_sec": float(max_lag_sec),
                "n_valid_samples": valid_samples,
            }
        ]
    )
    return summary, curve

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


def plot_labeled_recording(
    labeled_npz: PathLike,
    max_duration_sec: Optional[float] = 30.0,
    channel_names: Optional[Sequence[str]] = None,
    use_raw_eeg: bool = False,
):
    import matplotlib.pyplot as plt

    rec = load_labeled_recording(labeled_npz)
    eeg = rec["eeg_raw"] if use_raw_eeg and rec.get("eeg_raw") is not None else rec["eeg"]
    audio = rec.get("audio")
    acquired_channels = tuple(rec.get("acquired_channels") or ())
    eeg_channels = tuple(rec.get("eeg_channels") or range(1, eeg.shape[1] + 1))
    audio_channel = rec.get("audio_channel")
    labels = rec["sample_labels"]
    fs = int(rec["samplerate"])
    n = len(labels)
    if max_duration_sec is not None:
        n = min(n, int(round(float(max_duration_sec) * fs)))
    t = np.arange(n) / float(fs)
    cue_times = _cue_onset_times_for_plot(rec, max_duration_sec)

    channel_traces: List[Tuple[int, str, np.ndarray]] = []
    for idx in range(eeg.shape[1]):
        hardware_channel = int(eeg_channels[idx]) if idx < len(eeg_channels) else idx + 1
        channel_label = (
            str(channel_names[idx])
            if channel_names is not None and idx < len(channel_names)
            else f"EEG channel {hardware_channel}"
        )
        channel_traces.append((hardware_channel, channel_label, eeg[:n, idx]))
    if audio is not None:
        hardware_channel = int(audio_channel) if audio_channel is not None else len(channel_traces) + 1
        channel_traces.append((hardware_channel, f"Audio channel {hardware_channel}", np.asarray(audio)[:n]))

    if acquired_channels:
        channel_order = {int(ch): idx for idx, ch in enumerate(acquired_channels)}
        channel_traces.sort(key=lambda item: channel_order.get(int(item[0]), int(item[0])))
    else:
        channel_traces.sort(key=lambda item: int(item[0]))

    fig, axes = plt.subplots(
        len(channel_traces),
        1,
        sharex=True,
        figsize=(14, max(2.5, 1.6 * len(channel_traces))),
        squeeze=False,
    )
    axes_flat = axes.reshape(-1)
    for ax, (_, label, trace) in zip(axes_flat, channel_traces):
        ax.plot(t, trace, linewidth=0.7)
        for cue_time in cue_times:
            ax.axvline(cue_time, linestyle=":", color="tab:red", alpha=0.85, linewidth=2.0)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes_flat[-1].set_xlabel("Time (s)")
    axes_flat[0].set_title(f"Recording channels: {Path(labeled_npz).name}")
    plt.tight_layout()
    return fig, axes_flat


def plot_predictions_overlay(
    labeled_npz: PathLike,
    predictions: Union[pd.DataFrame, PathLike],
    max_duration_sec: Optional[float] = None,
    channel_names: Sequence[str] = ("O1", "Oz", "O2", "POz"),
    use_raw_eeg: bool = False,
    show_true_labels: bool = False,
):
    import matplotlib.pyplot as plt

    rec = load_labeled_recording(labeled_npz)
    pred_df = pd.read_csv(predictions) if not isinstance(predictions, pd.DataFrame) else predictions.copy()
    eeg = rec["eeg_raw"] if use_raw_eeg and rec.get("eeg_raw") is not None else rec["eeg"]
    labels = rec["sample_labels"]
    fs = int(rec["samplerate"])

    n = len(labels)
    if max_duration_sec is not None:
        n = min(n, int(round(float(max_duration_sec) * fs)))
        pred_df = pred_df[pred_df["end_time_sec"] <= float(max_duration_sec)].copy()

    t = np.arange(n) / float(fs)
    n_channels = min(4, eeg.shape[1])
    fig, axes = plt.subplots(
        n_channels,
        1,
        sharex=True,
        figsize=(14, max(3.0, 1.8 * n_channels)),
        squeeze=False,
    )
    axes_flat = axes.reshape(-1)

    class_names = tuple(rec.get("class_names") or ())
    pred_axes = []
    pred_t = np.asarray([], dtype=np.float64)
    pred = np.asarray([], dtype=np.float64)
    if len(pred_df):
        pred_t = pred_df["end_time_sec"].to_numpy(dtype=float)
        pred = pred_df["pred_label"].to_numpy(dtype=float)
    cue_times = _cue_onset_times_for_plot(rec, max_duration_sec)

    cue_color = "tab:red"
    prediction_color = "tab:orange"
    true_label_color = "tab:blue"
    legend_handles: List[Any] = []
    legend_labels: List[str] = []

    for idx, ax in enumerate(axes_flat):
        name = str(channel_names[idx]) if idx < len(channel_names) else f"Ch {idx + 1}"
        eeg_line, = ax.plot(t, eeg[:n, idx], linewidth=0.7, color="black", label="EEG")
        if idx == 0:
            legend_handles.append(eeg_line)
            legend_labels.append("EEG")

        for cue_idx, cue_time in enumerate(cue_times):
            cue_line = ax.axvline(
                cue_time,
                linestyle=":",
                color=cue_color,
                alpha=0.9,
                linewidth=2.2,
                label="cue onset" if idx == 0 and cue_idx == 0 else None,
                zorder=3,
            )
            if idx == 0 and cue_idx == 0:
                legend_handles.append(cue_line)
                legend_labels.append("cue onset")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)

        pred_ax = ax.twinx()
        pred_axes.append(pred_ax)
        if show_true_labels and len(labels):
            true_line, = pred_ax.step(
                t,
                labels[:n].astype(float),
                where="post",
                linewidth=1.4,
                linestyle="--",
                color=true_label_color,
                alpha=0.8,
                label="true label",
                zorder=2,
            )
            if idx == 0:
                legend_handles.append(true_line)
                legend_labels.append("true label")
        if len(pred_t):
            pred_line, = pred_ax.step(
                pred_t,
                pred,
                where="post",
                linewidth=1.6,
                color=prediction_color,
                alpha=0.9,
                label="prediction",
            )
            if idx == 0:
                legend_handles.append(pred_line)
                legend_labels.append("prediction")
        if class_names:
            pred_ax.set_yticks(np.arange(len(class_names)))
            pred_ax.set_yticklabels([str(name) for name in class_names])
            pred_ax.set_ylim(-0.5, max(0.5, len(class_names) - 0.5))
        elif len(pred):
            pred_ax.set_ylim(float(np.min(pred)) - 0.5, float(np.max(pred)) + 0.5)
        pred_ax.tick_params(axis="y", colors=prediction_color)

    axes_flat[-1].set_xlabel("Time (s)")
    axes_flat[0].set_title(f"Realtime predictions over EEG channels: {Path(labeled_npz).name}")
    if legend_handles:
        axes_flat[0].legend(legend_handles, legend_labels, loc="upper left")
    plt.tight_layout()
    return fig, axes_flat
