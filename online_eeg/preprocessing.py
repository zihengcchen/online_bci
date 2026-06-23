"""Audio-cue labeling, EEG filtering, and labeled recording I/O."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    from .acquisition import load_raw_recording
    from .config import ArrayLike, AudioLabelConfig, PathLike, PreprocessConfig
    from .utils import _as_2d_samples_channels, _json_safe, ensure_dir, extract_hardware_channels
except ImportError:
    from acquisition import load_raw_recording
    from config import ArrayLike, AudioLabelConfig, PathLike, PreprocessConfig
    from utils import _as_2d_samples_channels, _json_safe, ensure_dir, extract_hardware_channels


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


__all__ = [
    "AudioLabelConfig",
    "PreprocessConfig",
    "rolling_rms_causal",
    "detect_audio_onsets",
    "labels_from_audio_onsets",
    "bandpass_filter",
    "notch_filter",
    "preprocess_eeg_signal",
    "preprocess_recording",
    "preprocess_many_recordings",
    "load_labeled_recording",
    "labeled_preprocess_summary",
]
