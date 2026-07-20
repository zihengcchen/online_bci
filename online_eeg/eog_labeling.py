"""EOG-offset labeling for offline EEG analyses."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from .acquisition import load_raw_recording
    from .config import AudioLabelConfig, PathLike, PreprocessConfig
    from .preprocessing import (
        _extract_optional_hardware_channels,
        bandpass_filter,
        detect_audio_onsets,
        labels_from_audio_onsets,
        normalized_signal_derivative,
        preprocess_eeg_signal,
        rolling_rms_causal,
    )
    from .utils import _json_safe, extract_hardware_channels
except ImportError:
    from acquisition import load_raw_recording
    from config import AudioLabelConfig, PathLike, PreprocessConfig
    from preprocessing import (
        _extract_optional_hardware_channels,
        bandpass_filter,
        detect_audio_onsets,
        labels_from_audio_onsets,
        normalized_signal_derivative,
        preprocess_eeg_signal,
        rolling_rms_causal,
    )
    from utils import _json_safe, extract_hardware_channels


@dataclass
class EogOffsetLabelConfig:
    """Settings for deriving label transitions from EOG activity offsets.

    By default, ``label_source="audio_anchored"`` uses audio cue onsets only as
    anchors to find the corresponding EOG activity event and preserve the
    existing cue order. With ``label_source="eog_events"``, labels are created
    directly from detected EOG event offsets and audio cues are only saved for
    plotting. The actual transition sample is the NeuroKit2-derived EOG event
    offset plus ``label_offset_sec``.

    EOG event onset/offset detection is delegated to NeuroKit2:
    ``eog_clean`` -> ``eog_findpeaks`` -> ``eog_features``. NeuroKit2's
    ``eog_features`` returns ``Blink_LeftZeros`` and ``Blink_RightZeros`` and
    cites BLINKER:
    Kleifges et al. (2017), Frontiers in Neuroscience, 11, 12.
    Docs: https://neuropsychology.github.io/NeuroKit/functions/eog.html
    """

    neurokit_clean_method: str = "neurokit"
    neurokit_peak_method: str = "brainstorm"
    detection_signal_mode: str = "raw"
    derivative_lowpass_hz: Optional[float] = None
    derivative_lowpass_order: int = 4
    detect_both_polarities: bool = True
    eog_channel_index: int = 0
    min_activity_duration_sec: float = 0.05
    merge_gap_sec: float = 0.15
    search_start_offset_sec: float = 0.0
    search_max_sec: Optional[float] = 6.0
    end_before_next_cue_sec: float = 0.20
    label_offset_sec: float = 0.0
    fallback_to_audio_onset: bool = False
    label_source: str = "audio_anchored"
    label_event_start_sec: Optional[float] = None
    label_event_stop_sec: Optional[float] = None
    label_event_min_interval_sec: float = 0.0
    label_event_min_peak_abs_value: Optional[float] = None


def eog_activity_score(
    eog_raw: np.ndarray,
    fs: int,
    activity_window_sec: float,
) -> np.ndarray:
    """Return a robust derivative-based EOG activity score."""

    derivative = normalized_signal_derivative(eog_raw).astype(np.float64)
    if derivative.shape[1] == 0:
        raise ValueError("EOG-offset labeling requires at least one recorded EOG channel.")
    activity = np.nanmax(np.abs(derivative), axis=1)
    window_samples = max(1, int(round(float(activity_window_sec) * int(fs))))
    return rolling_rms_causal(activity, window_samples)


def prepare_eog_detection_signal(
    eog_raw: np.ndarray,
    fs: int,
    config: EogOffsetLabelConfig,
) -> np.ndarray:
    """Return the signal that should be passed into NeuroKit2 EOG detection."""

    mode = str(config.detection_signal_mode).strip().lower()
    eog = np.asarray(eog_raw, dtype=np.float64)
    if eog.ndim == 1:
        eog = eog[:, None]

    if mode == "raw":
        return eog
    if mode == "normalized_derivative":
        return normalized_signal_derivative(eog).astype(np.float64)
    if mode == "normalized_derivative_lowpass":
        derivative = normalized_signal_derivative(eog).astype(np.float64)
        cutoff = config.derivative_lowpass_hz
        if cutoff is None:
            raise ValueError(
                "derivative_lowpass_hz must be set when detection_signal_mode="
                "'normalized_derivative_lowpass'."
            )
        return bandpass_filter(
            derivative,
            fs=int(fs),
            low_hz=None,
            high_hz=float(cutoff),
            order=int(config.derivative_lowpass_order),
        ).astype(np.float64)

    raise ValueError(
        "detection_signal_mode must be one of 'raw', 'normalized_derivative', "
        f"or 'normalized_derivative_lowpass', got {config.detection_signal_mode!r}."
    )


def _fallback_eog_features(cleaned: np.ndarray, peaks: np.ndarray, fs: int) -> pd.DataFrame:
    """Delineate EOG events when NeuroKit's feature extractor hits a tie edge case."""

    signal = np.asarray(cleaned, dtype=np.float64).reshape(-1)
    peaks = np.asarray(peaks, dtype=np.int64).reshape(-1)
    half_window = max(1, int(round(0.5 * int(fs))))
    leftzeros: List[int] = []
    rightzeros: List[int] = []

    for peak in peaks:
        peak = int(np.clip(int(peak), 0, max(0, len(signal) - 1)))
        start = max(0, peak - half_window)
        stop = min(len(signal), peak + half_window + 1)
        epoch = signal[start:stop]
        indices = np.arange(start, stop, dtype=np.int64)
        if epoch.size == 0:
            leftzeros.append(peak)
            rightzeros.append(peak)
            continue

        max_value = float(np.nanmax(epoch))
        max_candidates = indices[np.flatnonzero(epoch == max_value)]
        max_frame = int(max_candidates[0]) if len(max_candidates) else peak
        max_pos = int(np.searchsorted(indices, max_frame))

        signs = np.signbit(epoch)
        crossing_pos = np.flatnonzero(signs[1:] != signs[:-1]) + 1
        crossings = np.sort(np.append(indices[crossing_pos], max_frame))
        max_matches = np.flatnonzero(crossings == max_frame)
        max_crossing_pos = int(max_matches[0]) if len(max_matches) else int(np.searchsorted(crossings, max_frame))

        if max_crossing_pos - 1 >= 0:
            leftzero = int(crossings[max_crossing_pos - 1])
        else:
            before = epoch[: max_pos + 1]
            before_indices = indices[: max_pos + 1]
            min_value = float(np.nanmin(before))
            candidates = before_indices[np.flatnonzero(before == min_value)]
            leftzero = int(candidates[-1]) if len(candidates) else max(start, peak - 1)

        if max_crossing_pos + 1 < len(crossings):
            rightzero = int(crossings[max_crossing_pos + 1])
        else:
            after = epoch[max_pos:]
            after_indices = indices[max_pos:]
            min_value = float(np.nanmin(after))
            candidates = after_indices[np.flatnonzero(after == min_value)]
            rightzero = int(candidates[0]) if len(candidates) else min(stop - 1, peak + 1)

        if rightzero <= leftzero:
            leftzero = max(start, peak - 1)
            rightzero = min(stop - 1, peak + 1)
        leftzeros.append(int(leftzero))
        rightzeros.append(int(rightzero))

    return pd.DataFrame(
        {
            "Blink_LeftZeros": np.asarray(leftzeros, dtype=np.int64),
            "Blink_RightZeros": np.asarray(rightzeros, dtype=np.int64),
        }
    )


def _safe_neurokit_eog_features(cleaned: np.ndarray, peaks: np.ndarray, fs: int) -> pd.DataFrame:
    try:
        import neurokit2 as nk

        return nk.eog_features(cleaned, peaks, sampling_rate=int(fs))
    except ValueError as exc:
        if "can only convert an array of size 1 to a Python scalar" not in str(exc):
            raise
        return _fallback_eog_features(cleaned, peaks, fs)


def _neurokit_eog_events(
    eog_raw: np.ndarray,
    fs: int,
    config: EogOffsetLabelConfig,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Detect EOG event boundaries using NeuroKit2's EOG pipeline."""

    try:
        import neurokit2 as nk
    except ImportError as exc:
        raise ImportError(
            "NeuroKit2 is required for EOG onset/offset detection. "
            "Install neurokit2 or run this notebook in the configured super environment."
        ) from exc

    eog = np.asarray(eog_raw, dtype=np.float64)
    if eog.ndim == 1:
        eog = eog[:, None]
    if eog.shape[1] == 0:
        raise ValueError("EOG-offset labeling requires at least one recorded EOG channel.")

    channel_idx = int(config.eog_channel_index)
    if channel_idx < 0 or channel_idx >= eog.shape[1]:
        raise ValueError(
            f"eog_channel_index={channel_idx} is outside available EOG channels 0..{eog.shape[1] - 1}."
        )

    source = eog[:, channel_idx]
    polarities = (1, -1) if bool(config.detect_both_polarities) else (1,)
    rows: List[Dict[str, Any]] = []
    source_trace = None

    for polarity in polarities:
        oriented = float(polarity) * source
        cleaned = np.asarray(
            nk.eog_clean(
                oriented,
                sampling_rate=int(fs),
                method=str(config.neurokit_clean_method),
            ),
            dtype=np.float64,
        )
        if source_trace is None:
            source_trace = cleaned.astype(np.float64)

        peaks = np.asarray(
            nk.eog_findpeaks(
                cleaned,
                sampling_rate=int(fs),
                method=str(config.neurokit_peak_method),
            ),
            dtype=np.int64,
        ).reshape(-1)
        if peaks.size == 0:
            continue

        features = _safe_neurokit_eog_features(cleaned, peaks, fs)
        starts = np.asarray(features.get("Blink_LeftZeros", []), dtype=np.float64).reshape(-1)
        ends = np.asarray(features.get("Blink_RightZeros", []), dtype=np.float64).reshape(-1)

        for peak, start, end in zip(peaks, starts, ends):
            if not (np.isfinite(start) and np.isfinite(end)):
                continue
            start = int(np.clip(int(round(float(start))), 0, len(source)))
            end = int(np.clip(int(round(float(end))), 0, len(source)))
            peak = int(np.clip(int(peak), 0, max(0, len(source) - 1)))
            if end <= start:
                continue
            rows.append(
                {
                    "start_sample": start,
                    "end_sample": end,
                    "peak_sample": peak,
                    "polarity": int(polarity),
                    "source_channel_index": channel_idx,
                    "peak_value": float(cleaned[peak]),
                    "peak_abs_value": float(abs(cleaned[peak])),
                    "detection_source": (
                        "NeuroKit2 eog_clean/eog_findpeaks/eog_features "
                        f"(clean={config.neurokit_clean_method}, peaks={config.neurokit_peak_method})"
                    ),
                }
            )

    events = pd.DataFrame(rows)
    if events.empty:
        raise RuntimeError(
            "NeuroKit2 did not detect any EOG events. Check EOG polarity/channel or NeuroKit2 settings."
        )

    min_samples = max(1, int(round(float(config.min_activity_duration_sec) * int(fs))))
    events = events[(events["end_sample"] - events["start_sample"]) >= min_samples].copy()
    if events.empty:
        raise RuntimeError(
            "NeuroKit2 EOG events were all shorter than min_activity_duration_sec. "
            "Lower the duration setting or inspect the EOG signal."
        )

    events = events.sort_values(["start_sample", "end_sample", "peak_abs_value"]).reset_index(drop=True)
    merged: List[Dict[str, Any]] = []
    merge_gap_samples = max(0, int(round(float(config.merge_gap_sec) * int(fs))))
    for row in events.to_dict("records"):
        if not merged or int(row["start_sample"]) - int(merged[-1]["end_sample"]) > merge_gap_samples:
            merged.append(row)
            continue

        previous = merged[-1]
        keep = row if float(row["peak_abs_value"]) > float(previous["peak_abs_value"]) else previous
        merged[-1] = {
            **keep,
            "start_sample": min(int(previous["start_sample"]), int(row["start_sample"])),
            "end_sample": max(int(previous["end_sample"]), int(row["end_sample"])),
            "merged_neurokit_events": int(previous.get("merged_neurokit_events", 1))
            + int(row.get("merged_neurokit_events", 1)),
        }

    merged_events = pd.DataFrame(merged).sort_values(["start_sample", "end_sample"]).reset_index(drop=True)
    if source_trace is None:
        source_trace = np.zeros(len(source), dtype=np.float64)
    return merged_events, source_trace


def detect_eog_offsets_after_audio_cues(
    eog_raw: np.ndarray,
    fs: int,
    audio_cue_samples: Sequence[int],
    config: EogOffsetLabelConfig,
) -> Dict[str, Any]:
    """Find the NeuroKit2 EOG event end associated with each audio cue."""

    fs = int(fs)
    detection_signal = prepare_eog_detection_signal(eog_raw, fs=fs, config=config)
    events, source_trace = _neurokit_eog_events(detection_signal, fs=fs, config=config)

    audio_cue_samples = np.asarray(audio_cue_samples, dtype=np.int64).reshape(-1)
    n_samples = int(np.asarray(eog_raw).shape[0])
    label_offset = int(round(float(config.label_offset_sec) * fs))
    search_start_offset = int(round(float(config.search_start_offset_sec) * fs))
    search_max = (
        None
        if config.search_max_sec is None
        else max(1, int(round(float(config.search_max_sec) * fs)))
    )
    next_margin = max(0, int(round(float(config.end_before_next_cue_sec) * fs)))

    rows = []
    event_starts = []
    event_ends = []
    label_samples = []
    peaks = []
    for cue_idx, cue_sample in enumerate(audio_cue_samples):
        cue_sample = int(cue_sample)
        next_cue = int(audio_cue_samples[cue_idx + 1]) if cue_idx + 1 < len(audio_cue_samples) else n_samples
        search_start = int(np.clip(cue_sample + search_start_offset, 0, n_samples))
        search_stop = next_cue - next_margin if cue_idx + 1 < len(audio_cue_samples) else n_samples
        if search_max is not None:
            search_stop = min(search_stop, cue_sample + search_max)
        search_stop = int(np.clip(max(search_stop, search_start + 1), 0, n_samples))

        candidates = events[
            (events["end_sample"].astype(int) > search_start)
            & (events["start_sample"].astype(int) < search_stop)
        ].copy()
        future_candidates = candidates[candidates["peak_sample"].astype(int) >= search_start]
        chosen_row = (
            future_candidates.iloc[0]
            if len(future_candidates)
            else candidates.iloc[0] if len(candidates) else None
        )
        used_fallback = False

        if chosen_row is None:
            if not config.fallback_to_audio_onset:
                raise RuntimeError(
                    f"No EOG activity segment found after audio cue {cue_idx} at {cue_sample / fs:.3f}s. "
                    "Inspect NeuroKit2 EOG detection settings, increase search_max_sec, "
                    "or set fallback_to_audio_onset=True."
                )
            start = cue_sample
            end = cue_sample
            peak_sample = cue_sample
            polarity = 0
            peak_value = np.nan
            peak_abs_value = np.nan
            merged_neurokit_events = 0
            detection_source = "audio fallback"
            used_fallback = True
        else:
            start = int(chosen_row["start_sample"])
            end = int(chosen_row["end_sample"])
            peak_sample = int(chosen_row["peak_sample"])
            polarity = int(chosen_row["polarity"])
            peak_value = float(chosen_row["peak_value"])
            peak_abs_value = float(chosen_row["peak_abs_value"])
            raw_merged_count = chosen_row.get("merged_neurokit_events", 1)
            merged_neurokit_events = 1 if pd.isna(raw_merged_count) else int(raw_merged_count)
            detection_source = str(chosen_row["detection_source"])
            start = max(start, search_start)
            end = min(end, search_stop)

        label_sample = int(np.clip(end + label_offset, 0, n_samples))

        event_starts.append(start)
        event_ends.append(end)
        label_samples.append(label_sample)
        peaks.append(peak_abs_value)
        rows.append(
            {
                "cue_index": int(cue_idx),
                "audio_onset_sample": cue_sample,
                "audio_onset_time_sec": cue_sample / float(fs),
                "eog_activity_start_sample": int(start),
                "eog_activity_start_time_sec": start / float(fs),
                "eog_activity_end_sample": int(end),
                "eog_activity_end_time_sec": end / float(fs),
                "neurokit_peak_sample": int(peak_sample),
                "neurokit_peak_time_sec": int(peak_sample) / float(fs),
                "neurokit_peak_value": peak_value,
                "neurokit_peak_abs_value": peak_abs_value,
                "neurokit_event_polarity": polarity,
                "merged_neurokit_events": merged_neurokit_events,
                "eog_detection_source": detection_source,
                "eog_detection_signal_mode": str(config.detection_signal_mode),
                "eog_derivative_lowpass_hz": (
                    np.nan
                    if config.derivative_lowpass_hz is None
                    else float(config.derivative_lowpass_hz)
                ),
                "label_transition_sample": int(label_sample),
                "label_transition_time_sec": label_sample / float(fs),
                "audio_to_eog_offset_sec": (label_sample - cue_sample) / float(fs),
                "eog_activity_peak_value": peak_abs_value,
                "used_audio_fallback": bool(used_fallback),
            }
        )

    return {
        "score": source_trace.astype(np.float64),
        "threshold": np.nan,
        "detection_signal": detection_signal.astype(np.float64),
        "active_segments": events[["start_sample", "end_sample"]].to_numpy(dtype=np.int64),
        "all_events": events,
        "event_start_samples": np.asarray(event_starts, dtype=np.int64),
        "event_end_samples": np.asarray(event_ends, dtype=np.int64),
        "label_event_samples": np.asarray(label_samples, dtype=np.int64),
        "peak_values": np.asarray(peaks, dtype=np.float64),
        "event_table": pd.DataFrame(rows),
    }


def detect_eog_label_events(
    eog_raw: np.ndarray,
    fs: int,
    config: EogOffsetLabelConfig,
) -> Dict[str, Any]:
    """Find label transitions from EOG event offsets without using audio cues."""

    fs = int(fs)
    detection_signal = prepare_eog_detection_signal(eog_raw, fs=fs, config=config)
    events, source_trace = _neurokit_eog_events(detection_signal, fs=fs, config=config)

    n_samples = int(np.asarray(eog_raw).shape[0])
    label_offset = int(round(float(config.label_offset_sec) * fs))
    start_sample = (
        None
        if config.label_event_start_sec is None
        else int(round(float(config.label_event_start_sec) * fs))
    )
    stop_sample = (
        None
        if config.label_event_stop_sec is None
        else int(round(float(config.label_event_stop_sec) * fs))
    )
    min_interval_samples = max(0, int(round(float(config.label_event_min_interval_sec) * fs)))
    min_peak_abs = config.label_event_min_peak_abs_value

    rows = []
    event_starts = []
    event_ends = []
    label_samples = []
    peaks = []
    last_label_sample: Optional[int] = None

    for row in events.sort_values(["end_sample", "start_sample"]).to_dict("records"):
        start = int(row["start_sample"])
        end = int(row["end_sample"])
        label_sample = int(np.clip(end + label_offset, 0, n_samples))
        peak_abs_value = float(row["peak_abs_value"])

        if start_sample is not None and label_sample < start_sample:
            continue
        if stop_sample is not None and label_sample > stop_sample:
            continue
        if min_peak_abs is not None and peak_abs_value < float(min_peak_abs):
            continue
        if last_label_sample is not None and label_sample - last_label_sample < min_interval_samples:
            continue

        raw_merged_count = row.get("merged_neurokit_events", 1)
        merged_neurokit_events = 1 if pd.isna(raw_merged_count) else int(raw_merged_count)
        peak_sample = int(row["peak_sample"])
        polarity = int(row["polarity"])
        peak_value = float(row["peak_value"])
        detection_source = str(row["detection_source"])

        cue_idx = len(label_samples)
        event_starts.append(start)
        event_ends.append(end)
        label_samples.append(label_sample)
        peaks.append(peak_abs_value)
        rows.append(
            {
                "cue_index": int(cue_idx),
                "eog_event_index": int(cue_idx),
                "audio_onset_sample": np.nan,
                "audio_onset_time_sec": np.nan,
                "eog_activity_start_sample": int(start),
                "eog_activity_start_time_sec": start / float(fs),
                "eog_activity_end_sample": int(end),
                "eog_activity_end_time_sec": end / float(fs),
                "neurokit_peak_sample": int(peak_sample),
                "neurokit_peak_time_sec": int(peak_sample) / float(fs),
                "neurokit_peak_value": peak_value,
                "neurokit_peak_abs_value": peak_abs_value,
                "neurokit_event_polarity": polarity,
                "merged_neurokit_events": merged_neurokit_events,
                "eog_detection_source": detection_source,
                "eog_detection_signal_mode": str(config.detection_signal_mode),
                "eog_derivative_lowpass_hz": (
                    np.nan
                    if config.derivative_lowpass_hz is None
                    else float(config.derivative_lowpass_hz)
                ),
                "label_transition_sample": int(label_sample),
                "label_transition_time_sec": label_sample / float(fs),
                "audio_to_eog_offset_sec": np.nan,
                "eog_activity_peak_value": peak_abs_value,
                "used_audio_fallback": False,
            }
        )
        last_label_sample = label_sample

    return {
        "score": source_trace.astype(np.float64),
        "threshold": np.nan,
        "detection_signal": detection_signal.astype(np.float64),
        "active_segments": events[["start_sample", "end_sample"]].to_numpy(dtype=np.int64),
        "all_events": events,
        "event_start_samples": np.asarray(event_starts, dtype=np.int64),
        "event_end_samples": np.asarray(event_ends, dtype=np.int64),
        "label_event_samples": np.asarray(label_samples, dtype=np.int64),
        "peak_values": np.asarray(peaks, dtype=np.float64),
        "event_table": pd.DataFrame(rows),
    }


def preprocess_recording_with_eog_offset_labels(
    raw_npz: PathLike,
    output_npz: PathLike,
    preprocess_config: PreprocessConfig,
    label_config: AudioLabelConfig,
    eog_label_config: EogOffsetLabelConfig,
) -> Tuple[Path, pd.DataFrame]:
    """Preprocess a raw recording and save labels transitioning at EOG offsets."""

    raw = load_raw_recording(raw_npz)
    data = raw["data"]
    fs = int(raw["samplerate"])
    channels = tuple(raw["channels"])

    eeg_raw = extract_hardware_channels(data, channels, preprocess_config.eeg_channels)
    eog_raw, eog_channels, missing_eog_channels = _extract_optional_hardware_channels(
        data,
        channels,
        getattr(preprocess_config, "eog_channels", ()),
    )
    if eog_raw.shape[1] == 0:
        raise ValueError(f"No requested EOG channels found in {raw_npz}; channels={channels}.")

    audio = extract_hardware_channels(data, channels, (preprocess_config.audio_channel,))[:, 0]
    eeg = preprocess_eeg_signal(eeg_raw, fs=fs, preprocess_config=preprocess_config)

    audio_onset_info = detect_audio_onsets(audio, fs, label_config)
    label_source = str(getattr(eog_label_config, "label_source", "audio_anchored")).strip().lower()
    if label_source in {"eog", "eog_events", "eog_only"}:
        eog_info = detect_eog_label_events(
            eog_raw,
            fs,
            eog_label_config,
        )
        label_convention = (
            "sample_labels indexes class_names; transitions occur at detected EOG activity ends; "
            "audio_cue_onset_samples are visual references only"
        )
    elif label_source in {"audio_anchored", "audio_anchored_eog_offset"}:
        eog_info = detect_eog_offsets_after_audio_cues(
            eog_raw,
            fs,
            audio_onset_info["onset_samples"],
            eog_label_config,
        )
        label_convention = (
            "sample_labels indexes class_names; transitions occur at EOG activity end after each audio cue; "
            "cue_onset_samples stores those EOG-derived transition samples"
        )
    else:
        raise ValueError(
            "EogOffsetLabelConfig.label_source must be 'audio_anchored' or 'eog_events', "
            f"got {eog_label_config.label_source!r}."
        )
    eog_derivative = normalized_signal_derivative(eog_raw)
    eog_onset_info = {
        "onset_samples": eog_info["label_event_samples"],
        "onset_times_sec": eog_info["label_event_samples"].astype(np.float64) / float(fs),
        "peak_values": eog_info["peak_values"],
    }
    sample_labels, cue_table = labels_from_audio_onsets(
        n_samples=eeg.shape[0],
        fs=fs,
        onset_info=eog_onset_info,
        config=label_config,
    )
    cue_table = cue_table.rename(
        columns={
            "onset_sample": "eog_label_event_sample",
            "onset_time_sec": "eog_label_event_time_sec",
            "peak_value": "eog_activity_peak_value",
        }
    )
    cue_table = eog_info["event_table"].merge(
        cue_table,
        on="cue_index",
        how="left",
        suffixes=("", "_label"),
    )

    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    cue_csv = output_npz.with_suffix(".cue_table.csv")
    cue_table.to_csv(cue_csv, index=False)

    np.savez(
        output_npz,
        eeg=eeg.astype(np.float32),
        eeg_raw=eeg_raw.astype(np.float32),
        eog=eog_raw.astype(np.float32),
        eog_raw=eog_raw.astype(np.float32),
        eog_normalized_derivative=eog_derivative.astype(np.float32),
        eog_detection_signal=np.asarray(eog_info["detection_signal"], dtype=np.float32),
        audio=audio.astype(np.float32),
        audio_envelope=np.asarray(audio_onset_info["envelope"], dtype=np.float64),
        audio_threshold=np.array(float(audio_onset_info["threshold"]), dtype=np.float64),
        audio_cue_onset_samples=np.asarray(audio_onset_info["onset_samples"], dtype=np.int64),
        audio_cue_peak_values=np.asarray(audio_onset_info["peak_values"], dtype=np.float64),
        eog_activity_score=np.asarray(eog_info["score"], dtype=np.float64),
        eog_activity_threshold=np.array(float(eog_info["threshold"]), dtype=np.float64),
        eog_activity_segments=np.asarray(eog_info["active_segments"], dtype=np.int64),
        eog_activity_start_samples=np.asarray(eog_info["event_start_samples"], dtype=np.int64),
        eog_activity_end_samples=np.asarray(eog_info["event_end_samples"], dtype=np.int64),
        eog_label_event_samples=np.asarray(eog_info["label_event_samples"], dtype=np.int64),
        eog_detection_signal_mode=np.array(str(eog_label_config.detection_signal_mode)),
        eog_derivative_lowpass_hz=np.array(
            np.nan
            if eog_label_config.derivative_lowpass_hz is None
            else float(eog_label_config.derivative_lowpass_hz),
            dtype=np.float64,
        ),
        eog_detection_source=np.array(
            "NeuroKit2 eog_clean/eog_findpeaks/eog_features; "
            f"clean={eog_label_config.neurokit_clean_method}; "
            f"peaks={eog_label_config.neurokit_peak_method}; "
            f"detection_signal={eog_label_config.detection_signal_mode}"
        ),
        eog_detection_reference=np.array(
            "NeuroKit2 EOG docs: https://neuropsychology.github.io/NeuroKit/functions/eog.html; "
            "Kleifges et al. 2017 BLINKER, Frontiers in Neuroscience 11:12"
        ),
        cue_onset_samples=np.asarray(eog_info["label_event_samples"], dtype=np.int64),
        cue_peak_values=np.asarray(eog_info["peak_values"], dtype=np.float64),
        sample_labels=sample_labels.astype(np.int64),
        samplerate=np.array(fs, dtype=np.int64),
        acquired_channels=np.asarray(channels, dtype=np.int64),
        eeg_channels=np.asarray(preprocess_config.eeg_channels, dtype=np.int64),
        eog_channels=np.asarray(eog_channels, dtype=np.int64),
        requested_eog_channels=np.asarray(getattr(preprocess_config, "eog_channels", ()), dtype=np.int64),
        missing_eog_channels=np.asarray(missing_eog_channels, dtype=np.int64),
        audio_channel=np.array(int(preprocess_config.audio_channel), dtype=np.int64),
        class_names=np.asarray(label_config.class_names),
        source_raw_npz=np.array(str(raw_npz)),
        segment_name=np.array(raw["segment_name"]),
        preprocess_config_json=np.array(json.dumps(_json_safe(asdict(preprocess_config)))),
        label_config_json=np.array(json.dumps(_json_safe(asdict(label_config)))),
        eog_labeling_config_json=np.array(json.dumps(_json_safe(asdict(eog_label_config)))),
        cue_table_csv=np.array(str(cue_csv)),
        label_convention=np.array(label_convention),
    )
    return output_npz, cue_table


__all__ = [
    "EogOffsetLabelConfig",
    "detect_eog_label_events",
    "detect_eog_offsets_after_audio_cues",
    "eog_activity_score",
    "prepare_eog_detection_signal",
    "preprocess_recording_with_eog_offset_labels",
]
