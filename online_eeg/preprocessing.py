"""Preprocessing and audio-cue labeling entry points."""

try:
    from .real_subject_pipeline import (
        AudioLabelConfig,
        PreprocessConfig,
        detect_audio_onsets,
        labels_from_audio_onsets,
        load_labeled_recording,
        preprocess_eeg_signal,
        preprocess_many_recordings,
        preprocess_recording,
        rolling_rms_causal,
    )
except ImportError:
    from real_subject_pipeline import (
        AudioLabelConfig,
        PreprocessConfig,
        detect_audio_onsets,
        labels_from_audio_onsets,
        load_labeled_recording,
        preprocess_eeg_signal,
        preprocess_many_recordings,
        preprocess_recording,
        rolling_rms_causal,
    )



def labeled_preprocess_summary(labeled_npz_paths):
    """Summarize preprocessing metadata saved inside labeled NPZ files."""

    import json
    import os
    from pathlib import Path

    import numpy as np
    import pandas as pd

    if isinstance(labeled_npz_paths, dict):
        items = [(str(name), Path(path)) for name, path in labeled_npz_paths.items()]
    elif isinstance(labeled_npz_paths, (str, os.PathLike)):
        path = Path(labeled_npz_paths)
        items = [(path.stem, path)]
    else:
        items = [(Path(path).stem, Path(path)) for path in labeled_npz_paths]

    rows = []
    for name, path in items:
        labeled = np.load(path, allow_pickle=True)
        try:
            files = set(labeled.files)
            preprocess_config = {}
            if "preprocess_config_json" in files:
                raw_json = str(np.asarray(labeled["preprocess_config_json"]).item())
                preprocess_config = json.loads(raw_json) if raw_json else {}

            fs = int(np.asarray(labeled["samplerate"]).item()) if "samplerate" in files else np.nan
            n_samples = int(labeled["eeg"].shape[0]) if "eeg" in files else np.nan
            duration_sec = float(n_samples) / float(fs) if np.isfinite(fs) and fs else np.nan
            has_apply_flag = "apply_software_filters" in preprocess_config
            rows.append(
                {
                    "name": name,
                    "path": str(path),
                    "samplerate": fs,
                    "duration_sec": duration_sec,
                    "has_preprocess_config": bool(preprocess_config),
                    "has_apply_software_filters_flag": bool(has_apply_flag),
                    "apply_software_filters": preprocess_config.get(
                        "apply_software_filters",
                        True if preprocess_config else np.nan,
                    ),
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

__all__ = [
    "AudioLabelConfig",
    "PreprocessConfig",
    "detect_audio_onsets",
    "labels_from_audio_onsets",
    "labeled_preprocess_summary",
    "load_labeled_recording",
    "preprocess_eeg_signal",
    "preprocess_many_recordings",
    "preprocess_recording",
    "rolling_rms_causal",
]
