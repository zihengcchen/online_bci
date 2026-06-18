"""Preprocessing and audio-cue labeling entry points."""

try:
    from .real_subject_pipeline import (
        AudioLabelConfig,
        PreprocessConfig,
        detect_audio_onsets,
        labels_from_audio_onsets,
        load_labeled_recording,
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
        preprocess_many_recordings,
        preprocess_recording,
        rolling_rms_causal,
    )

__all__ = [
    "AudioLabelConfig",
    "PreprocessConfig",
    "detect_audio_onsets",
    "labels_from_audio_onsets",
    "load_labeled_recording",
    "preprocess_many_recordings",
    "preprocess_recording",
    "rolling_rms_causal",
]
