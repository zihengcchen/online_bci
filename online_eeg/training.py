"""Windowing and LSTM training entry points."""

try:
    from .real_subject_pipeline import (
        DatasetBundle,
        LSTMClassifier,
        TrainingConfig,
        WindowConfig,
        build_train_val_dataset,
        load_checkpoint,
        train_lstm,
        train_validate_pipeline,
        window_config_from_checkpoint,
    )
except ImportError:
    from real_subject_pipeline import (
        DatasetBundle,
        LSTMClassifier,
        TrainingConfig,
        WindowConfig,
        build_train_val_dataset,
        load_checkpoint,
        train_lstm,
        train_validate_pipeline,
        window_config_from_checkpoint,
    )

__all__ = [
    "DatasetBundle",
    "LSTMClassifier",
    "TrainingConfig",
    "WindowConfig",
    "build_train_val_dataset",
    "load_checkpoint",
    "train_lstm",
    "train_validate_pipeline",
    "window_config_from_checkpoint",
]
