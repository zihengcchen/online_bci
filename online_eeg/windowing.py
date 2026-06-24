"""Sliding-window labeling, feature extraction, and prediction/sample alignment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from .config import DatasetBundle, PathLike, TrainingConfig, WindowConfig
    from .preprocessing import load_labeled_recording
    from .utils import _as_2d_samples_channels, drop_single_value_columns
except ImportError:
    from config import DatasetBundle, PathLike, TrainingConfig, WindowConfig
    from preprocessing import load_labeled_recording
    from utils import _as_2d_samples_channels, drop_single_value_columns


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

def canonical_feature_mode(feature_mode: str) -> str:
    """Return the canonical feature-mode name for preprocessed EEG windows."""

    mode = str(feature_mode).lower()
    if mode == "raw_signal":
        return "filtered_signal"
    if mode == "filtered_signal":
        return mode
    raise ValueError("feature_mode must be 'filtered_signal'.")

def extract_window_features(window: np.ndarray, fs: int, config: WindowConfig) -> np.ndarray:
    """Return the preprocessed time-domain EEG window used by the model."""

    canonical_feature_mode(config.feature_mode)
    return _as_2d_samples_channels(window).astype(np.float32)

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

def make_prediction_aligned_eeg_table(
    eeg: np.ndarray,
    prediction_df: pd.DataFrame,
    fs: int,
    eeg_channels: Optional[Sequence[int]] = None,
    sample_labels: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Return one EEG sample per row with the latest available prediction held forward.

    A prediction at ``end_sample`` is treated as available starting at that
    sample, then copied through each following EEG row until the next
    prediction's ``end_sample``. Rows before the first prediction are left
    blank in prediction columns.
    """

    eeg = _as_2d_samples_channels(eeg).astype(np.float32)
    fs = int(fs)
    if fs <= 0:
        raise ValueError("fs must be positive.")

    n_samples, n_channels = eeg.shape
    table = pd.DataFrame(
        {
            "sample": np.arange(n_samples, dtype=np.int64),
            "time_sec": np.arange(n_samples, dtype=np.float64) / float(fs),
        }
    )

    channel_ids = (
        tuple(int(ch) for ch in eeg_channels)
        if eeg_channels is not None and len(eeg_channels) > 0
        else tuple(range(1, n_channels + 1))
    )
    used_columns = set(table.columns)
    for idx in range(n_channels):
        hardware_channel = channel_ids[idx] if idx < len(channel_ids) else idx + 1
        column = f"eeg_ch{hardware_channel}"
        if column in used_columns:
            column = f"{column}_{idx + 1}"
        used_columns.add(column)
        table[column] = eeg[:, idx]

    if sample_labels is not None:
        labels = np.asarray(sample_labels, dtype=np.int64).reshape(-1)
        true_label = pd.Series(pd.NA, index=table.index, dtype="Int64")
        n_label = min(n_samples, labels.size)
        if n_label:
            true_label.iloc[:n_label] = labels[:n_label]
        table["true_label"] = true_label

    pred_df = prediction_df.copy()
    fill_pairs: List[Tuple[str, str]] = []
    for source, target in (
        ("start_sample", "prediction_window_start_sample"),
        ("end_sample", "prediction_end_sample"),
        ("start_time_sec", "prediction_window_start_time_sec"),
        ("end_time_sec", "prediction_end_time_sec"),
        ("wall_time_sec", "prediction_wall_time_sec"),
        ("pred_label", "pred_label"),
    ):
        if source in pred_df.columns:
            fill_pairs.append((source, target))
    fill_pairs.extend((col, col) for col in pred_df.columns if str(col).startswith("prob_"))

    for _, target in fill_pairs:
        table[target] = np.nan

    if len(pred_df) and fill_pairs:
        if "end_sample" in pred_df.columns:
            end_samples = pd.to_numeric(pred_df["end_sample"], errors="coerce").to_numpy(dtype=float)
        elif "end_time_sec" in pred_df.columns:
            end_samples = (
                pd.to_numeric(pred_df["end_time_sec"], errors="coerce").to_numpy(dtype=float)
                * float(fs)
            )
        else:
            raise ValueError("prediction_df must contain end_sample or end_time_sec.")

        valid = np.isfinite(end_samples)
        pred_df = pred_df.loc[valid].copy()
        end_samples = np.rint(end_samples[valid]).astype(int)
        order = np.argsort(end_samples, kind="stable")
        pred_df = pred_df.iloc[order].reset_index(drop=True)
        end_samples = end_samples[order]

        for row_idx, row in pred_df.iterrows():
            start = min(max(int(end_samples[row_idx]), 0), n_samples)
            stop = n_samples
            if row_idx + 1 < len(end_samples):
                stop = min(max(int(end_samples[row_idx + 1]), start), n_samples)
            if stop <= start:
                continue
            for source, target in fill_pairs:
                column_idx = table.columns.get_loc(target)
                table.iloc[start:stop, column_idx] = row[source]

    for int_column in (
        "prediction_window_start_sample",
        "prediction_end_sample",
        "pred_label",
    ):
        if int_column in table.columns:
            table[int_column] = pd.to_numeric(table[int_column], errors="coerce").round().astype("Int64")

    if "true_label" in table.columns and "pred_label" in table.columns:
        correct = pd.Series(pd.NA, index=table.index, dtype="boolean")
        valid = table["true_label"].notna() & table["pred_label"].notna()
        if valid.any():
            correct.loc[valid] = (
                table.loc[valid, "true_label"].astype(int).to_numpy()
                == table.loc[valid, "pred_label"].astype(int).to_numpy()
            )
        table["correct"] = correct

    return table

def make_prediction_aligned_eeg_tables_for_labeled_sources(
    prediction_df: pd.DataFrame,
    labeled_npz_paths: Sequence[PathLike],
) -> pd.DataFrame:
    """Build aligned EEG/prediction rows for one or more labeled recordings."""

    frames: List[pd.DataFrame] = []
    paths = [Path(path) for path in labeled_npz_paths]
    if not paths:
        return pd.DataFrame()

    for path in paths:
        rec = load_labeled_recording(path)
        if "source_file" in prediction_df.columns:
            source_file = prediction_df["source_file"].astype(str).map(lambda value: str(Path(value)))
            group = prediction_df[source_file == str(path)].copy()
        elif len(paths) == 1:
            group = prediction_df.copy()
        else:
            group = prediction_df.iloc[0:0].copy()
        if group.empty:
            continue

        aligned = make_prediction_aligned_eeg_table(
            rec["eeg"],
            group,
            fs=int(rec["samplerate"]),
            eeg_channels=rec.get("eeg_channels"),
            sample_labels=rec.get("sample_labels"),
        )
        aligned.insert(0, "recording_id", rec["segment_name"])
        aligned.insert(1, "source_file", str(path))
        frames.append(aligned)

    if not frames:
        return pd.DataFrame()
    return drop_single_value_columns(
        pd.concat(frames, ignore_index=True),
        ("recording_id", "source_file"),
    )

def build_train_val_dataset(
    labeled_npz_paths: Sequence[PathLike],
    window_config: WindowConfig,
    training_config: TrainingConfig,
    normalizer_mean: Optional[np.ndarray] = None,
    normalizer_std: Optional[np.ndarray] = None,
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

    if normalizer_mean is None or normalizer_std is None:
        mean, std = fit_normalizer(X_train_raw)
    else:
        mean = np.asarray(normalizer_mean, dtype=np.float32)
        std = np.asarray(normalizer_std, dtype=np.float32)
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


__all__ = [
    "WindowConfig",
    "DatasetBundle",
    "window_label",
    "canonical_feature_mode",
    "extract_window_features",
    "make_labeled_windows",
    "fit_normalizer",
    "apply_normalizer",
    "make_prediction_aligned_eeg_table",
    "make_prediction_aligned_eeg_tables_for_labeled_sources",
    "build_train_val_dataset",
]
