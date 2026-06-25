"""Short-trial prediction, post-hoc scoring, and testing orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    from .config import AcquisitionConfig, AudioLabelConfig, PathLike, PreprocessConfig
    from .metrics import (
        classification_summary,
        estimate_cue_prediction_delay,
        estimate_prediction_xcov_delay,
        estimate_transition_delay,
        summarize_cue_prediction_delay,
        summarize_transition_delay,
    )
    from .modeling import class_names_from_checkpoint, load_checkpoint, predict_array, window_config_from_checkpoint
    from .preprocessing import load_labeled_recording, preprocess_recording
    from .realtime import run_realtime_mp150_prediction
    from .utils import add_probability_columns, drop_single_value_columns, ensure_dir
    from .windowing import (
        apply_normalizer,
        make_labeled_windows,
        make_prediction_aligned_eeg_table,
        make_prediction_aligned_eeg_tables_for_labeled_sources,
        window_label,
    )
except ImportError:
    from config import AcquisitionConfig, AudioLabelConfig, PathLike, PreprocessConfig
    from metrics import (
        classification_summary,
        estimate_cue_prediction_delay,
        estimate_prediction_xcov_delay,
        estimate_transition_delay,
        summarize_cue_prediction_delay,
        summarize_transition_delay,
    )
    from modeling import class_names_from_checkpoint, load_checkpoint, predict_array, window_config_from_checkpoint
    from preprocessing import load_labeled_recording, preprocess_recording
    from realtime import run_realtime_mp150_prediction
    from utils import add_probability_columns, drop_single_value_columns, ensure_dir
    from windowing import (
        apply_normalizer,
        make_labeled_windows,
        make_prediction_aligned_eeg_table,
        make_prediction_aligned_eeg_tables_for_labeled_sources,
        window_label,
    )


def _prediction_analysis_tables(
    prediction_df: pd.DataFrame,
    rec: Dict[str, Any],
    class_names: tuple[str, ...],
) -> Dict[str, pd.DataFrame]:
    fs = int(rec["samplerate"])
    sample_labels = rec["sample_labels"]
    delay = estimate_transition_delay(prediction_df, sample_labels=sample_labels, fs=fs)
    cue_delay = estimate_cue_prediction_delay(
        prediction_df,
        sample_labels=sample_labels,
        cue_onset_samples=rec.get("cue_onset_samples", np.empty(0, dtype=np.int64)),
        fs=fs,
        class_names=class_names,
    )
    xcov_delay_summary, xcov_curve = estimate_prediction_xcov_delay(
        prediction_df,
        sample_labels=sample_labels,
        fs=fs,
        class_names=class_names,
        target_label=1,
    )
    return {
        "delay": delay,
        "delay_summary": summarize_transition_delay(delay),
        "cue_delay": cue_delay,
        "cue_delay_summary": summarize_cue_prediction_delay(cue_delay),
        "xcov_delay_summary": xcov_delay_summary,
        "xcov_curve": xcov_curve,
    }


def _aligned_prediction_table(rec: Dict[str, Any], prediction_df: pd.DataFrame) -> pd.DataFrame:
    return make_prediction_aligned_eeg_table(
        rec["eeg"],
        prediction_df,
        fs=int(rec["samplerate"]),
        eeg_channels=rec.get("eeg_channels"),
        sample_labels=rec.get("sample_labels"),
    )


def _analysis_csv_paths(output_dir: Path, stem: str, mode: str) -> Dict[str, Path]:
    return {
        "summary_csv": output_dir / f"{stem}_{mode}_summary.csv",
        "per_class_csv": output_dir / f"{stem}_{mode}_per_class.csv",
        "delay_csv": output_dir / f"{stem}_{mode}_delay_by_transition.csv",
        "delay_summary_csv": output_dir / f"{stem}_{mode}_delay_summary.csv",
        "cue_delay_csv": output_dir / f"{stem}_{mode}_cue_delay_by_cue.csv",
        "cue_delay_summary_csv": output_dir / f"{stem}_{mode}_cue_delay_summary.csv",
        "xcov_delay_summary_csv": output_dir / f"{stem}_{mode}_xcov_delay_summary.csv",
        "xcov_curve_csv": output_dir / f"{stem}_{mode}_xcov_curve.csv",
    }


def _write_analysis_csvs(tables: Dict[str, pd.DataFrame], paths: Dict[str, Path]) -> None:
    for table_key, path_key in (
        ("summary", "summary_csv"),
        ("per_class", "per_class_csv"),
        ("delay", "delay_csv"),
        ("delay_summary", "delay_summary_csv"),
        ("cue_delay", "cue_delay_csv"),
        ("cue_delay_summary", "cue_delay_summary_csv"),
        ("xcov_delay_summary", "xcov_delay_summary_csv"),
        ("xcov_curve", "xcov_curve_csv"),
    ):
        tables[table_key].to_csv(paths[path_key], index=False)

def _optional_path(value: Any, fallback: Path) -> Path:
    if value is None:
        return fallback
    try:
        if bool(pd.isna(value)):
            return fallback
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return Path(text) if text else fallback

def test_variant_artifact_paths(
    variant_row: Any,
    labeled_npz: PathLike,
) -> Dict[str, Path]:
    """Return standard test artifact paths for one offline sweep variant row."""

    row = variant_row.to_dict() if hasattr(variant_row, "to_dict") else dict(variant_row)
    variant_dir = Path(row["variant_dir"])
    stem = Path(labeled_npz).stem
    aligned_fallback = variant_dir / f"{stem}_test_predictions_aligned_eeg.csv"
    return {
        "variant_dir": variant_dir,
        "prediction_csv": variant_dir / f"{stem}_test_predictions.csv",
        "aligned_prediction_csv": _optional_path(row.get("test_aligned_prediction_csv"), aligned_fallback),
        "cue_delay_summary_csv": variant_dir / f"{stem}_test_cue_delay_summary.csv",
        "xcov_delay_summary_csv": variant_dir / f"{stem}_test_xcov_delay_summary.csv",
        "xcov_curve_csv": variant_dir / f"{stem}_test_xcov_curve.csv",
    }

def load_test_variant_artifacts(
    variant_row: Any,
    labeled_npz: PathLike,
) -> Dict[str, Any]:
    """Load the prediction tables used to inspect one offline sweep variant."""

    row = variant_row.to_dict() if hasattr(variant_row, "to_dict") else dict(variant_row)
    paths = test_variant_artifact_paths(row, labeled_npz)
    prediction_csv = paths["prediction_csv"]
    if not prediction_csv.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {prediction_csv}")

    def _read_optional_csv(path: Path) -> Optional[pd.DataFrame]:
        return pd.read_csv(path) if path.exists() else None

    return {
        "variant": row.get("variant", ""),
        "row": row,
        "paths": paths,
        "predictions": pd.read_csv(prediction_csv),
        "cue_delay_summary": _read_optional_csv(paths["cue_delay_summary_csv"]),
        "xcov_delay_summary": _read_optional_csv(paths["xcov_delay_summary_csv"]),
        "xcov_curve": _read_optional_csv(paths["xcov_curve_csv"]),
    }


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
    add_probability_columns(pred_df, prob, class_names)

    tables = {
        "summary": summary,
        "per_class": per_class,
        **_prediction_analysis_tables(pred_df, rec, class_names),
    }
    aligned_pred_df = _aligned_prediction_table(rec, pred_df)

    stem = Path(labeled_npz).stem
    pred_csv = output_dir / f"{stem}_test_predictions.csv"
    aligned_pred_csv = output_dir / f"{stem}_test_predictions_aligned_eeg.csv"
    paths = _analysis_csv_paths(output_dir, stem, "test")

    drop_single_value_columns(pred_df, ("recording_id", "source_file")).to_csv(pred_csv, index=False)
    aligned_pred_df.to_csv(aligned_pred_csv, index=False)
    _write_analysis_csvs(tables, paths)

    return {
        "predictions": pred_df,
        "aligned_predictions": aligned_pred_df,
        **tables,
        "prediction_csv": pred_csv,
        "aligned_prediction_csv": aligned_pred_csv,
        **paths,
    }


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
        class_names = tuple(rec["class_names"])
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

    valid_pred_df = pred_df[pred_df["valid_true_label"].astype(bool)] if "valid_true_label" in pred_df else pred_df
    tables = {
        "summary": summary,
        "per_class": per_class,
        **_prediction_analysis_tables(valid_pred_df, rec, class_names),
    }
    aligned_pred_df = _aligned_prediction_table(rec, pred_df)

    stem = Path(labeled_npz).stem
    evaluated_csv = output_dir / f"{stem}_realtime_predictions_evaluated.csv"
    aligned_evaluated_csv = output_dir / f"{stem}_realtime_predictions_aligned_eeg.csv"
    paths = _analysis_csv_paths(output_dir, stem, "realtime")

    drop_single_value_columns(
        pred_df,
        ("window_samples", "stride_samples", "feature_mode", "recording_id", "source_file"),
    ).to_csv(evaluated_csv, index=False)
    aligned_pred_df.to_csv(aligned_evaluated_csv, index=False)
    _write_analysis_csvs(tables, paths)

    return {
        "evaluated_predictions": pred_df,
        "aligned_predictions": aligned_pred_df,
        **tables,
        "evaluated_csv": evaluated_csv,
        "aligned_prediction_csv": aligned_evaluated_csv,
        **paths,
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
    prediction_window_sec: Optional[float] = None,
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
        prediction_window_sec=prediction_window_sec,
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


__all__ = [
    "collect_preprocess_and_test_trial",
    "evaluate_prediction_log_against_labeled_recording",
    "load_test_variant_artifacts",
    "estimate_transition_delay",
    "make_prediction_aligned_eeg_table",
    "make_prediction_aligned_eeg_tables_for_labeled_sources",
    "posthoc_analyze_realtime_trial",
    "test_variant_artifact_paths",
    "predict_labeled_recording",
    "run_realtime_mp150_prediction",
    "summarize_transition_delay",
]
