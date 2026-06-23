"""Dataset windowing, LSTM training, validation, and offline sweeps."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from .config import DatasetBundle, PathLike, TrainingConfig, WindowConfig
    from .metrics import classification_summary
    from .modeling import (
        LSTMClassifier,
        WindowDataset,
        _device,
        class_names_from_checkpoint,
        load_checkpoint,
        predict_array,
        window_config_from_checkpoint,
    )
    from .testing import predict_labeled_recording
    from .utils import add_probability_columns, drop_single_value_columns, ensure_dir, save_json, set_seed
    from .windowing import (
        build_train_val_dataset,
        canonical_feature_mode,
        make_prediction_aligned_eeg_tables_for_labeled_sources,
    )
except ImportError:
    from config import DatasetBundle, PathLike, TrainingConfig, WindowConfig
    from metrics import classification_summary
    from modeling import (
        LSTMClassifier,
        WindowDataset,
        _device,
        class_names_from_checkpoint,
        load_checkpoint,
        predict_array,
        window_config_from_checkpoint,
    )
    from testing import predict_labeled_recording
    from utils import add_probability_columns, drop_single_value_columns, ensure_dir, save_json, set_seed
    from windowing import (
        build_train_val_dataset,
        canonical_feature_mode,
        make_prediction_aligned_eeg_tables_for_labeled_sources,
    )


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
    add_probability_columns(val_predictions, val_prob, bundle.class_names)

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
    val_aligned_predictions_csv = output_dir / "validation_predictions_aligned_eeg.csv"
    val_summary_csv = output_dir / "validation_summary.csv"
    val_per_class_csv = output_dir / "validation_per_class.csv"
    metadata_json = output_dir / "checkpoint_metadata.json"

    val_aligned_predictions = make_prediction_aligned_eeg_tables_for_labeled_sources(
        val_predictions,
        bundle.source_files,
    )

    history_df.to_csv(history_csv, index=False)
    drop_single_value_columns(val_predictions, ("recording_id", "source_file")).to_csv(val_predictions_csv, index=False)
    val_aligned_predictions.to_csv(val_aligned_predictions_csv, index=False)
    val_summary.to_csv(val_summary_csv, index=False)
    val_per_class.to_csv(val_per_class_csv, index=False)
    save_json({k: v for k, v in checkpoint.items() if k != "model_state_dict"}, metadata_json)

    return {
        "model": model,
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "history": history_df,
        "validation_predictions": val_predictions,
        "validation_aligned_predictions": val_aligned_predictions,
        "validation_summary": val_summary,
        "validation_per_class": val_per_class,
        "validation_aligned_prediction_csv": val_aligned_predictions_csv,
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

def _coerce_numeric_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out

def rank_sweep_summary(
    summary: pd.DataFrame,
    rank_column: str = "test_xcov_peak_coeff",
) -> pd.DataFrame:
    """Rank sweep rows by a primary metric, then test/validation accuracy."""

    if rank_column not in summary.columns:
        raise KeyError(f"Missing ranking column: {rank_column}")
    ranked = _coerce_numeric_columns(
        summary,
        (rank_column, "test_balanced_accuracy", "val_balanced_accuracy"),
    )
    return (
        ranked
        .sort_values(
            [rank_column, "test_balanced_accuracy", "val_balanced_accuracy"],
            ascending=[False, False, False],
            na_position="last",
        )
        .reset_index(drop=True)
    )

def rank_sweep_by_causal_delay(summary: pd.DataFrame) -> pd.DataFrame:
    """Rank sweep rows by valid nonnegative xcov delay, then quality metrics.

    Negative xcov delay means the prediction trace leads the label trace. Those
    rows are kept for review but ranked after rows with nonnegative delay.
    """

    ranked = _coerce_numeric_columns(
        summary,
        (
            "test_xcov_delay_sec",
            "test_xcov_peak_coeff",
            "test_balanced_accuracy",
            "val_balanced_accuracy",
        ),
    )
    ranked["delay_rank_group"] = 2
    ranked.loc[ranked["test_xcov_delay_sec"] >= 0, "delay_rank_group"] = 0
    ranked.loc[ranked["test_xcov_delay_sec"] < 0, "delay_rank_group"] = 1
    return (
        ranked
        .sort_values(
            [
                "delay_rank_group",
                "test_xcov_delay_sec",
                "test_xcov_peak_coeff",
                "test_balanced_accuracy",
                "val_balanced_accuracy",
            ],
            ascending=[True, True, False, False, False],
            na_position="last",
        )
        .reset_index(drop=True)
    )

def select_lowest_causal_delay_variant(summary: pd.DataFrame) -> pd.Series:
    """Return the sweep row with the lowest valid nonnegative xcov delay."""

    ranked = rank_sweep_by_causal_delay(summary)
    causal = ranked[ranked["delay_rank_group"] == 0]
    if causal.empty:
        raise ValueError(
            "No valid nonnegative xcov delays were found. Review the full summary "
            "for negative-delay variants or failed xcov estimates."
        )
    return causal.iloc[0]

def offline_train_test_sweep(
    train_labeled_npz: PathLike,
    test_labeled_npz: PathLike,
    output_dir: PathLike,
    feature_modes: Sequence[str] = ("filtered_signal",),
    window_secs: Sequence[float] = (1.0, 1.5, 2.0),
    stride_secs: Sequence[float] = (0.2,),
    training_config: Optional[TrainingConfig] = None,
    label_mode: Any = "endpoint",
    label_modes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Train offline time-domain EEG model variants and test each on a labeled trial.

    ``label_modes`` can include both ``"endpoint"`` and ``"majority"`` to make
    window-labeling strategy part of the sweep. ``label_mode`` is kept for
    existing single-mode calls, and can also receive a sequence for backwards
    compatibility with exploratory notebooks.
    """

    output_dir = ensure_dir(output_dir)
    training_config = training_config or TrainingConfig()
    if label_modes is None:
        if isinstance(label_mode, str):
            label_modes = (label_mode,)
        else:
            label_modes = tuple(label_mode)

    label_modes = tuple(str(mode).lower() for mode in label_modes)
    if not label_modes:
        raise ValueError("label_modes cannot be empty.")
    invalid_modes = [mode for mode in label_modes if mode not in {"endpoint", "majority"}]
    if invalid_modes:
        raise ValueError(
            "label_modes must contain only 'endpoint' and/or 'majority'; "
            f"got {invalid_modes!r}."
        )

    rows: List[Dict[str, Any]] = []
    result_dirs: List[Path] = []

    for feature_mode in feature_modes:
        feature_mode = canonical_feature_mode(str(feature_mode))
        for mode in label_modes:
            for window_sec in window_secs:
                for stride_sec in stride_secs:
                    win_cfg = WindowConfig(
                        feature_mode=feature_mode,
                        window_sec=float(window_sec),
                        stride_sec=float(stride_sec),
                        label_mode=mode,
                    )
                    variant_name = "__".join(
                        [
                            feature_mode,
                            f"win_{slugify_config_value(window_sec)}s",
                            f"stride_{slugify_config_value(stride_sec)}s",
                            f"labels_{slugify_config_value(mode)}",
                        ]
                    )
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
                        "label_mode": win_cfg.label_mode,
                        "window_sec": win_cfg.window_sec,
                        "stride_sec": win_cfg.stride_sec,
                        "checkpoint_path": str(train_result["checkpoint_path"]),
                        "variant_dir": str(variant_dir),
                        "validation_aligned_prediction_csv": str(train_result.get("validation_aligned_prediction_csv", "")),
                        "test_aligned_prediction_csv": str(test_result.get("aligned_prediction_csv", "")),
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


__all__ = [
    "DatasetBundle",
    "LSTMClassifier",
    "TrainingConfig",
    "WindowConfig",
    "build_train_val_dataset",
    "class_names_from_checkpoint",
    "load_checkpoint",
    "offline_train_test_sweep",
    "rank_sweep_by_causal_delay",
    "rank_sweep_summary",
    "select_lowest_causal_delay_variant",
    "train_lstm",
    "train_validate_pipeline",
    "window_config_from_checkpoint",
]
