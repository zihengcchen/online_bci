"""Dataset windowing, LSTM training, validation, and offline sweeps."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
        canonical_normalization_mode,
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
        canonical_normalization_mode,
        make_prediction_aligned_eeg_tables_for_labeled_sources,
    )

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent / "models"


def train_lstm(
    bundle: DatasetBundle,
    training_config: TrainingConfig,
    output_dir: PathLike,
    initial_checkpoint_path: Optional[PathLike] = None,
    checkpoint_name: str = "lstm_checkpoint.pt",
    extra_checkpoint_metadata: Optional[Dict[str, Any]] = None,
    models_dir: Optional[PathLike] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Train and validate the LSTM, then save checkpoint and CSV artifacts."""

    set_seed(int(training_config.seed), deterministic=bool(training_config.deterministic))
    output_dir = ensure_dir(output_dir)
    device = _device(training_config)

    input_size = int(bundle.X_train.shape[-1])
    num_classes = int(max(len(bundle.class_names), np.max(bundle.y_train) + 1, np.max(bundle.y_val) + 1))
    initial_checkpoint = None
    if initial_checkpoint_path is None:
        model_config = {
            "input_size": input_size,
            "hidden_size": int(training_config.hidden_size),
            "num_layers": int(training_config.num_layers),
            "num_classes": num_classes,
            "dropout": float(training_config.dropout),
        }
        model = LSTMClassifier(**model_config).to(device)
    else:
        model, initial_checkpoint, device = load_checkpoint(initial_checkpoint_path, device=str(device))
        model_config = dict(initial_checkpoint["model_config"])
        if int(model_config.get("input_size", -1)) != input_size:
            raise ValueError(
                f"Checkpoint input_size={model_config.get('input_size')} does not match "
                f"dataset input_size={input_size}. Check channels/window settings."
            )
        if int(model_config.get("num_classes", -1)) != num_classes:
            raise ValueError(
                f"Checkpoint num_classes={model_config.get('num_classes')} does not match "
                f"dataset num_classes={num_classes}. Check class labels."
            )

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

    checkpoint_path = output_dir / str(checkpoint_name)
    model_catalog_dir = ensure_dir(DEFAULT_MODELS_DIR if models_dir is None else models_dir)
    model_catalog_name = model_name or model_checkpoint_name(
        bundle,
        training_config,
        checkpoint_name=checkpoint_name,
        continued_from=initial_checkpoint_path,
    )
    model_catalog_path = model_catalog_dir / model_catalog_name
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "window_config": asdict(bundle.window_config),
        "training_config": asdict(training_config),
        "normalizer_mean": bundle.normalizer_mean,
        "normalizer_std": bundle.normalizer_std,
        "normalizer_mode": canonical_normalization_mode(getattr(training_config, "normalization", "train_zscore")),
        "fs": int(bundle.fs),
        "class_names": tuple(bundle.class_names),
        "source_files": tuple(bundle.source_files),
        "final_val_summary": val_summary.to_dict(orient="records"),
        "history": history,
        "artifact_checkpoint_path": str(checkpoint_path),
        "model_catalog_name": str(model_catalog_name),
        "model_catalog_path": str(model_catalog_path),
        "models_dir": str(model_catalog_dir),
    }
    source_preprocess_configs = _labeled_preprocess_configs(bundle.source_files)
    unique_preprocess_configs = _unique_config_dicts(source_preprocess_configs)
    if source_preprocess_configs:
        checkpoint["source_preprocess_configs"] = source_preprocess_configs
    if len(unique_preprocess_configs) == 1:
        checkpoint["preprocess_config"] = unique_preprocess_configs[0]
    elif len(unique_preprocess_configs) > 1:
        checkpoint["preprocess_config_mismatch_warning"] = (
            "Training labeled files were created with multiple preprocessing configs."
        )
    if initial_checkpoint_path is not None:
        checkpoint["continued_from_checkpoint"] = str(initial_checkpoint_path)
        checkpoint["previous_checkpoint_source_files"] = tuple(initial_checkpoint.get("source_files", ())) if initial_checkpoint else tuple()
    if extra_checkpoint_metadata:
        checkpoint.update(extra_checkpoint_metadata)
    torch.save(checkpoint, checkpoint_path)
    if not _same_path(checkpoint_path, model_catalog_path):
        torch.save(checkpoint, model_catalog_path)

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
        "checkpoint_path": model_catalog_path,
        "artifact_checkpoint_path": checkpoint_path,
        "model_catalog_path": model_catalog_path,
        "model_catalog_name": model_catalog_name,
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
    models_dir: Optional[PathLike] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    bundle = build_train_val_dataset(labeled_npz_paths, window_config, training_config)
    result = train_lstm(
        bundle,
        training_config,
        output_dir,
        models_dir=models_dir,
        model_name=model_name,
    )
    result["dataset_bundle"] = bundle
    return result

def labeled_training_paths_for_runs(
    runs_root: PathLike,
    run_ids: Sequence[str],
    labeled_subdir: str = "labeled_training",
    pattern: str = "*.npz",
) -> List[Path]:
    """Return labeled training NPZ paths for one or more run folders."""

    runs_root = Path(runs_root)
    paths: List[Path] = []
    missing: List[str] = []
    for run_id in run_ids:
        labeled_dir = runs_root / str(run_id) / labeled_subdir
        run_paths = sorted(path for path in labeled_dir.glob(pattern) if path.is_file())
        if not run_paths:
            missing.append(str(labeled_dir))
        paths.extend(run_paths)
    if missing:
        raise FileNotFoundError(
            "No labeled training NPZ files found in: " + ", ".join(missing)
        )
    return paths

def _same_path(a: PathLike, b: PathLike) -> bool:
    try:
        return Path(a).resolve() == Path(b).resolve()
    except FileNotFoundError:
        return Path(a).absolute() == Path(b).absolute()

def _copy_checkpoint(source: PathLike, destination: PathLike) -> Path:
    source = Path(source)
    destination = Path(destination)
    if not source.exists():
        raise FileNotFoundError(f"Checkpoint not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not _same_path(source, destination):
        shutil.copy2(source, destination)
    return destination

def _append_general_model_log(log_csv: PathLike, row: Dict[str, Any]) -> Path:
    log_csv = Path(log_csv)
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    row = {key: value for key, value in row.items()}
    row["created_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    new_df = pd.DataFrame([row])
    if log_csv.exists():
        old_df = pd.read_csv(log_csv)
        new_df = pd.concat([old_df, new_df], ignore_index=True)
    new_df.to_csv(log_csv, index=False)
    return log_csv

def _compatible_window_config(checkpoint: Dict[str, Any], requested: Optional[WindowConfig]) -> WindowConfig:
    checkpoint_cfg = window_config_from_checkpoint(checkpoint)
    if requested is None:
        return checkpoint_cfg

    checks = (
        ("feature_mode", canonical_feature_mode(requested.feature_mode), canonical_feature_mode(checkpoint_cfg.feature_mode)),
        ("window_sec", float(requested.window_sec), float(checkpoint_cfg.window_sec)),
        ("stride_sec", float(requested.stride_sec), float(checkpoint_cfg.stride_sec)),
        ("label_mode", str(requested.label_mode).lower(), str(checkpoint_cfg.label_mode).lower()),
    )
    mismatches = []
    for name, requested_value, checkpoint_value in checks:
        if isinstance(requested_value, float):
            ok = np.isclose(requested_value, checkpoint_value)
        else:
            ok = requested_value == checkpoint_value
        if not ok:
            mismatches.append(f"{name}: requested={requested_value!r}, checkpoint={checkpoint_value!r}")
    if mismatches:
        raise ValueError(
            "General-model checkpoint settings do not match the requested window config: "
            + "; ".join(mismatches)
        )
    return requested

def initialize_general_model(
    source_checkpoint_path: PathLike,
    general_model_path: PathLike,
    snapshot_path: Optional[PathLike] = None,
    update_name: str = "",
    log_csv: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """Initialize ``general_model.pt`` by copying an existing run checkpoint."""

    general_model_path = _copy_checkpoint(source_checkpoint_path, general_model_path)
    snapshot = _copy_checkpoint(general_model_path, snapshot_path) if snapshot_path is not None else None
    log_path = None
    if log_csv is not None:
        log_path = _append_general_model_log(
            log_csv,
            {
                "mode": "initialized",
                "update_name": update_name,
                "source_checkpoint": str(source_checkpoint_path),
                "general_model_path": str(general_model_path),
                "snapshot_path": "" if snapshot is None else str(snapshot),
            },
        )
    return {
        "mode": "initialized",
        "general_model_path": general_model_path,
        "snapshot_path": snapshot,
        "log_csv": log_path,
    }

def update_general_model(
    labeled_npz_paths: Sequence[PathLike],
    general_model_path: PathLike,
    output_dir: PathLike,
    training_config: Optional[TrainingConfig] = None,
    window_config: Optional[WindowConfig] = None,
    snapshot_path: Optional[PathLike] = None,
    update_name: str = "",
    log_csv: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """Continue training ``general_model.pt`` using selected labeled run data.

    The existing checkpoint normalizer is reused so input scaling remains stable
    across updates. To reduce forgetting, pass labeled paths from multiple runs
    instead of only the newest run.
    """

    general_model_path = Path(general_model_path)
    if not general_model_path.exists():
        raise FileNotFoundError(f"General model checkpoint not found: {general_model_path}")
    output_dir = ensure_dir(output_dir)
    training_config = training_config or TrainingConfig()
    set_seed(int(training_config.seed), deterministic=bool(training_config.deterministic))

    _, checkpoint, _ = load_checkpoint(general_model_path, device=training_config.device)
    if "normalizer_mean" not in checkpoint or "normalizer_std" not in checkpoint:
        raise KeyError("General model checkpoint is missing normalizer_mean/normalizer_std.")
    win_cfg = _compatible_window_config(checkpoint, window_config)

    bundle = build_train_val_dataset(
        labeled_npz_paths,
        win_cfg,
        training_config,
        normalizer_mean=checkpoint["normalizer_mean"],
        normalizer_std=checkpoint["normalizer_std"],
    )
    result = train_lstm(
        bundle,
        training_config,
        output_dir,
        initial_checkpoint_path=general_model_path,
        checkpoint_name="general_model.pt",
        extra_checkpoint_metadata={
            "general_model_update": True,
            "general_model_update_name": str(update_name),
            "general_model_input_checkpoint": str(general_model_path),
        },
    )

    updated_checkpoint = Path(result["checkpoint_path"])
    _copy_checkpoint(updated_checkpoint, general_model_path)
    snapshot = _copy_checkpoint(general_model_path, snapshot_path) if snapshot_path is not None else None

    log_path = None
    if log_csv is not None:
        val_summary = result["validation_summary"].iloc[0].to_dict() if len(result["validation_summary"]) else {}
        log_path = _append_general_model_log(
            log_csv,
            {
                "mode": "continued_training",
                "update_name": update_name,
                "general_model_path": str(general_model_path),
                "update_checkpoint_path": str(updated_checkpoint),
                "snapshot_path": "" if snapshot is None else str(snapshot),
                "n_labeled_files": int(len(labeled_npz_paths)),
                "labeled_files": ";".join(str(path) for path in labeled_npz_paths),
                "window_sec": float(win_cfg.window_sec),
                "stride_sec": float(win_cfg.stride_sec),
                "label_mode": str(win_cfg.label_mode),
                "val_accuracy": val_summary.get("accuracy", np.nan),
                "val_balanced_accuracy": val_summary.get("balanced_accuracy", np.nan),
            },
        )

    result.update(
        {
            "mode": "continued_training",
            "general_model_path": general_model_path,
            "snapshot_path": snapshot,
            "log_csv": log_path,
            "dataset_bundle": bundle,
        }
    )
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

def _short_digest(text: str, length: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]

def _shorten_slug(text: str, max_len: int = 120) -> str:
    text = text.strip("_-") or "value"
    if len(text) <= max_len:
        return text
    digest = _short_digest(text)
    keep = max(1, max_len - len(digest) - 2)
    return f"{text[:keep].strip('_-')}__{digest}"

def _source_run_slug(path: PathLike) -> str:
    stem = Path(path).stem
    for suffix in ("_eog_offset_labeled", "_labeled"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if "__" in stem:
        stem = stem.split("__", 1)[0]
    return slugify_config_value(stem)

def _source_runs_slug(source_files: Sequence[PathLike]) -> str:
    run_slugs: List[str] = []
    seen = set()
    for source_file in source_files:
        slug = _source_run_slug(source_file)
        if slug and slug not in seen:
            seen.add(slug)
            run_slugs.append(slug)
    return _shorten_slug("--".join(run_slugs) if run_slugs else "unknown")

def model_checkpoint_name(
    bundle: DatasetBundle,
    training_config: TrainingConfig,
    checkpoint_name: str = "lstm_checkpoint.pt",
    continued_from: Optional[PathLike] = None,
) -> str:
    """Return a deterministic, descriptive checkpoint filename for ``models/``."""

    win = bundle.window_config
    source_key = "|".join(Path(path).stem for path in bundle.source_files)
    stem_parts = [
        f"runs_{_source_runs_slug(bundle.source_files)}",
        f"files_{slugify_config_value(len(bundle.source_files))}",
        f"src_{_short_digest(source_key)}",
        canonical_feature_mode(str(win.feature_mode)),
        f"win_{slugify_config_value(win.window_sec)}s",
        f"stride_{slugify_config_value(win.stride_sec)}s",
        f"labels_{slugify_config_value(win.label_mode)}",
        f"norm_{slugify_config_value(canonical_normalization_mode(getattr(training_config, 'normalization', 'train_zscore')))}",
        f"h{slugify_config_value(training_config.hidden_size)}",
        f"layers_{slugify_config_value(training_config.num_layers)}",
        f"drop_{slugify_config_value(training_config.dropout)}",
        f"epochs_{slugify_config_value(training_config.epochs)}",
        f"lr_{slugify_config_value(training_config.lr)}",
        f"batch_{slugify_config_value(training_config.batch_size)}",
        f"trainfrac_{slugify_config_value(training_config.train_fraction)}",
        f"seed_{slugify_config_value(training_config.seed)}",
    ]
    if continued_from is not None:
        stem_parts.append(f"continued_{slugify_config_value(Path(continued_from).stem)}")
    stem = _shorten_slug("__".join(stem_parts), max_len=180)
    suffix = Path(str(checkpoint_name)).suffix or ".pt"
    return f"{stem}{suffix}"

def _labeled_preprocess_configs(source_files: Sequence[PathLike]) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for source_file in source_files:
        try:
            labeled = np.load(Path(source_file), allow_pickle=True)
        except FileNotFoundError:
            continue
        try:
            if "preprocess_config_json" not in labeled.files:
                continue
            raw_json = str(np.asarray(labeled["preprocess_config_json"]).item())
            if raw_json:
                configs.append(json.loads(raw_json))
        finally:
            labeled.close()
    return configs

def _unique_config_dicts(configs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique: List[Dict[str, Any]] = []
    seen = set()
    for config in configs:
        key = json.dumps(config, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(config)
    return unique

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

def _optional_row_path(row: Any, key: str) -> Optional[Path]:
    value = row.get(key, None)
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or ";" in text:
        return None
    return Path(text)

def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (FileNotFoundError, ValueError):
        try:
            path.absolute().relative_to(root.absolute())
            return True
        except ValueError:
            return False

def prune_offline_sweep_artifacts(
    ranked_summary: pd.DataFrame,
    top_n: Optional[int],
    summary_csv: Optional[PathLike] = None,
    sweep_dir: Optional[PathLike] = None,
    remove_model_catalog: bool = True,
) -> Dict[str, Any]:
    """Keep only the top-N ranked offline sweep variants and remove the rest.

    The input summary should already be ranked in the order to keep. The main
    summary CSV is overwritten with the kept rows when ``summary_csv`` is given.
    Removed variant directories are only deleted when they are inside
    ``sweep_dir`` or the summary CSV's parent directory.
    """

    if top_n is None:
        return {
            "summary": ranked_summary.copy().reset_index(drop=True),
            "removed_summary": ranked_summary.iloc[0:0].copy(),
            "removed_variant_dirs": [],
            "removed_model_paths": [],
        }
    top_n = int(top_n)
    if top_n <= 0:
        raise ValueError("top_n must be positive or None.")

    ranked = ranked_summary.copy().reset_index(drop=True)
    kept = ranked.head(top_n).copy().reset_index(drop=True)
    removed = ranked.iloc[top_n:].copy().reset_index(drop=True)

    summary_path = Path(summary_csv) if summary_csv is not None else None
    safe_sweep_root = Path(sweep_dir) if sweep_dir is not None else (summary_path.parent if summary_path is not None else None)

    kept_variant_dirs = {
        path.resolve()
        for _, row in kept.iterrows()
        for path in [_optional_row_path(row, "variant_dir")]
        if path is not None
    }
    kept_model_paths = {
        path.resolve()
        for _, row in kept.iterrows()
        for key in ("model_catalog_path", "checkpoint_path")
        for path in [_optional_row_path(row, key)]
        if path is not None
    }

    removed_variant_dirs: List[str] = []
    removed_model_paths: List[str] = []
    for _, row in removed.iterrows():
        variant_dir = _optional_row_path(row, "variant_dir")
        if variant_dir is not None:
            resolved_variant_dir = variant_dir.resolve()
            if (
                resolved_variant_dir not in kept_variant_dirs
                and safe_sweep_root is not None
                and _path_is_within(variant_dir, safe_sweep_root)
                and variant_dir.exists()
            ):
                shutil.rmtree(variant_dir)
                removed_variant_dirs.append(str(variant_dir))

        if remove_model_catalog:
            for key in ("model_catalog_path", "checkpoint_path"):
                model_path = _optional_row_path(row, key)
                if model_path is None:
                    continue
                resolved_model_path = model_path.resolve()
                if resolved_model_path in kept_model_paths:
                    continue
                if model_path.exists() and model_path.is_file() and _path_is_within(model_path, DEFAULT_MODELS_DIR):
                    model_path.unlink()
                    removed_model_paths.append(str(model_path))

    if summary_path is not None:
        kept.to_csv(summary_path, index=False)

    return {
        "summary": kept,
        "removed_summary": removed,
        "removed_variant_dirs": removed_variant_dirs,
        "removed_model_paths": removed_model_paths,
    }

def _as_path_list(paths: Union[PathLike, Sequence[PathLike]], name: str) -> List[Path]:
    if isinstance(paths, (str, Path)):
        result = [Path(paths)]
    else:
        result = [Path(path) for path in paths]
    if not result:
        raise ValueError(f"{name} cannot be empty.")
    return result

def _nanmean_from_rows(rows: Sequence[Dict[str, Any]], key: str) -> float:
    values = np.asarray([row.get(key, np.nan) for row in rows], dtype=np.float64)
    return float(np.nanmean(values)) if np.any(np.isfinite(values)) else np.nan

def _nansum_from_rows(rows: Sequence[Dict[str, Any]], key: str) -> float:
    values = np.asarray([row.get(key, np.nan) for row in rows], dtype=np.float64)
    return float(np.nansum(values)) if values.size else np.nan

def _first_nonempty_from_rows(rows: Sequence[Dict[str, Any]], key: str) -> str:
    for row in rows:
        value = row.get(key, "")
        if isinstance(value, str) and value:
            return value
    return ""

def offline_train_test_sweep(
    train_labeled_npz: Union[PathLike, Sequence[PathLike]],
    test_labeled_npz: Union[PathLike, Sequence[PathLike]],
    output_dir: PathLike,
    feature_modes: Sequence[str] = ("filtered_signal",),
    window_secs: Sequence[float] = (1.0, 1.5, 2.0),
    stride_secs: Sequence[float] = (0.2,),
    training_config: Optional[TrainingConfig] = None,
    label_mode: Any = "endpoint",
    label_modes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Train offline time-domain EEG model variants and test on labeled trial(s).

    ``label_modes`` can include both ``"endpoint"`` and ``"majority"`` to make
    window-labeling strategy part of the sweep. ``label_mode`` is kept for
    existing single-mode calls, and can also receive a sequence for backwards
    compatibility with exploratory notebooks. ``train_labeled_npz`` and
    ``test_labeled_npz`` each accept either one path or a sequence of paths.
    """

    output_dir = ensure_dir(output_dir)
    training_config = training_config or TrainingConfig()
    train_labeled_npz_paths = _as_path_list(train_labeled_npz, "train_labeled_npz")
    test_labeled_npz_paths = _as_path_list(test_labeled_npz, "test_labeled_npz")
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
                        labeled_npz_paths=train_labeled_npz_paths,
                        output_dir=variant_dir,
                        window_config=win_cfg,
                        training_config=training_config,
                    )
                    test_run_rows: List[Dict[str, Any]] = []
                    for test_path in test_labeled_npz_paths:
                        test_result = predict_labeled_recording(
                            labeled_npz=test_path,
                            checkpoint_path=train_result["checkpoint_path"],
                            output_dir=variant_dir,
                            batch_size=training_config.batch_size,
                        )
                        test_summary = test_result["summary"].iloc[0].to_dict()
                        cue_delay_summary = test_result["cue_delay_summary"].iloc[0].to_dict()
                        xcov_delay_summary = test_result["xcov_delay_summary"].iloc[0].to_dict()
                        test_run_rows.append(
                            {
                                "test_labeled_npz": str(test_path),
                                "test_stem": Path(test_path).stem,
                                "test_prediction_csv": str(test_result.get("prediction_csv", "")),
                                "test_aligned_prediction_csv": str(test_result.get("aligned_prediction_csv", "")),
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
                        )
                    test_runs_summary = pd.DataFrame(test_run_rows)
                    test_runs_summary_csv = variant_dir / "test_runs_summary.csv"
                    test_runs_summary.to_csv(test_runs_summary_csv, index=False)

                    val_summary = train_result["validation_summary"].iloc[0].to_dict()
                    row = {
                        "variant": variant_name,
                        "feature_mode": win_cfg.feature_mode,
                        "label_mode": win_cfg.label_mode,
                        "window_sec": win_cfg.window_sec,
                        "stride_sec": win_cfg.stride_sec,
                        "normalization": canonical_normalization_mode(getattr(training_config, "normalization", "train_zscore")),
                        "checkpoint_path": str(train_result["checkpoint_path"]),
                        "model_catalog_path": str(train_result.get("model_catalog_path", train_result["checkpoint_path"])),
                        "model_catalog_name": str(train_result.get("model_catalog_name", Path(train_result["checkpoint_path"]).name)),
                        "artifact_checkpoint_path": str(train_result.get("artifact_checkpoint_path", "")),
                        "variant_dir": str(variant_dir),
                        "n_train_labeled_files": int(len(train_labeled_npz_paths)),
                        "train_labeled_files": ";".join(str(path) for path in train_labeled_npz_paths),
                        "n_test_labeled_files": int(len(test_labeled_npz_paths)),
                        "test_labeled_files": ";".join(str(path) for path in test_labeled_npz_paths),
                        "test_runs_summary_csv": str(test_runs_summary_csv),
                        "validation_aligned_prediction_csv": str(train_result.get("validation_aligned_prediction_csv", "")),
                        "test_prediction_csv": ";".join(item["test_prediction_csv"] for item in test_run_rows),
                        "test_aligned_prediction_csv": ";".join(item["test_aligned_prediction_csv"] for item in test_run_rows),
                        "val_accuracy": val_summary.get("accuracy", np.nan),
                        "val_balanced_accuracy": val_summary.get("balanced_accuracy", np.nan),
                        "test_accuracy": _nanmean_from_rows(test_run_rows, "test_accuracy"),
                        "test_balanced_accuracy": _nanmean_from_rows(test_run_rows, "test_balanced_accuracy"),
                        "test_n_windows": _nansum_from_rows(test_run_rows, "test_n_windows"),
                        "test_mean_cue_to_first_correct_sec": _nanmean_from_rows(test_run_rows, "test_mean_cue_to_first_correct_sec"),
                        "test_median_cue_to_first_correct_sec": _nanmean_from_rows(test_run_rows, "test_median_cue_to_first_correct_sec"),
                        "test_mean_cue_to_predicted_transition_sec": _nanmean_from_rows(test_run_rows, "test_mean_cue_to_predicted_transition_sec"),
                        "test_median_cue_to_predicted_transition_sec": _nanmean_from_rows(test_run_rows, "test_median_cue_to_predicted_transition_sec"),
                        "test_mean_cue_to_sustained_prediction_sec": _nanmean_from_rows(test_run_rows, "test_mean_cue_to_sustained_prediction_sec"),
                        "test_median_cue_to_sustained_prediction_sec": _nanmean_from_rows(test_run_rows, "test_median_cue_to_sustained_prediction_sec"),
                        "test_xcov_delay_sec": _nanmean_from_rows(test_run_rows, "test_xcov_delay_sec"),
                        "test_xcov_peak_coeff": _nanmean_from_rows(test_run_rows, "test_xcov_peak_coeff"),
                        "test_xcov_signal_column": _first_nonempty_from_rows(test_run_rows, "test_xcov_signal_column"),
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
    "initialize_general_model",
    "labeled_training_paths_for_runs",
    "load_checkpoint",
    "model_checkpoint_name",
    "offline_train_test_sweep",
    "prune_offline_sweep_artifacts",
    "update_general_model",
    "rank_sweep_by_causal_delay",
    "rank_sweep_summary",
    "select_lowest_causal_delay_variant",
    "train_lstm",
    "train_validate_pipeline",
    "window_config_from_checkpoint",
]
