"""LSTM model definitions, checkpoint loading, and inference helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from .config import PathLike, TrainingConfig, WindowConfig
    from .windowing import canonical_feature_mode
except ImportError:
    from config import PathLike, TrainingConfig, WindowConfig
    from windowing import canonical_feature_mode


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
    """Return window settings from saved checkpoints."""

    if "window_config" in checkpoint:
        cfg = dict(checkpoint["window_config"])
        cfg["feature_mode"] = canonical_feature_mode(cfg.get("feature_mode", "filtered_signal"))
        allowed = set(WindowConfig.__dataclass_fields__)
        cfg = {key: value for key, value in cfg.items() if key in allowed}
        return WindowConfig(**cfg)

    saved_cfg = dict(checkpoint.get("config", {}))
    fs = int(checkpoint.get("fs", saved_cfg.get("fs", 200)))
    feature_mode = canonical_feature_mode(checkpoint.get("feature_mode", saved_cfg.get("feature_mode", "filtered_signal")))

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
    )


__all__ = [
    "WindowDataset",
    "LSTMClassifier",
    "predict_array",
    "predict_single_window",
    "load_checkpoint",
    "class_names_from_checkpoint",
    "window_config_from_checkpoint",
]
