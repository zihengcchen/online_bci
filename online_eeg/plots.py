"""Plotting helpers for labeled recordings and prediction overlays."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    from .config import PathLike
    from .preprocessing import _cue_onset_times_for_plot, load_labeled_recording
except ImportError:
    from config import PathLike
    from preprocessing import _cue_onset_times_for_plot, load_labeled_recording


def plot_labeled_recording(
    labeled_npz: PathLike,
    max_duration_sec: Optional[float] = 30.0,
    channel_names: Optional[Sequence[str]] = None,
    use_raw_eeg: bool = False,
):
    import matplotlib.pyplot as plt

    rec = load_labeled_recording(labeled_npz)
    eeg = rec["eeg_raw"] if use_raw_eeg and rec.get("eeg_raw") is not None else rec["eeg"]
    audio = rec.get("audio")
    acquired_channels = tuple(rec.get("acquired_channels") or ())
    eeg_channels = tuple(rec.get("eeg_channels") or range(1, eeg.shape[1] + 1))
    audio_channel = rec.get("audio_channel")
    labels = rec["sample_labels"]
    fs = int(rec["samplerate"])
    n = len(labels)
    if max_duration_sec is not None:
        n = min(n, int(round(float(max_duration_sec) * fs)))
    t = np.arange(n) / float(fs)
    cue_times = _cue_onset_times_for_plot(rec, max_duration_sec)

    channel_traces: List[Tuple[int, str, np.ndarray]] = []
    for idx in range(eeg.shape[1]):
        hardware_channel = int(eeg_channels[idx]) if idx < len(eeg_channels) else idx + 1
        channel_label = (
            str(channel_names[idx])
            if channel_names is not None and idx < len(channel_names)
            else f"EEG channel {hardware_channel}"
        )
        channel_traces.append((hardware_channel, channel_label, eeg[:n, idx]))
    if audio is not None:
        hardware_channel = int(audio_channel) if audio_channel is not None else len(channel_traces) + 1
        channel_traces.append((hardware_channel, f"Audio channel {hardware_channel}", np.asarray(audio)[:n]))

    if acquired_channels:
        channel_order = {int(ch): idx for idx, ch in enumerate(acquired_channels)}
        channel_traces.sort(key=lambda item: channel_order.get(int(item[0]), int(item[0])))
    else:
        channel_traces.sort(key=lambda item: int(item[0]))

    fig, axes = plt.subplots(
        len(channel_traces),
        1,
        sharex=True,
        figsize=(14, max(2.5, 1.6 * len(channel_traces))),
        squeeze=False,
    )
    axes_flat = axes.reshape(-1)
    for ax, (_, label, trace) in zip(axes_flat, channel_traces):
        ax.plot(t, trace, linewidth=0.7)
        for cue_time in cue_times:
            ax.axvline(cue_time, linestyle=":", color="tab:red", alpha=0.85, linewidth=2.0)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes_flat[-1].set_xlabel("Time (s)")
    axes_flat[0].set_title(f"Recording channels: {Path(labeled_npz).name}")
    plt.tight_layout()
    return fig, axes_flat

def plot_predictions_overlay(
    labeled_npz: PathLike,
    predictions: Union[pd.DataFrame, PathLike],
    max_duration_sec: Optional[float] = None,
    channel_names: Sequence[str] = ("O1", "Oz", "O2", "POz"),
    use_raw_eeg: bool = False,
    show_true_labels: bool = False,
    legend_loc: str = "upper left",
):
    import matplotlib.pyplot as plt

    rec = load_labeled_recording(labeled_npz)
    pred_df = pd.read_csv(predictions) if not isinstance(predictions, pd.DataFrame) else predictions.copy()
    eeg = rec["eeg_raw"] if use_raw_eeg and rec.get("eeg_raw") is not None else rec["eeg"]
    labels = rec["sample_labels"]
    fs = int(rec["samplerate"])

    n = len(labels)
    if max_duration_sec is not None:
        n = min(n, int(round(float(max_duration_sec) * fs)))
        pred_df = pred_df[pred_df["end_time_sec"] <= float(max_duration_sec)].copy()

    t = np.arange(n) / float(fs)
    n_channels = min(4, eeg.shape[1])
    fig, axes = plt.subplots(
        n_channels,
        1,
        sharex=True,
        figsize=(14, max(3.0, 1.8 * n_channels)),
        squeeze=False,
    )
    axes_flat = axes.reshape(-1)

    class_names = tuple(rec.get("class_names") or ())
    pred_axes = []
    pred_t = np.asarray([], dtype=np.float64)
    pred = np.asarray([], dtype=np.float64)
    if len(pred_df):
        pred_t = pred_df["end_time_sec"].to_numpy(dtype=float)
        pred = pred_df["pred_label"].to_numpy(dtype=float)
    cue_times = _cue_onset_times_for_plot(rec, max_duration_sec)

    cue_color = "tab:red"
    prediction_color = "tab:orange"
    true_label_color = "tab:blue"
    legend_handles: List[Any] = []
    legend_labels: List[str] = []

    for idx, ax in enumerate(axes_flat):
        name = str(channel_names[idx]) if idx < len(channel_names) else f"Ch {idx + 1}"
        eeg_line, = ax.plot(t, eeg[:n, idx], linewidth=0.7, color="black", label="EEG")
        if idx == 0:
            legend_handles.append(eeg_line)
            legend_labels.append("EEG")

        for cue_idx, cue_time in enumerate(cue_times):
            cue_line = ax.axvline(
                cue_time,
                linestyle=":",
                color=cue_color,
                alpha=0.9,
                linewidth=2.2,
                label="cue onset" if idx == 0 and cue_idx == 0 else None,
                zorder=3,
            )
            if idx == 0 and cue_idx == 0:
                legend_handles.append(cue_line)
                legend_labels.append("cue onset")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)

        pred_ax = ax.twinx()
        pred_axes.append(pred_ax)
        if show_true_labels and len(labels):
            true_line, = pred_ax.step(
                t,
                labels[:n].astype(float),
                where="post",
                linewidth=1.4,
                linestyle="--",
                color=true_label_color,
                alpha=0.8,
                label="true label",
                zorder=2,
            )
            if idx == 0:
                legend_handles.append(true_line)
                legend_labels.append("true label")
        if len(pred_t):
            pred_line, = pred_ax.step(
                pred_t,
                pred,
                where="post",
                linewidth=1.6,
                color=prediction_color,
                alpha=0.9,
                label="prediction",
            )
            if idx == 0:
                legend_handles.append(pred_line)
                legend_labels.append("prediction")
        if class_names:
            pred_ax.set_yticks(np.arange(len(class_names)))
            pred_ax.set_yticklabels([str(name) for name in class_names])
            pred_ax.set_ylim(-0.5, max(0.5, len(class_names) - 0.5))
        elif len(pred):
            pred_ax.set_ylim(float(np.min(pred)) - 0.5, float(np.max(pred)) + 0.5)
        pred_ax.tick_params(axis="y", colors=prediction_color)

    axes_flat[-1].set_xlabel("Time (s)")
    axes_flat[0].set_title(f"Realtime predictions over EEG channels: {Path(labeled_npz).name}")
    if legend_handles:
        axes_flat[0].legend(legend_handles, legend_labels, loc=legend_loc)
    plt.tight_layout()
    return fig, axes_flat

def plot_xcov_curve(
    xcov_curve: pd.DataFrame,
    xcov_delay_summary: pd.DataFrame,
    title: str,
    figsize: Tuple[float, float] = (12, 5),
    legend_loc: str = "center right",
):
    """Plot an xcov curve and mark the peak lag used as the delay estimate."""

    import matplotlib.pyplot as plt

    if xcov_curve is None or xcov_delay_summary is None:
        raise ValueError("xcov_curve and xcov_delay_summary are required.")
    if len(xcov_curve) == 0 or len(xcov_delay_summary) == 0:
        raise ValueError("xcov_curve and xcov_delay_summary cannot be empty.")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(xcov_curve["lag_sec"], xcov_curve["xcov_coeff"], linewidth=1.8)
    peak_delay = float(xcov_delay_summary.loc[0, "xcov_delay_sec"])
    peak_coeff = float(xcov_delay_summary.loc[0, "xcov_peak_coeff"])
    ax.axvline(
        peak_delay,
        linestyle=":",
        color="tab:red",
        linewidth=2.0,
        label=f"peak lag = {peak_delay:.3f} s",
    )
    ax.scatter([peak_delay], [peak_coeff], color="tab:red", zorder=3)
    ax.axvline(0.0, linestyle="--", color="black", alpha=0.5, linewidth=1.0)
    ax.set_xlabel("Lag (s)", fontsize=14)
    ax.set_ylabel("Normalized xcov coefficient", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc=legend_loc)
    plt.tight_layout()
    return fig, ax

def _offline_variant_mask(
    summary: pd.DataFrame,
    window_sec: float,
    stride_sec: float,
    label_mode: str,
    feature_mode: Optional[str],
) -> np.ndarray:
    required = {"window_sec", "stride_sec", "label_mode"}
    if feature_mode is not None:
        required.add("feature_mode")
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise KeyError(f"Summary is missing required columns: {missing}.")

    mask = (
        np.isclose(pd.to_numeric(summary["window_sec"], errors="coerce"), float(window_sec))
        & np.isclose(pd.to_numeric(summary["stride_sec"], errors="coerce"), float(stride_sec))
        & (summary["label_mode"].astype(str).str.lower() == str(label_mode).lower())
    )
    if feature_mode is not None:
        mask = mask & (summary["feature_mode"].astype(str).str.lower() == str(feature_mode).lower())
    return np.asarray(mask, dtype=bool)

def select_offline_variant_by_settings(
    summary: pd.DataFrame,
    window_sec: float,
    stride_sec: float,
    label_mode: str,
    feature_mode: Optional[str] = "filtered_signal",
) -> pd.Series:
    """Return the offline sweep row matching one window/stride/label setting."""

    mask = _offline_variant_mask(summary, window_sec, stride_sec, label_mode, feature_mode)
    matches = summary.loc[mask].copy()
    if matches.empty:
        available_columns = [
            col for col in ("feature_mode", "window_sec", "stride_sec", "label_mode", "variant")
            if col in summary.columns
        ]
        available = summary[available_columns].drop_duplicates().sort_values(available_columns).head(20)
        raise ValueError(
            "No offline variant matched "
            f"feature_mode={feature_mode!r}, window_sec={window_sec}, "
            f"stride_sec={stride_sec}, label_mode={label_mode!r}. "
            f"First available settings:\n{available.to_string(index=False)}"
        )
    return matches.iloc[0]

def plot_offline_variant_trace_and_xcov(
    summary: pd.DataFrame,
    labeled_npz: PathLike,
    window_sec: float,
    stride_sec: float,
    label_mode: str,
    feature_mode: Optional[str] = "filtered_signal",
    channel_names: Sequence[str] = ("O1", "Oz", "O2", "POz"),
    max_duration_sec: Optional[float] = None,
    show_true_labels: bool = True,
    legend_loc: str = "center right",
    title_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """Load one offline sweep variant by settings and plot trace plus xcov.

    This is intended for notebook comparisons: keep the ranked best/lowest
    sections unchanged, then call this helper with any window, stride, and label
    mode you want to inspect side by side.
    """

    try:
        from .testing import load_test_variant_artifacts
    except ImportError:
        from testing import load_test_variant_artifacts

    row = select_offline_variant_by_settings(
        summary=summary,
        window_sec=window_sec,
        stride_sec=stride_sec,
        label_mode=label_mode,
        feature_mode=feature_mode,
    )
    artifacts = load_test_variant_artifacts(row, labeled_npz)
    variant = str(row.get("variant", "selected variant"))
    if title_prefix is None:
        title_prefix = (
            f"Offline variant window={float(window_sec):g}s, "
            f"stride={float(stride_sec):g}s, labels={label_mode}"
        )

    trace_fig, trace_axes = plot_predictions_overlay(
        labeled_npz,
        artifacts["predictions"],
        max_duration_sec=max_duration_sec,
        channel_names=channel_names,
        show_true_labels=show_true_labels,
        legend_loc=legend_loc,
    )
    trace_axes[0].set_title(f"{title_prefix}: {variant}")

    xcov_fig = None
    xcov_ax = None
    if artifacts["xcov_curve"] is not None and artifacts["xcov_delay_summary"] is not None:
        xcov_fig, xcov_ax = plot_xcov_curve(
            artifacts["xcov_curve"],
            artifacts["xcov_delay_summary"],
            title=f"Prediction xcov lag for {title_prefix}: {variant}",
            legend_loc=legend_loc,
        )

    return {
        "row": row,
        "artifacts": artifacts,
        "trace_fig": trace_fig,
        "trace_axes": trace_axes,
        "xcov_fig": xcov_fig,
        "xcov_ax": xcov_ax,
    }


__all__ = [
    "plot_labeled_recording",
    "plot_predictions_overlay",
    "plot_xcov_curve",
    "plot_offline_variant_trace_and_xcov",
    "select_offline_variant_by_settings",
]
