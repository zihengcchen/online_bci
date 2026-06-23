"""Realtime MP150 prediction loop and live plotting internals."""

from __future__ import annotations

import csv
import json
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from .acquisition import _import_mp150_class
    from .config import AcquisitionConfig, PathLike, PreprocessConfig
    from .modeling import (
        class_names_from_checkpoint,
        load_checkpoint,
        predict_single_window,
        window_config_from_checkpoint,
    )
    from .preprocessing import preprocess_eeg_signal
    from .utils import _as_2d_samples_channels, _checked_acquisition_chunk, _duration_to_sample_count, _now_string, ensure_dir, extract_hardware_channels, probability_column_map
    from .windowing import apply_normalizer, extract_window_features, make_prediction_aligned_eeg_table
except ImportError:
    from acquisition import _import_mp150_class
    from config import AcquisitionConfig, PathLike, PreprocessConfig
    from modeling import (
        class_names_from_checkpoint,
        load_checkpoint,
        predict_single_window,
        window_config_from_checkpoint,
    )
    from preprocessing import preprocess_eeg_signal
    from utils import _as_2d_samples_channels, _checked_acquisition_chunk, _duration_to_sample_count, _now_string, ensure_dir, extract_hardware_channels, probability_column_map
    from windowing import apply_normalizer, extract_window_features, make_prediction_aligned_eeg_table



def _format_realtime_prediction_status(row: Dict[str, Any], class_names: Sequence[str]) -> str:
    pred_label = int(row["pred_label"])
    pred_name = class_names[pred_label] if 0 <= pred_label < len(class_names) else f"class_{pred_label}"
    probability_parts = []
    for label_idx, name in enumerate(class_names):
        key = f"prob_{name}"
        if key in row:
            probability_parts.append(f"{name}={float(row[key]):.3f}")
        else:
            probability_parts.append(f"class_{label_idx}=nan")
    return (
        f"t={float(row['end_time_sec']):7.2f}s "
        f"pred={pred_label} ({pred_name}) "
        f"probabilities: {', '.join(probability_parts)}"
    )

def _setup_realtime_plot(
    acquired_channels: Sequence[int],
    class_names: Sequence[str],
) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    plt.ion()
    n_channels = len(acquired_channels)
    fig, axes = plt.subplots(
        n_channels + 1,
        1,
        sharex=True,
        figsize=(14, max(4.0, 1.45 * (n_channels + 1))),
        squeeze=False,
    )
    axes_flat = axes.reshape(-1)

    channel_lines = []
    for ax, channel in zip(axes_flat[:-1], acquired_channels):
        line, = ax.plot([], [], linewidth=0.7)
        channel_lines.append(line)
        ax.set_ylabel(f"Ch {int(channel)}")
        ax.grid(True, alpha=0.3)

    pred_line, = axes_flat[-1].step([], [], where="post", linewidth=1.8, color="tab:red")
    axes_flat[-1].set_ylabel("Prediction")
    axes_flat[-1].set_xlabel("Time (s)")
    axes_flat[-1].grid(True, alpha=0.3)
    if class_names:
        axes_flat[-1].set_yticks(np.arange(len(class_names)))
        axes_flat[-1].set_yticklabels([str(name) for name in class_names])
        axes_flat[-1].set_ylim(-0.5, max(0.5, len(class_names) - 0.5))

    axes_flat[0].set_title("Realtime channels and prediction")
    plt.tight_layout()

    display_handle = None
    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            from IPython.display import display

            display_handle = display(fig, display_id=True)
    except Exception:
        display_handle = None

    if display_handle is None:
        try:
            backend = str(plt.get_backend()).lower()
            if "agg" not in backend:
                fig.show()
        except Exception:
            pass

    return {
        "fig": fig,
        "axes": axes_flat,
        "channel_lines": channel_lines,
        "prediction_line": pred_line,
        "display_handle": display_handle,
        "plt": plt,
    }

def _set_signal_axis_limits(ax: Any, trace: np.ndarray) -> None:
    finite = np.asarray(trace, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return
    ymin = float(np.min(finite))
    ymax = float(np.max(finite))
    if np.isclose(ymin, ymax):
        pad = max(1.0, abs(ymin) * 0.1)
    else:
        pad = (ymax - ymin) * 0.1
    ax.set_ylim(ymin - pad, ymax + pad)

def _update_realtime_plot(
    plot_state: Optional[Dict[str, Any]],
    raw_data: np.ndarray,
    fs: int,
    prediction_rows: Sequence[Dict[str, Any]],
    class_names: Sequence[str],
    live_plot_window_sec: Optional[float],
) -> None:
    if plot_state is None:
        return

    raw = _as_2d_samples_channels(raw_data)
    if raw.shape[0] == 0:
        return

    axes = plot_state["axes"]
    end_time = raw.shape[0] / float(fs)
    if live_plot_window_sec is None:
        start_time = 0.0
    else:
        start_time = max(0.0, end_time - float(live_plot_window_sec))
    start_sample = max(0, int(np.floor(start_time * int(fs))))
    t = np.arange(start_sample, raw.shape[0], dtype=np.float64) / float(fs)
    visible = raw[start_sample:]

    for idx, line in enumerate(plot_state["channel_lines"]):
        if idx >= visible.shape[1]:
            line.set_data([], [])
            continue
        line.set_data(t, visible[:, idx])
        _set_signal_axis_limits(axes[idx], visible[:, idx])

    pred_times = []
    pred_labels = []
    previous_label = None
    for row in prediction_rows:
        row_time = float(row["end_time_sec"])
        row_label = int(row["pred_label"])
        if row_time < start_time:
            previous_label = row_label
            continue
        pred_times.append(row_time)
        pred_labels.append(row_label)

    if previous_label is not None:
        pred_times.insert(0, start_time)
        pred_labels.insert(0, previous_label)

    pred_line = plot_state["prediction_line"]
    pred_line.set_data(np.asarray(pred_times, dtype=np.float64), np.asarray(pred_labels, dtype=np.float64))
    if class_names:
        axes[-1].set_ylim(-0.5, max(0.5, len(class_names) - 0.5))
    elif pred_labels:
        ymin = min(pred_labels)
        ymax = max(pred_labels)
        axes[-1].set_ylim(float(ymin) - 0.5, float(ymax) + 0.5)

    if end_time <= start_time:
        end_time = start_time + 1.0
    for ax in axes:
        ax.set_xlim(start_time, end_time)

    fig = plot_state["fig"]
    fig.canvas.draw_idle()
    display_handle = plot_state.get("display_handle")
    if display_handle is not None:
        try:
            display_handle.update(fig)
        except Exception:
            pass
    if display_handle is None:
        try:
            backend = str(plot_state["plt"].get_backend()).lower()
            if "agg" not in backend:
                plot_state["plt"].pause(0.001)
        except Exception:
            pass

def run_realtime_mp150_prediction(
    output_dir: PathLike,
    checkpoint_path: PathLike,
    acquisition_config: AcquisitionConfig,
    preprocess_config: PreprocessConfig,
    duration_sec: Optional[float] = 60.0,
    trial_name: str = "realtime_trial",
    print_every_prediction: bool = True,
    live_plot: bool = False,
    live_plot_window_sec: Optional[float] = 15.0,
    live_plot_update_sec: Optional[float] = None,
    prediction_stride_sec: Optional[float] = None,
    prediction_flush_every: Optional[int] = 10,
) -> Dict[str, Any]:
    """Stream MP150 data and emit causal model predictions in real time.

    The full raw trial is saved continuously. Model predictions use the
    checkpoint's window length and feature settings. ``prediction_stride_sec``
    can override how often the window advances at test time. Prediction CSV
    writes are flushed every ``prediction_flush_every`` rows; use 0 or None to
    rely on normal file buffering until the trial ends.
    """

    output_dir = ensure_dir(output_dir)
    raw_path = output_dir / f"{trial_name}_raw.npz"
    prediction_csv = output_dir / f"{trial_name}_realtime_predictions.csv"
    aligned_prediction_csv = output_dir / f"{trial_name}_realtime_predictions_aligned_eeg.csv"

    model, checkpoint, device = load_checkpoint(checkpoint_path)
    win_cfg = window_config_from_checkpoint(checkpoint)
    fs = int(checkpoint["fs"])
    if int(acquisition_config.samplerate) != fs:
        raise ValueError(
            f"Samplerate mismatch: acquisition fs={acquisition_config.samplerate}, checkpoint fs={fs}."
        )
    acquired_channels = tuple(int(ch) for ch in acquisition_config.channels)
    if not acquired_channels:
        raise ValueError("acquisition_config.channels must contain at least one channel.")
    chunk_samples = _duration_to_sample_count(
        acquisition_config.chunk_sec, fs, "acquisition_config.chunk_sec"
    )
    target_samples = (
        _duration_to_sample_count(duration_sec, fs)
        if duration_sec is not None
        else None
    )

    window_samples = int(round(float(win_cfg.window_sec) * fs))
    stride_sec = float(win_cfg.stride_sec if prediction_stride_sec is None else prediction_stride_sec)
    stride_samples = int(round(stride_sec * fs))
    class_names = class_names_from_checkpoint(checkpoint)
    if stride_samples <= 0 or window_samples <= 0:
        raise ValueError("window_sec and prediction stride must produce positive sample counts.")
    plot_update_sec = float(stride_sec if live_plot_update_sec is None else live_plot_update_sec)
    if live_plot and plot_update_sec <= 0:
        raise ValueError("live_plot_update_sec must be positive.")
    flush_every = int(prediction_flush_every or 0)
    if flush_every < 0:
        raise ValueError("prediction_flush_every must be non-negative or None.")

    MP150 = _import_mp150_class()
    mp = MP150(samplerate=fs, channels=list(acquired_channels))

    fieldnames = [
        "wall_time_sec",
        "end_sample",
        "end_time_sec",
        "pred_label",
    ] + [f"prob_{name}" for name in class_names]

    raw_chunks: List[np.ndarray] = []
    chunk_start_wall: List[float] = []
    chunk_end_wall: List[float] = []
    prediction_rows: List[Dict[str, Any]] = []
    eeg_buffer = np.empty((0, len(preprocess_config.eeg_channels)), dtype=np.float32)
    buffer_start_sample = 0
    total_samples_seen = 0
    next_prediction_end = window_samples
    start_wall = time.time()
    live_plot_state = _setup_realtime_plot(acquired_channels, class_names) if live_plot else None
    last_live_plot_wall = 0.0
    printed_prediction_line = False

    print(
        f"Realtime prediction: feature_mode={win_cfg.feature_mode}, "
        f"window={float(win_cfg.window_sec):.3f}s, stride={float(stride_sec):.3f}s",
        flush=True,
    )

    prediction_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(prediction_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        predictions_since_flush = 0

        try:
            while target_samples is None or total_samples_seen < target_samples:
                if target_samples is not None:
                    remaining_samples = target_samples - total_samples_seen
                    if remaining_samples <= 0:
                        break
                    request_samples = min(chunk_samples, remaining_samples)
                    request_sec = request_samples / float(fs)
                else:
                    request_sec = float(acquisition_config.chunk_sec)

                chunk_start_wall.append(time.time() - start_wall)
                chunk = _checked_acquisition_chunk(mp.get_chunk(request_sec), len(acquired_channels))
                chunk_end_wall.append(time.time() - start_wall)
                if target_samples is not None:
                    remaining_samples = target_samples - total_samples_seen
                    if chunk.shape[0] > remaining_samples:
                        chunk = chunk[:remaining_samples]
                raw_chunks.append(chunk.astype(np.float32))

                eeg_raw = extract_hardware_channels(
                    chunk, acquired_channels, preprocess_config.eeg_channels
                )
                eeg_buffer = np.concatenate([eeg_buffer, eeg_raw.astype(np.float32)], axis=0)
                total_samples_seen += int(chunk.shape[0])

                while next_prediction_end <= total_samples_seen:
                    window_start_global = next_prediction_end - window_samples
                    local_start = window_start_global - buffer_start_sample
                    local_end = next_prediction_end - buffer_start_sample
                    if local_start < 0 or local_end > len(eeg_buffer):
                        break

                    raw_window = eeg_buffer[local_start:local_end]
                    model_window = preprocess_eeg_signal(
                        raw_window,
                        fs=fs,
                        preprocess_config=preprocess_config,
                    )
                    X_raw = extract_window_features(model_window, fs=fs, config=win_cfg)[None, ...]
                    X = apply_normalizer(X_raw, checkpoint["normalizer_mean"], checkpoint["normalizer_std"])
                    pred, prob = predict_single_window(model, X, device=device)

                    row = {
                        "wall_time_sec": time.time() - start_wall,
                        "end_sample": int(next_prediction_end),
                        "end_time_sec": float(next_prediction_end) / float(fs),
                        "pred_label": int(pred[0]),
                    }
                    for key, value in probability_column_map(prob, class_names).items():
                        row[key] = float(value[0])
                    writer.writerow(row)
                    predictions_since_flush += 1
                    if flush_every > 0 and predictions_since_flush >= flush_every:
                        f.flush()
                        predictions_since_flush = 0
                    prediction_rows.append(row)

                    if print_every_prediction:
                        status = _format_realtime_prediction_status(row, class_names)
                        sys.stdout.write("\r" + status.ljust(160))
                        sys.stdout.flush()
                        printed_prediction_line = True
                    next_prediction_end += stride_samples

                keep_from_global = max(0, next_prediction_end - window_samples)
                drop = keep_from_global - buffer_start_sample
                if drop > 0:
                    eeg_buffer = eeg_buffer[drop:]
                    buffer_start_sample = keep_from_global

                if live_plot_state is not None:
                    now_wall = time.time()
                    if now_wall - last_live_plot_wall >= plot_update_sec:
                        _update_realtime_plot(
                            live_plot_state,
                            np.concatenate(raw_chunks, axis=0),
                            fs=fs,
                            prediction_rows=prediction_rows,
                            class_names=class_names,
                            live_plot_window_sec=live_plot_window_sec,
                        )
                        last_live_plot_wall = now_wall
        finally:
            try:
                f.flush()
            except Exception:
                pass
            if printed_prediction_line:
                sys.stdout.write("\n")
                sys.stdout.flush()
            mp.close()
            if live_plot_state is not None and raw_chunks:
                _update_realtime_plot(
                    live_plot_state,
                    np.concatenate(raw_chunks, axis=0),
                    fs=fs,
                    prediction_rows=prediction_rows,
                    class_names=class_names,
                    live_plot_window_sec=live_plot_window_sec,
                )

    raw_data = np.concatenate(raw_chunks, axis=0).astype(np.float32) if raw_chunks else np.empty(
        (0, len(acquired_channels)), dtype=np.float32
    )
    prediction_df = pd.DataFrame(prediction_rows, columns=fieldnames)

    np.savez(
        raw_path,
        data=raw_data,
        samplerate=np.array(fs, dtype=np.int64),
        channels=np.asarray(acquired_channels, dtype=np.int64),
        time_sec=np.arange(raw_data.shape[0], dtype=np.float64) / float(fs),
        segment_name=np.array(str(trial_name)),
        chunk_start_wall_sec=np.asarray(chunk_start_wall, dtype=np.float64),
        chunk_end_wall_sec=np.asarray(chunk_end_wall, dtype=np.float64),
        requested_duration_sec=np.array(np.nan if duration_sec is None else float(duration_sec), dtype=np.float64),
        created_at=np.array(_now_string()),
        metadata_json=np.array(json.dumps({
            "mode": "realtime_prediction",
            "checkpoint_path": str(checkpoint_path),
            "feature_mode": str(win_cfg.feature_mode),
            "window_sec": float(win_cfg.window_sec),
            "window_samples": int(window_samples),
            "checkpoint_stride_sec": float(win_cfg.stride_sec),
            "prediction_stride_sec": float(stride_sec),
            "prediction_stride_samples": int(stride_samples),
            "prediction_flush_every": int(flush_every),
            "class_names": list(class_names),
        })),
    )


    if raw_data.shape[0]:
        eeg_for_alignment = preprocess_eeg_signal(
            extract_hardware_channels(raw_data, acquired_channels, preprocess_config.eeg_channels),
            fs=fs,
            preprocess_config=preprocess_config,
        )
    else:
        eeg_for_alignment = np.empty((0, len(preprocess_config.eeg_channels)), dtype=np.float32)
    aligned_prediction_df = make_prediction_aligned_eeg_table(
        eeg_for_alignment,
        prediction_df,
        fs=fs,
        eeg_channels=preprocess_config.eeg_channels,
    )
    aligned_prediction_df.to_csv(aligned_prediction_csv, index=False)

    return {
        "raw_path": raw_path,
        "prediction_csv": prediction_csv,
        "aligned_prediction_csv": aligned_prediction_csv,
        "predictions": prediction_df,
        "aligned_predictions": aligned_prediction_df,
    }


__all__ = [
    "run_realtime_mp150_prediction",
]
