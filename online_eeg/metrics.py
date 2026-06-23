"""Classification summaries and prediction-delay metrics."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def classification_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    n_classes = max(len(class_names), int(max(np.max(y_true), np.max(y_pred))) + 1 if len(y_true) else 0)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        if 0 <= truth < n_classes and 0 <= pred < n_classes:
            confusion[truth, pred] += 1

    rows = []
    recalls = []
    for label in range(n_classes):
        tp = int(confusion[label, label])
        support = int(confusion[label, :].sum())
        predicted = int(confusion[:, label].sum())
        recall = tp / support if support else np.nan
        precision = tp / predicted if predicted else np.nan
        recalls.append(recall)
        rows.append(
            {
                "label": label,
                "class_name": class_names[label] if label < len(class_names) else f"class_{label}",
                "support": support,
                "predicted": predicted,
                "precision": precision,
                "recall": recall,
            }
        )

    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) else np.nan
    finite_recalls = [r for r in recalls if np.isfinite(r)]
    balanced_accuracy = float(np.mean(finite_recalls)) if finite_recalls else np.nan

    summary = pd.DataFrame(
        [
            {
                "n_windows": int(len(y_true)),
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
            }
        ]
    )
    per_class = pd.DataFrame(rows)
    return summary, per_class

def estimate_transition_delay(
    prediction_df: pd.DataFrame,
    sample_labels: np.ndarray,
    fs: int,
    max_delay_sec: Optional[float] = 5.0,
) -> pd.DataFrame:
    labels = np.asarray(sample_labels, dtype=np.int64).reshape(-1)
    if len(labels) < 2 or len(prediction_df) == 0:
        return pd.DataFrame()

    true_change_samples = np.flatnonzero(np.diff(labels) != 0) + 1
    true_change_times = true_change_samples / float(fs)
    pred_df = prediction_df.sort_values("end_time_sec").reset_index(drop=True)
    pred_times = pred_df["end_time_sec"].to_numpy(dtype=float)
    pred_labels = pred_df["pred_label"].astype(int).to_numpy()

    pred_change_idx = np.flatnonzero(np.diff(pred_labels) != 0) + 1
    pred_change_times = pred_times[pred_change_idx]
    pred_change_to_labels = pred_labels[pred_change_idx]

    rows = []
    for i, sample_idx in enumerate(true_change_samples):
        true_t = float(true_change_times[i])
        from_label = int(labels[sample_idx - 1])
        to_label = int(labels[sample_idx])
        next_true_t = float(true_change_times[i + 1]) if i + 1 < len(true_change_times) else np.inf
        latest_t = next_true_t
        if max_delay_sec is not None:
            latest_t = min(latest_t, true_t + float(max_delay_sec))

        post_mask = (pred_times >= true_t) & (pred_times <= latest_t)
        post_idx = np.flatnonzero(post_mask)
        correct_idx = post_idx[pred_labels[post_idx] == to_label] if len(post_idx) else np.array([], dtype=int)

        if len(correct_idx):
            first_correct_time = float(pred_times[int(correct_idx[0])])
            delay_first_correct = first_correct_time - true_t
        else:
            first_correct_time = np.nan
            delay_first_correct = np.nan

        transition_mask = (
            (pred_change_times >= true_t)
            & (pred_change_times <= latest_t)
            & (pred_change_to_labels == to_label)
        )
        transition_idx = np.flatnonzero(transition_mask)
        if len(transition_idx):
            predicted_transition_time = float(pred_change_times[int(transition_idx[0])])
            delay_pred_transition = predicted_transition_time - true_t
        else:
            predicted_transition_time = np.nan
            delay_pred_transition = np.nan

        rows.append(
            {
                "true_transition_sample": int(sample_idx),
                "true_transition_time_sec": true_t,
                "from_label": from_label,
                "to_label": to_label,
                "transition_type": f"{from_label}->{to_label}",
                "first_correct_prediction_time_sec": first_correct_time,
                "delay_to_first_correct_prediction_sec": delay_first_correct,
                "predicted_transition_time_sec": predicted_transition_time,
                "delay_to_predicted_transition_sec": delay_pred_transition,
                "matched_first_correct_prediction": bool(np.isfinite(delay_first_correct)),
                "matched_predicted_transition": bool(np.isfinite(delay_pred_transition)),
            }
        )

    return pd.DataFrame(rows)

def _label_name(label: int, class_names: Sequence[str]) -> str:
    label = int(label)
    return str(class_names[label]) if 0 <= label < len(class_names) else f"class_{label}"

def _prediction_probability_column(
    prediction_df: pd.DataFrame,
    target_label: int,
    class_names: Sequence[str],
) -> Optional[str]:
    candidates: List[str] = []
    if 0 <= int(target_label) < len(class_names):
        candidates.append(f"prob_{class_names[int(target_label)]}")
    candidates.append(f"prob_class_{int(target_label)}")
    candidates.append(f"prob_{int(target_label)}")
    for column in candidates:
        if column in prediction_df.columns:
            return column
    return None

def estimate_cue_prediction_delay(
    prediction_df: pd.DataFrame,
    sample_labels: np.ndarray,
    cue_onset_samples: np.ndarray,
    fs: int,
    class_names: Sequence[str] = (),
    max_delay_sec: Optional[float] = 10.0,
    sustained_windows: int = 3,
) -> pd.DataFrame:
    """Measure delay from each audio cue onset to the corresponding prediction change."""

    labels = np.asarray(sample_labels, dtype=np.int64).reshape(-1)
    cues = np.asarray(cue_onset_samples, dtype=np.int64).reshape(-1)
    cues = cues[(cues >= 0) & (cues < len(labels))]
    if len(labels) == 0 or len(cues) == 0 or len(prediction_df) == 0:
        return pd.DataFrame()

    pred_df = prediction_df.sort_values("end_time_sec").reset_index(drop=True)
    pred_times = pred_df["end_time_sec"].to_numpy(dtype=float)
    pred_labels = pred_df["pred_label"].astype(int).to_numpy()
    if len(pred_times) == 0:
        return pd.DataFrame()

    pred_change_idx = np.flatnonzero(np.diff(pred_labels) != 0) + 1
    pred_change_times = pred_times[pred_change_idx]
    pred_change_to_labels = pred_labels[pred_change_idx]
    hold = max(1, int(sustained_windows))

    rows: List[Dict[str, Any]] = []
    for cue_idx, cue_sample in enumerate(cues):
        cue_sample = int(cue_sample)
        cue_t = cue_sample / float(fs)
        next_cue_sample = int(cues[cue_idx + 1]) if cue_idx + 1 < len(cues) else len(labels)
        next_cue_t = next_cue_sample / float(fs) if next_cue_sample < len(labels) else np.inf

        from_label = int(labels[cue_sample - 1]) if cue_sample > 0 else int(labels[cue_sample])
        target_sample = cue_sample
        if cue_sample > 0:
            search_stop = max(cue_sample + 1, min(next_cue_sample, len(labels)))
            changed = np.flatnonzero(labels[cue_sample:search_stop] != from_label)
            if len(changed):
                target_sample = cue_sample + int(changed[0])
        target_label = int(labels[target_sample])
        label_transition_t = target_sample / float(fs)

        latest_t = next_cue_t
        if max_delay_sec is not None:
            latest_t = min(latest_t, cue_t + float(max_delay_sec))

        post_idx = np.flatnonzero((pred_times >= cue_t) & (pred_times <= latest_t))
        correct_idx = post_idx[pred_labels[post_idx] == target_label] if len(post_idx) else np.array([], dtype=int)
        if len(correct_idx):
            first_correct_time = float(pred_times[int(correct_idx[0])])
            delay_first_correct = first_correct_time - cue_t
        else:
            first_correct_time = np.nan
            delay_first_correct = np.nan

        transition_mask = (
            (pred_change_times >= cue_t)
            & (pred_change_times <= latest_t)
            & (pred_change_to_labels == target_label)
        )
        transition_idx = np.flatnonzero(transition_mask)
        if len(transition_idx):
            predicted_transition_time = float(pred_change_times[int(transition_idx[0])])
            delay_pred_transition = predicted_transition_time - cue_t
        else:
            predicted_transition_time = np.nan
            delay_pred_transition = np.nan

        sustained_time = np.nan
        sustained_confirm_time = np.nan
        if len(post_idx):
            for idx in post_idx:
                idx = int(idx)
                end_idx = idx + hold
                if end_idx > len(pred_labels):
                    break
                if pred_times[end_idx - 1] > latest_t:
                    break
                if np.all(pred_labels[idx:end_idx] == target_label):
                    sustained_time = float(pred_times[idx])
                    sustained_confirm_time = float(pred_times[end_idx - 1])
                    break
        delay_sustained = sustained_time - cue_t if np.isfinite(sustained_time) else np.nan

        rows.append(
            {
                "cue_index": int(cue_idx),
                "cue_onset_sample": cue_sample,
                "cue_onset_time_sec": cue_t,
                "label_transition_sample": int(target_sample),
                "label_transition_time_sec": label_transition_t,
                "from_label": from_label,
                "from_label_name": _label_name(from_label, class_names),
                "target_label": target_label,
                "target_label_name": _label_name(target_label, class_names),
                "transition_type": f"{from_label}->{target_label}",
                "first_correct_prediction_time_sec": first_correct_time,
                "cue_to_first_correct_prediction_sec": delay_first_correct,
                "predicted_transition_time_sec": predicted_transition_time,
                "cue_to_predicted_transition_sec": delay_pred_transition,
                "sustained_prediction_time_sec": sustained_time,
                "sustained_prediction_confirm_time_sec": sustained_confirm_time,
                "cue_to_sustained_prediction_sec": delay_sustained,
                "sustained_windows": hold,
                "matched_first_correct_prediction": bool(np.isfinite(delay_first_correct)),
                "matched_predicted_transition": bool(np.isfinite(delay_pred_transition)),
                "matched_sustained_prediction": bool(np.isfinite(delay_sustained)),
            }
        )

    return pd.DataFrame(rows)

def summarize_cue_prediction_delay(cue_delay_df: pd.DataFrame) -> pd.DataFrame:
    if cue_delay_df is None or len(cue_delay_df) == 0:
        return pd.DataFrame(
            [
                {
                    "n_cues": 0,
                    "n_matched_first_correct": 0,
                    "mean_cue_to_first_correct_sec": np.nan,
                    "median_cue_to_first_correct_sec": np.nan,
                    "n_matched_predicted_transition": 0,
                    "mean_cue_to_predicted_transition_sec": np.nan,
                    "median_cue_to_predicted_transition_sec": np.nan,
                    "n_matched_sustained": 0,
                    "mean_cue_to_sustained_prediction_sec": np.nan,
                    "median_cue_to_sustained_prediction_sec": np.nan,
                }
            ]
        )

    first = cue_delay_df["cue_to_first_correct_prediction_sec"].dropna().to_numpy(dtype=float)
    transition = cue_delay_df["cue_to_predicted_transition_sec"].dropna().to_numpy(dtype=float)
    sustained = cue_delay_df["cue_to_sustained_prediction_sec"].dropna().to_numpy(dtype=float)
    return pd.DataFrame(
        [
            {
                "n_cues": int(len(cue_delay_df)),
                "n_matched_first_correct": int(cue_delay_df["matched_first_correct_prediction"].sum()),
                "mean_cue_to_first_correct_sec": float(np.mean(first)) if len(first) else np.nan,
                "median_cue_to_first_correct_sec": float(np.median(first)) if len(first) else np.nan,
                "n_matched_predicted_transition": int(cue_delay_df["matched_predicted_transition"].sum()),
                "mean_cue_to_predicted_transition_sec": float(np.mean(transition)) if len(transition) else np.nan,
                "median_cue_to_predicted_transition_sec": float(np.median(transition)) if len(transition) else np.nan,
                "n_matched_sustained": int(cue_delay_df["matched_sustained_prediction"].sum()),
                "mean_cue_to_sustained_prediction_sec": float(np.mean(sustained)) if len(sustained) else np.nan,
                "median_cue_to_sustained_prediction_sec": float(np.median(sustained)) if len(sustained) else np.nan,
            }
        ]
    )

def _prediction_signal_on_samples(
    prediction_df: pd.DataFrame,
    n_samples: int,
    fs: int,
    value_column: Optional[str],
    target_label: int,
) -> Tuple[np.ndarray, str]:
    pred_df = prediction_df.sort_values("end_time_sec").reset_index(drop=True)
    signal = np.full(int(n_samples), np.nan, dtype=np.float64)
    if len(pred_df) == 0 or n_samples <= 0:
        return signal, value_column or "pred_label"

    if value_column is not None and value_column in pred_df.columns:
        values = pred_df[value_column].to_numpy(dtype=float)
        source = value_column
    else:
        values = (pred_df["pred_label"].astype(int).to_numpy() == int(target_label)).astype(float)
        source = f"pred_label_equals_{int(target_label)}"

    if "end_sample" in pred_df.columns:
        end_samples = pred_df["end_sample"].to_numpy(dtype=float)
    else:
        end_samples = pred_df["end_time_sec"].to_numpy(dtype=float) * float(fs)
    end_samples = np.asarray(np.round(end_samples), dtype=np.int64)

    for i, sample in enumerate(end_samples):
        start = int(np.clip(sample, 0, n_samples))
        stop = int(np.clip(end_samples[i + 1], 0, n_samples)) if i + 1 < len(end_samples) else int(n_samples)
        if stop > start:
            signal[start:stop] = float(values[i])

    return signal, source

def normalized_xcov_coefficients(
    reference: np.ndarray,
    response: np.ndarray,
    fs: int,
    max_lag_sec: float = 10.0,
    min_overlap_samples: Optional[int] = None,
) -> pd.DataFrame:
    """Python equivalent of normalized xcov with positive lag meaning response lags reference."""

    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    resp = np.asarray(response, dtype=np.float64).reshape(-1)
    n = min(len(ref), len(resp))
    ref = ref[:n]
    resp = resp[:n]
    if n == 0:
        return pd.DataFrame(columns=["lag_samples", "lag_sec", "xcov_coeff", "n_overlap"])

    max_lag = min(int(round(float(max_lag_sec) * int(fs))), max(0, n - 1))
    min_overlap = int(min_overlap_samples) if min_overlap_samples is not None else max(3, int(round(0.5 * int(fs))))
    rows: List[Dict[str, Any]] = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a = ref[: n - lag]
            b = resp[lag:n]
        else:
            a = ref[-lag:n]
            b = resp[: n + lag]

        valid = np.isfinite(a) & np.isfinite(b)
        n_overlap = int(np.sum(valid))
        if n_overlap < min_overlap:
            coeff = np.nan
        else:
            av = a[valid] - np.mean(a[valid])
            bv = b[valid] - np.mean(b[valid])
            denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
            coeff = float(np.dot(av, bv) / denom) if denom > 0 else np.nan
        rows.append(
            {
                "lag_samples": int(lag),
                "lag_sec": float(lag) / float(fs),
                "xcov_coeff": coeff,
                "n_overlap": n_overlap,
            }
        )

    return pd.DataFrame(rows)

def estimate_prediction_xcov_delay(
    prediction_df: pd.DataFrame,
    sample_labels: np.ndarray,
    fs: int,
    class_names: Sequence[str] = (),
    target_label: int = 1,
    max_lag_sec: float = 10.0,
    signal_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate global prediction lag using normalized cross-covariance."""

    labels = np.asarray(sample_labels, dtype=np.int64).reshape(-1)
    if len(labels) == 0 or len(prediction_df) == 0:
        empty_curve = pd.DataFrame(columns=["lag_samples", "lag_sec", "xcov_coeff", "n_overlap"])
        summary = pd.DataFrame(
            [
                {
                    "target_label": int(target_label),
                    "target_label_name": _label_name(target_label, class_names),
                    "prediction_signal_column": signal_column or "",
                    "xcov_delay_sec": np.nan,
                    "xcov_lag_samples": np.nan,
                    "xcov_peak_coeff": np.nan,
                    "max_lag_sec": float(max_lag_sec),
                    "n_valid_samples": 0,
                }
            ]
        )
        return summary, empty_curve

    prob_column = signal_column or _prediction_probability_column(prediction_df, target_label, class_names)
    pred_signal, source = _prediction_signal_on_samples(
        prediction_df,
        n_samples=len(labels),
        fs=fs,
        value_column=prob_column,
        target_label=target_label,
    )
    true_signal = (labels == int(target_label)).astype(np.float64)
    valid_samples = int(np.sum(np.isfinite(pred_signal)))
    curve = normalized_xcov_coefficients(
        true_signal,
        pred_signal,
        fs=fs,
        max_lag_sec=max_lag_sec,
    )

    valid_curve = curve[np.isfinite(curve["xcov_coeff"].to_numpy(dtype=float))]
    if len(valid_curve):
        best_idx = int(valid_curve["xcov_coeff"].astype(float).idxmax())
        best = curve.loc[best_idx]
        delay_sec = float(best["lag_sec"])
        lag_samples = int(best["lag_samples"])
        coeff = float(best["xcov_coeff"])
    else:
        delay_sec = np.nan
        lag_samples = np.nan
        coeff = np.nan

    summary = pd.DataFrame(
        [
            {
                "target_label": int(target_label),
                "target_label_name": _label_name(target_label, class_names),
                "prediction_signal_column": source,
                "xcov_delay_sec": delay_sec,
                "xcov_lag_samples": lag_samples,
                "xcov_peak_coeff": coeff,
                "max_lag_sec": float(max_lag_sec),
                "n_valid_samples": valid_samples,
            }
        ]
    )
    return summary, curve

def summarize_transition_delay(delay_df: pd.DataFrame) -> pd.DataFrame:
    if delay_df is None or len(delay_df) == 0:
        return pd.DataFrame(
            [
                {
                    "n_true_transitions": 0,
                    "n_matched_first_correct": 0,
                    "mean_delay_to_first_correct_sec": np.nan,
                    "median_delay_to_first_correct_sec": np.nan,
                    "n_matched_predicted_transition": 0,
                    "mean_delay_to_predicted_transition_sec": np.nan,
                    "median_delay_to_predicted_transition_sec": np.nan,
                }
            ]
        )

    first = delay_df["delay_to_first_correct_prediction_sec"].dropna().to_numpy(dtype=float)
    transition = delay_df["delay_to_predicted_transition_sec"].dropna().to_numpy(dtype=float)
    return pd.DataFrame(
        [
            {
                "n_true_transitions": int(len(delay_df)),
                "n_matched_first_correct": int(delay_df["matched_first_correct_prediction"].sum()),
                "mean_delay_to_first_correct_sec": float(np.mean(first)) if len(first) else np.nan,
                "median_delay_to_first_correct_sec": float(np.median(first)) if len(first) else np.nan,
                "n_matched_predicted_transition": int(delay_df["matched_predicted_transition"].sum()),
                "mean_delay_to_predicted_transition_sec": float(np.mean(transition)) if len(transition) else np.nan,
                "median_delay_to_predicted_transition_sec": float(np.median(transition)) if len(transition) else np.nan,
            }
        ]
    )


__all__ = [
    "classification_summary",
    "estimate_transition_delay",
    "estimate_cue_prediction_delay",
    "summarize_cue_prediction_delay",
    "normalized_xcov_coefficients",
    "estimate_prediction_xcov_delay",
    "summarize_transition_delay",
]
