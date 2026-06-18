"""Short-trial testing entry points."""

try:
    from .real_subject_pipeline import (
        collect_preprocess_and_test_trial,
        evaluate_prediction_log_against_labeled_recording,
        estimate_transition_delay,
        posthoc_analyze_realtime_trial,
        predict_labeled_recording,
        run_realtime_mp150_prediction,
        summarize_transition_delay,
    )
except ImportError:
    from real_subject_pipeline import (
        collect_preprocess_and_test_trial,
        evaluate_prediction_log_against_labeled_recording,
        estimate_transition_delay,
        posthoc_analyze_realtime_trial,
        predict_labeled_recording,
        run_realtime_mp150_prediction,
        summarize_transition_delay,
    )

__all__ = [
    "collect_preprocess_and_test_trial",
    "evaluate_prediction_log_against_labeled_recording",
    "estimate_transition_delay",
    "posthoc_analyze_realtime_trial",
    "predict_labeled_recording",
    "run_realtime_mp150_prediction",
    "summarize_transition_delay",
]
