"""Short-trial testing entry points."""

try:
    from .real_subject_pipeline import (
        collect_preprocess_and_test_trial,
        estimate_transition_delay,
        predict_labeled_recording,
        summarize_transition_delay,
    )
except ImportError:
    from real_subject_pipeline import (
        collect_preprocess_and_test_trial,
        estimate_transition_delay,
        predict_labeled_recording,
        summarize_transition_delay,
    )

__all__ = [
    "collect_preprocess_and_test_trial",
    "estimate_transition_delay",
    "predict_labeled_recording",
    "summarize_transition_delay",
]
