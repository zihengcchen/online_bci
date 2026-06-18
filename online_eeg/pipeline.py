"""Convenience imports for the full real-subject pipeline."""

try:
    from .real_subject_pipeline import *  # noqa: F401,F403
except ImportError:
    from real_subject_pipeline import *  # noqa: F401,F403
