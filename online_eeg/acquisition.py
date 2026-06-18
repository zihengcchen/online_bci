"""MP150 acquisition entry points."""

try:
    from .real_subject_pipeline import (
        AcquisitionConfig,
        collect_mp150_recording,
        collect_training_segments,
        load_raw_recording,
    )
except ImportError:
    from real_subject_pipeline import (
        AcquisitionConfig,
        collect_mp150_recording,
        collect_training_segments,
        load_raw_recording,
    )

__all__ = [
    "AcquisitionConfig",
    "collect_mp150_recording",
    "collect_training_segments",
    "load_raw_recording",
]
