"""Compatibility imports for the real-subject online EEG pipeline.

The implementation is split by functionality across focused top-level modules.
Existing notebooks can still import from ``real_subject_pipeline`` if needed.
"""

from __future__ import annotations

try:
    from .config import *  # noqa: F401,F403
    from .utils import *  # noqa: F401,F403
    from .acquisition import *  # noqa: F401,F403
    from .preprocessing import *  # noqa: F401,F403
    from .windowing import *  # noqa: F401,F403
    from .modeling import *  # noqa: F401,F403
    from .metrics import *  # noqa: F401,F403
    from .training import *  # noqa: F401,F403
    from .realtime import *  # noqa: F401,F403
    from .testing import *  # noqa: F401,F403
    from .plots import *  # noqa: F401,F403
except ImportError:
    from config import *  # noqa: F401,F403
    from utils import *  # noqa: F401,F403
    from acquisition import *  # noqa: F401,F403
    from preprocessing import *  # noqa: F401,F403
    from windowing import *  # noqa: F401,F403
    from modeling import *  # noqa: F401,F403
    from metrics import *  # noqa: F401,F403
    from training import *  # noqa: F401,F403
    from realtime import *  # noqa: F401,F403
    from testing import *  # noqa: F401,F403
    from plots import *  # noqa: F401,F403
