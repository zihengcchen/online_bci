"""BIOPAC MP150 acquisition and raw recording I/O."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    from .config import AcquisitionConfig, PathLike
    from .utils import (
        _as_2d_samples_channels,
        _checked_acquisition_chunk,
        _duration_to_sample_count,
        _json_safe,
        _now_string,
        ensure_dir,
    )
except ImportError:
    from config import AcquisitionConfig, PathLike
    from utils import (
        _as_2d_samples_channels,
        _checked_acquisition_chunk,
        _duration_to_sample_count,
        _json_safe,
        _now_string,
        ensure_dir,
    )


def _print_recording_elapsed(
    start_wall: float,
    duration_sec: float,
    stop_event: threading.Event,
    interval_sec: float = 1.0,
) -> None:
    while not stop_event.wait(float(interval_sec)):
        elapsed = max(0.0, time.time() - float(start_wall))
        shown_elapsed = min(elapsed, float(duration_sec))
        sys.stdout.write(
            f"\rElapsed since recording start: {shown_elapsed:6.1f}s / {float(duration_sec):.1f}s"
        )
        sys.stdout.flush()

def _import_mp150_class():
    try:
        from .mpy150_chunk import MP150  # type: ignore
    except Exception:
        from mpy150_chunk import MP150  # type: ignore
    return MP150

def collect_mp150_recording(
    output_npz: PathLike,
    duration_sec: float,
    config: AcquisitionConfig,
    segment_name: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Collect one MP150 recording and save it as a raw NPZ.

    Saved key ``data`` has shape ``(samples, channels)`` and uses the hardware
    channel order in ``config.channels``. Fixed-duration recordings are one
    hardware chunk whose length is ``duration_sec``.
    """

    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    sample_rate = int(config.samplerate)
    channels = tuple(int(ch) for ch in config.channels)
    if not channels:
        raise ValueError("config.channels must contain at least one channel.")
    target_samples = _duration_to_sample_count(duration_sec, sample_rate)

    MP150 = _import_mp150_class()
    mp = MP150(samplerate=sample_rate, channels=list(channels))

    chunk_start_wall: List[float] = []
    chunk_end_wall: List[float] = []

    start_wall = time.time()
    segment_label = str(segment_name).strip() or output_npz.stem
    print(
        f"Recording started: {segment_label} at {_now_string()} "
        f"({float(duration_sec):.1f}s, channels={channels})",
        flush=True,
    )
    stop_progress = threading.Event()
    progress_thread = threading.Thread(
        target=_print_recording_elapsed,
        args=(start_wall, float(duration_sec), stop_progress),
        daemon=True,
    )
    progress_thread.start()
    try:
        chunk_start_wall.append(time.time() - start_wall)
        data = _checked_acquisition_chunk(mp.get_chunk(float(duration_sec)), len(channels))
        chunk_end_wall.append(time.time() - start_wall)
    finally:
        stop_progress.set()
        progress_thread.join(timeout=2.0)
        elapsed = max(0.0, time.time() - start_wall)
        sys.stdout.write(
            f"\rElapsed since recording start: {min(elapsed, float(duration_sec)):6.1f}s / "
            f"{float(duration_sec):.1f}s\n"
        )
        sys.stdout.flush()
        mp.close()
        print(
            f"Recording ended: {segment_label} at {_now_string()} "
            f"(elapsed {elapsed:.1f}s)",
            flush=True,
        )

    if data.shape[0] < target_samples:
        raise RuntimeError(
            f"Expected one {duration_sec:.3f}s chunk with {target_samples} samples, "
            f"got {data.shape[0]} samples."
        )
    if data.shape[0] > target_samples:
        data = data[:target_samples]
    data = data.astype(np.float32)

    time_sec = np.arange(data.shape[0], dtype=np.float64) / float(sample_rate)
    np.savez(
        output_npz,
        data=data,
        samplerate=np.array(sample_rate, dtype=np.int64),
        channels=np.asarray(channels, dtype=np.int64),
        time_sec=time_sec,
        segment_name=np.array(str(segment_name)),
        chunk_start_wall_sec=np.asarray(chunk_start_wall, dtype=np.float64),
        chunk_end_wall_sec=np.asarray(chunk_end_wall, dtype=np.float64),
        requested_duration_sec=np.array(float(duration_sec), dtype=np.float64),
        created_at=np.array(_now_string()),
        metadata_json=np.array(json.dumps(_json_safe(metadata or {}))),
    )
    return output_npz

def collect_training_segments(
    output_dir: PathLike,
    segment_names: Sequence[str],
    config: AcquisitionConfig,
    duration_sec: float = 300.0,
) -> List[Path]:
    """Collect several fixed-duration training recordings."""

    output_dir = ensure_dir(output_dir)
    paths = []
    for name in segment_names:
        safe_name = str(name).strip().replace(" ", "_") or f"segment_{len(paths) + 1:02d}"
        out = output_dir / f"{safe_name}.npz"
        print(
            f"Collecting training segment {len(paths) + 1}/{len(segment_names)}: {safe_name}",
            flush=True,
        )
        paths.append(
            collect_mp150_recording(
                output_npz=out,
                duration_sec=duration_sec,
                config=config,
                segment_name=safe_name,
            )
        )
    return paths

def load_raw_recording(path: PathLike) -> Dict[str, Any]:
    raw = np.load(Path(path), allow_pickle=True)
    if "data" not in raw.files:
        raise KeyError(f"Raw recording {path} is missing key 'data'.")
    if "samplerate" not in raw.files:
        raise KeyError(f"Raw recording {path} is missing key 'samplerate'.")
    if "channels" not in raw.files:
        raise KeyError(f"Raw recording {path} is missing key 'channels'.")
    return {
        "data": _as_2d_samples_channels(raw["data"]),
        "samplerate": int(np.asarray(raw["samplerate"]).item()),
        "channels": tuple(int(x) for x in np.asarray(raw["channels"]).reshape(-1)),
        "path": str(path),
        "segment_name": str(np.asarray(raw["segment_name"]).item()) if "segment_name" in raw.files else Path(path).stem,
    }


__all__ = [
    "AcquisitionConfig",
    "collect_mp150_recording",
    "collect_training_segments",
    "load_raw_recording",
]
