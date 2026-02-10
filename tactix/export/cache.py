"""
Project: Tactix
File Created: 2026-02-10
Author: Xingnan Zhu
File Name: cache.py
Description:
    Pickle-based cache for expensive per-run computations (tracking results,
    homography sequences, etc.).

    The cache key is a short SHA-256 hash derived from the video filename and
    its total frame count — cheap to compute and stable across runs on the
    same file.

    Usage:
        cache = TrackingCache(cache_dir=cfg.CACHE_DIR)
        key = cache.make_key(cfg.INPUT_VIDEO, video_info.total_frames)
        data = cache.load(key)
        if data is None:
            data = run_expensive_operation()
            cache.save(key, data)
"""

import hashlib
import os
import pickle
from pathlib import Path
from typing import Any, Optional


class TrackingCache:
    def __init__(self, cache_dir: str = "assets/cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def make_key(self, video_path: str, total_frames: int) -> str:
        """Returns a 16-char hex key derived from the video filename and frame count."""
        raw = f"{os.path.basename(video_path)}:{total_frames}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pkl"

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def load(self, key: str) -> Optional[Any]:
        """Returns the cached object, or None on a miss or corrupt entry."""
        path = self._path(key)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def save(self, key: str, data: Any) -> None:
        with open(self._path(key), "wb") as f:
            pickle.dump(data, f)

    def invalidate(self, key: str) -> None:
        path = self._path(key)
        if path.exists():
            path.unlink()
