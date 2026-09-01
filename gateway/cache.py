from collections import OrderedDict
import hashlib
import time
from typing import Any, Dict, Optional, Tuple


class AudioCacheEntry:
    __slots__ = ("data", "format_name", "created_at", "size_bytes")

    def __init__(self, data: bytes, format_name: str):
        self.data = data
        self.format_name = format_name
        self.created_at = time.time()
        self.size_bytes = len(data)


class AudioCache:
    """
    In-memory LRU Audio Cache with SHA-256 key hashing, TTL expiration, and byte size limits.
    """

    def __init__(
        self,
        enabled: bool = True,
        max_entries: int = 500,
        max_memory_mb: int = 50,
        ttl_seconds: int = 86400,
    ):
        self.enabled = enabled
        self.max_entries = max_entries
        self.max_bytes = max_memory_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds

        self._cache: OrderedDict[str, AudioCacheEntry] = OrderedDict()
        self._current_bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @staticmethod
    def generate_key(
        engine: str,
        voice: str,
        text: str,
        speed: float,
        response_format: str,
    ) -> str:
        raw = f"{engine}:{voice}:{text}:{speed:.2f}:{response_format.lower()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Tuple[bytes, str]]:
        if not self.enabled:
            self._misses += 1
            return None

        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None

        # Check TTL
        if (time.time() - entry.created_at) > self.ttl_seconds:
            self._evict_key(key)
            self._misses += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        self._hits += 1
        return entry.data, entry.format_name

    def put(self, key: str, data: bytes, format_name: str) -> None:
        if not self.enabled or len(data) > self.max_bytes:
            return

        if key in self._cache:
            self._evict_key(key)

        # Enforce capacity
        while (
            (len(self._cache) >= self.max_entries)
            or (self._current_bytes + len(data) > self.max_bytes)
        ) and self._cache:
            oldest_key = next(iter(self._cache))
            self._evict_key(oldest_key)
            self._evictions += 1

        entry = AudioCacheEntry(data=data, format_name=format_name)
        self._cache[key] = entry
        self._current_bytes += entry.size_bytes

    def _evict_key(self, key: str) -> None:
        if key in self._cache:
            entry = self._cache.pop(key)
            self._current_bytes -= entry.size_bytes

    def clear(self) -> None:
        self._cache.clear()
        self._current_bytes = 0

    def get_stats(self) -> Dict[str, Any]:
        total_requests = self._hits + self._misses
        hit_ratio = (self._hits / total_requests) if total_requests > 0 else 0.0
        return {
            "enabled": self.enabled,
            "entries": len(self._cache),
            "max_entries": self.max_entries,
            "memory_bytes": self._current_bytes,
            "memory_mb": round(self._current_bytes / (1024 * 1024), 2),
            "max_memory_mb": round(self.max_bytes / (1024 * 1024), 2),
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_ratio": round(hit_ratio, 4),
        }
