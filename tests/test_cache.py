import time
from gateway.cache import AudioCache


def test_cache_put_get():
    cache = AudioCache(enabled=True, max_entries=10, max_memory_mb=1, ttl_seconds=60)
    key = cache.generate_key("edge-tts", "en-US-AriaNeural", "Hello world", 1.0, "mp3")
    
    assert cache.get(key) is None
    cache.put(key, b"fake_mp3_data", "mp3")
    
    cached = cache.get(key)
    assert cached is not None
    data, fmt = cached
    assert data == b"fake_mp3_data"
    assert fmt == "mp3"


def test_cache_lru_eviction():
    cache = AudioCache(enabled=True, max_entries=2, max_memory_mb=1, ttl_seconds=60)
    
    k1 = cache.generate_key("edge-tts", "voice1", "Text 1", 1.0, "mp3")
    k2 = cache.generate_key("edge-tts", "voice2", "Text 2", 1.0, "mp3")
    k3 = cache.generate_key("edge-tts", "voice3", "Text 3", 1.0, "mp3")
    
    cache.put(k1, b"data1", "mp3")
    cache.put(k2, b"data2", "mp3")
    
    # Access k1 to make k2 least recently used
    _ = cache.get(k1)
    
    # Put k3, which should evict k2
    cache.put(k3, b"data3", "mp3")
    
    assert cache.get(k1) is not None
    assert cache.get(k2) is None
    assert cache.get(k3) is not None


def test_cache_ttl_expiration():
    cache = AudioCache(enabled=True, max_entries=10, max_memory_mb=1, ttl_seconds=1)
    key = cache.generate_key("edge-tts", "voice1", "Hello", 1.0, "mp3")
    
    cache.put(key, b"data", "mp3")
    assert cache.get(key) is not None
    
    time.sleep(1.1)
    assert cache.get(key) is None


def test_cache_stats():
    cache = AudioCache(enabled=True, max_entries=10, max_memory_mb=1, ttl_seconds=60)
    key = cache.generate_key("edge-tts", "voice1", "Hello", 1.0, "mp3")
    
    cache.put(key, b"12345", "mp3")
    _ = cache.get(key)  # hit
    _ = cache.get("non_existent")  # miss
    
    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["entries"] == 1
    assert stats["memory_bytes"] == 5
    assert stats["hit_ratio"] == 0.5
