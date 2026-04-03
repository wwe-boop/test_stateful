"""Scheduler module — functionality merged into engine_loop.py and kv_cache_pool.py.

The scheduling logic (batch formation, slot management, prefill queue) is now
directly integrated into EngineLoop for tighter coupling with the GPU pipeline.

KV slot allocation is handled by KVCachePool.
"""
