import pytest
from unittest.mock import AsyncMock, patch
from gateway.config import AppSettings
from gateway.router import TTSRouter


@pytest.mark.asyncio
async def test_circuit_breaker_fallback_on_edge_failure():
    settings = AppSettings()
    settings.circuit_breaker.enabled = True
    settings.circuit_breaker.fallback_engine = "piper"
    settings.circuit_breaker.fallback_voice = "en_US-ryan-medium"

    router = TTSRouter(settings=settings)

    # Mock edge engine failure
    router.engines["edge-tts"].synthesize_bytes = AsyncMock(side_effect=TimeoutError("Edge-TTS timed out"))

    # Mock piper engine success
    router.engines["piper"].synthesize_bytes = AsyncMock(return_value=(b"RIFF_mock_wav", "wav"))

    audio_bytes, fmt, engine_name = await router.synthesize(
        text="Testing circuit breaker fallback",
        model="edge-tts",
        voice="en-US-AriaNeural",
    )

    assert engine_name == "piper"
    assert audio_bytes == b"RIFF_mock_wav"
    assert fmt == "wav"
    router.engines["edge-tts"].synthesize_bytes.assert_called_once()
    router.engines["piper"].synthesize_bytes.assert_called_once()


@pytest.mark.asyncio
async def test_voice_and_engine_resolution():
    settings = AppSettings()
    router = TTSRouter(settings=settings)

    # 1. Standard OpenAI voice alias
    engine, voice = router.resolve_engine_and_voice(model="tts-1", voice="alloy")
    assert engine.engine_name == "edge-tts"
    assert voice == "en-US-AriaNeural"

    # 2. Kokoro voice prefix
    engine, voice = router.resolve_engine_and_voice(model=None, voice="af_heart")
    assert engine.engine_name == "kokoro"
    assert voice == "af_heart"

    # 3. Piper short alias
    engine, voice = router.resolve_engine_and_voice(model=None, voice="ryan")
    assert engine.engine_name == "piper"
    assert voice == "en_US-ryan-medium"

    # 4. Explicit model override
    engine, voice = router.resolve_engine_and_voice(model="piper", voice="custom-voice")
    assert engine.engine_name == "piper"
    assert voice == "custom-voice"
