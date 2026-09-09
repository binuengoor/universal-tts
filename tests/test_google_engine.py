import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from google.cloud import texttospeech
from gateway.engines.google_engine import GoogleTTSEngine
from gateway.schemas import VoiceObject


@pytest.fixture
def mock_google_engine():
    with patch("google.cloud.texttospeech.TextToSpeechClient") as mock_client_cls:
        engine = GoogleTTSEngine(
            credentials_path=None,
            default_voice="en-US-Neural2-F",
            timeout_seconds=5.0,
        )
        engine.client = MagicMock()
        return engine


@pytest.mark.asyncio
async def test_google_engine_init():
    with patch("google.cloud.texttospeech.TextToSpeechClient.from_service_account_json") as mock_from_json, \
         patch("pathlib.Path.exists", return_value=True):
        engine = GoogleTTSEngine(credentials_path="credentials/fake.json")
        mock_from_json.assert_called_once_with("credentials/fake.json")
        assert engine.engine_name == "google-cloud"
        assert engine.native_format == "mp3"


@pytest.mark.asyncio
async def test_google_engine_init_fallback():
    with patch("google.cloud.texttospeech.TextToSpeechClient") as mock_default, \
         patch("pathlib.Path.exists", return_value=False):
        engine = GoogleTTSEngine(credentials_path="nonexistent.json")
        mock_default.assert_called_once()
        assert engine.engine_name == "google-cloud"


@pytest.mark.asyncio
async def test_google_engine_synthesize_bytes_mp3(mock_google_engine):
    mock_response = MagicMock()
    mock_response.audio_content = b"ID3_google_mock_mp3"
    mock_google_engine.client.synthesize_speech.return_value = mock_response

    audio_bytes, fmt = await mock_google_engine.synthesize_bytes(
        text="Hello Google TTS",
        voice="en-US-Neural2-F",
        speed=1.2,
        response_format="mp3",
    )

    assert audio_bytes == b"ID3_google_mock_mp3"
    assert fmt == "mp3"

    mock_google_engine.client.synthesize_speech.assert_called_once()
    call_kwargs = mock_google_engine.client.synthesize_speech.call_args.kwargs
    assert call_kwargs["input"].text == "Hello Google TTS"
    assert call_kwargs["voice"].name == "en-US-Neural2-F"
    assert call_kwargs["voice"].language_code == "en-US"
    assert call_kwargs["audio_config"].speaking_rate == 1.2
    assert call_kwargs["audio_config"].audio_encoding == texttospeech.AudioEncoding.MP3


@pytest.mark.asyncio
async def test_google_engine_speed_clamping(mock_google_engine):
    mock_response = MagicMock()
    mock_response.audio_content = b"ID3_google_mock_mp3"
    mock_google_engine.client.synthesize_speech.return_value = mock_response

    # Test speed clamped to max 4.0
    await mock_google_engine.synthesize_bytes("Test", speed=6.0)
    call_kwargs = mock_google_engine.client.synthesize_speech.call_args.kwargs
    assert call_kwargs["audio_config"].speaking_rate == 4.0

    # Test speed clamped to min 0.25
    await mock_google_engine.synthesize_bytes("Test", speed=0.1)
    call_kwargs = mock_google_engine.client.synthesize_speech.call_args.kwargs
    assert call_kwargs["audio_config"].speaking_rate == 0.25


@pytest.mark.asyncio
async def test_google_engine_transcode_wav(mock_google_engine):
    mock_response = MagicMock()
    mock_response.audio_content = b"ID3_google_mock_mp3"
    mock_google_engine.client.synthesize_speech.return_value = mock_response

    with patch("gateway.engines.google_engine.convert_audio_ffmpeg", new_callable=AsyncMock) as mock_ffmpeg:
        mock_ffmpeg.return_value = b"RIFF_converted_wav"

        audio_bytes, fmt = await mock_google_engine.synthesize_bytes(
            text="Convert me",
            voice="en-US-Neural2-F",
            response_format="wav",
        )

        assert audio_bytes == b"RIFF_converted_wav"
        assert fmt == "wav"
        mock_ffmpeg.assert_called_once_with(b"ID3_google_mock_mp3", "mp3", "wav")


@pytest.mark.asyncio
async def test_google_engine_synthesize_streaming(mock_google_engine):
    fake_content = b"A" * 10000
    mock_response = MagicMock()
    mock_response.audio_content = fake_content
    mock_google_engine.client.synthesize_speech.return_value = mock_response

    chunks = []
    async for chunk in mock_google_engine.synthesize("Streaming text"):
        chunks.append(chunk)

    assembled = b"".join(chunks)
    assert assembled == fake_content
    assert len(chunks) > 1  # Chunked at 4096 bytes


@pytest.mark.asyncio
async def test_google_engine_get_voices(mock_google_engine):
    mock_voice1 = MagicMock()
    mock_voice1.name = "en-US-Neural2-F"
    mock_voice1.language_codes = ["en-US"]
    mock_voice1.ssml_gender = texttospeech.SsmlVoiceGender.FEMALE
    mock_voice1.natural_sample_rate_hertz = 24000

    mock_voice2 = MagicMock()
    mock_voice2.name = "en-US-Journey-D"
    mock_voice2.language_codes = ["en-US"]
    mock_voice2.ssml_gender = texttospeech.SsmlVoiceGender.MALE
    mock_voice2.natural_sample_rate_hertz = 24000

    mock_resp = MagicMock()
    mock_resp.voices = [mock_voice1, mock_voice2]
    mock_google_engine.client.list_voices.return_value = mock_resp

    voices = await mock_google_engine.get_voices()
    assert len(voices) == 2
    assert voices[0].id == "en-US-Neural2-F"
    assert voices[0].gender == "Female"
    assert voices[0].engine == "google-cloud"
    assert voices[1].id == "en-US-Journey-D"
    assert voices[1].gender == "Male"

    # Second call should return cached voices without calling list_voices again
    mock_google_engine.client.list_voices.reset_mock()
    cached = await mock_google_engine.get_voices()
    assert cached == voices
    mock_google_engine.client.list_voices.assert_not_called()


@pytest.mark.asyncio
async def test_google_engine_fallback_voices_on_error(mock_google_engine):
    mock_google_engine.client.list_voices.side_effect = RuntimeError("API unavailable")

    voices = await mock_google_engine.get_voices()
    assert len(voices) >= 5
    voice_ids = [v.id for v in voices]
    assert "en-US-Neural2-F" in voice_ids
    assert "en-US-Journey-D" in voice_ids


@pytest.mark.asyncio
async def test_google_engine_health_check(mock_google_engine):
    assert await mock_google_engine.health_check() is True

    mock_google_engine.client = None
    assert await mock_google_engine.health_check() is False
