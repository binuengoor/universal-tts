import asyncio
import io
import logging
from typing import AsyncGenerator, List, Optional, Tuple
import edge_tts

from gateway.audio import convert_audio_ffmpeg
from gateway.engines.base import BaseTTSEngine
from gateway.schemas import VoiceObject

logger = logging.getLogger(__name__)


class EdgeTTSEngine(BaseTTSEngine):
    engine_name = "edge-tts"
    native_format = "mp3"

    def __init__(self, default_voice: str = "en-US-AriaNeural", timeout_seconds: float = 5.0):
        self.default_voice = default_voice
        self.timeout_seconds = timeout_seconds
        self._cached_voices: Optional[List[VoiceObject]] = None

    def _format_rate(self, speed: float) -> str:
        pct = int(round((speed - 1.0) * 100))
        return f"{pct:+d}%"

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        response_format: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        target_voice = voice or self.default_voice
        target_fmt = (response_format or self.native_format).lower()
        rate_str = self._format_rate(speed)

        # If zero-transcode (target format is native mp3), stream directly chunk-by-chunk
        if target_fmt == self.native_format:
            communicate = edge_tts.Communicate(text=text, voice=target_voice, rate=rate_str)
            try:
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        yield chunk["data"]
            except Exception as e:
                logger.error("Edge-TTS streaming failed: %s", str(e))
                raise
        else:
            # Need conversion: accumulate native MP3 and convert
            full_mp3, _ = await self.synthesize_bytes(text, target_voice, speed, "mp3")
            converted = await convert_audio_ffmpeg(full_mp3, "mp3", target_fmt)
            yield converted

    async def synthesize_bytes(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        response_format: Optional[str] = None,
    ) -> Tuple[bytes, str]:
        target_voice = voice or self.default_voice
        target_fmt = (response_format or self.native_format).lower()
        rate_str = self._format_rate(speed)

        communicate = edge_tts.Communicate(text=text, voice=target_voice, rate=rate_str)
        buffer = io.BytesIO()

        try:
            async with asyncio.timeout(self.timeout_seconds):
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        buffer.write(chunk["data"])
        except asyncio.TimeoutError:
            logger.error("Edge-TTS timed out after %s seconds", self.timeout_seconds)
            raise TimeoutError(f"Edge-TTS timed out after {self.timeout_seconds}s")
        except Exception as e:
            logger.error("Edge-TTS synthesis error: %s", str(e))
            raise

        mp3_bytes = buffer.getvalue()
        if not mp3_bytes:
            raise RuntimeError("Edge-TTS returned empty audio data")

        if target_fmt == self.native_format:
            return mp3_bytes, "mp3"

        converted = await convert_audio_ffmpeg(mp3_bytes, "mp3", target_fmt)
        return converted, target_fmt

    async def get_voices(self) -> List[VoiceObject]:
        if self._cached_voices is not None:
            return self._cached_voices

        try:
            voices_data = await edge_tts.list_voices()
            voices: List[VoiceObject] = []
            for v in voices_data:
                voices.append(
                    VoiceObject(
                        id=v.get("ShortName", ""),
                        name=v.get("FriendlyName", v.get("ShortName", "")),
                        engine=self.engine_name,
                        language=v.get("Locale", "en-US"),
                        gender=v.get("Gender", None),
                    )
                )
            self._cached_voices = voices
            return voices
        except Exception as e:
            logger.warning("Failed to fetch Edge-TTS online voices: %s", str(e))
            # Fallback default list
            return [
                VoiceObject(
                    id="en-US-AriaNeural",
                    name="Microsoft Aria (Neural)",
                    engine=self.engine_name,
                    language="en-US",
                    gender="Female",
                ),
                VoiceObject(
                    id="en-US-GuyNeural",
                    name="Microsoft Guy (Neural)",
                    engine=self.engine_name,
                    language="en-US",
                    gender="Male",
                ),
                VoiceObject(
                    id="en-US-JennyNeural",
                    name="Microsoft Jenny (Neural)",
                    engine=self.engine_name,
                    language="en-US",
                    gender="Female",
                ),
            ]

    async def health_check(self) -> bool:
        try:
            # Quick list_voices or small check
            if self._cached_voices:
                return True
            voices = await self.get_voices()
            return len(voices) > 0
        except Exception:
            return False
