import asyncio
import logging
import os
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Tuple

from google.cloud import texttospeech

from gateway.audio import convert_audio_ffmpeg
from gateway.engines.base import BaseTTSEngine
from gateway.schemas import VoiceObject

logger = logging.getLogger(__name__)


class GoogleTTSEngine(BaseTTSEngine):
    engine_name = "google-cloud"
    native_format = "mp3"

    def __init__(
        self,
        credentials_path: Optional[str] = "credentials/google-service-account.json",
        default_voice: str = "en-US-Neural2-F",
        timeout_seconds: Optional[float] = 8.0,
    ):
        self.credentials_path = credentials_path
        self.default_voice = default_voice
        self.timeout_seconds = timeout_seconds
        self._cached_voices: Optional[List[VoiceObject]] = None
        self._voice_lang_map: Dict[str, str] = {}
        self.client: Optional[texttospeech.TextToSpeechClient] = None

        self._init_client()

    def _init_client(self) -> None:
        try:
            if self.credentials_path and Path(self.credentials_path).exists():
                self.client = texttospeech.TextToSpeechClient.from_service_account_json(
                    self.credentials_path
                )
                logger.info("Google Cloud TTS client initialized from %s", self.credentials_path)
            elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and Path(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]).exists():
                self.client = texttospeech.TextToSpeechClient()
                logger.info("Google Cloud TTS client initialized via GOOGLE_APPLICATION_CREDENTIALS")
            else:
                self.client = texttospeech.TextToSpeechClient()
                logger.info("Google Cloud TTS client initialized with default credentials")
        except Exception as e:
            logger.warning("Failed to initialize Google Cloud TTS client: %s", str(e))
            self.client = None

    def _extract_language_code(self, voice_name: str) -> str:
        if voice_name in self._voice_lang_map:
            return self._voice_lang_map[voice_name]
        parts = voice_name.split("-")
        if len(parts) >= 2:
            return f"{parts[0]}-{parts[1]}"
        return "en-US"

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        response_format: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        audio_bytes, _ = await self.synthesize_bytes(
            text=text,
            voice=voice,
            speed=speed,
            response_format=response_format,
        )
        chunk_size = 4096
        for i in range(0, len(audio_bytes), chunk_size):
            yield audio_bytes[i : i + chunk_size]

    async def synthesize_bytes(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        response_format: Optional[str] = None,
    ) -> Tuple[bytes, str]:
        if self.client is None:
            raise RuntimeError("Google Cloud TTS client is not initialized or credentials missing")

        target_voice = voice or self.default_voice
        target_fmt = (response_format or self.native_format).lower()
        language_code = self._extract_language_code(target_voice)
        clamped_speed = max(0.25, min(4.0, float(speed)))

        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice_params = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=target_voice,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=clamped_speed,
        )

        def _call_synthesize() -> bytes:
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice_params,
                audio_config=audio_config,
                timeout=self.timeout_seconds,
            )
            return response.audio_content

        try:
            if self.timeout_seconds:
                async with asyncio.timeout(self.timeout_seconds):
                    audio_bytes = await asyncio.to_thread(_call_synthesize)
            else:
                audio_bytes = await asyncio.to_thread(_call_synthesize)
        except asyncio.TimeoutError:
            logger.error("Google Cloud TTS timed out after %s seconds", self.timeout_seconds)
            raise TimeoutError(f"Google Cloud TTS timed out after {self.timeout_seconds}s")
        except Exception as e:
            logger.error("Google Cloud TTS synthesis error: %s", str(e))
            raise

        if not audio_bytes:
            raise RuntimeError("Google Cloud TTS returned empty audio data")

        # Zero-transcode fast path
        if target_fmt == self.native_format:
            return audio_bytes, "mp3"

        converted = await convert_audio_ffmpeg(audio_bytes, "mp3", target_fmt)
        return converted, target_fmt

    def _get_fallback_voices(self) -> List[VoiceObject]:
        fallback = [
            VoiceObject(
                id="en-US-Neural2-F",
                name="Google en-US (Neural2 Female)",
                engine=self.engine_name,
                language="en-US",
                gender="Female",
            ),
            VoiceObject(
                id="en-US-Neural2-D",
                name="Google en-US (Neural2 Male)",
                engine=self.engine_name,
                language="en-US",
                gender="Male",
            ),
            VoiceObject(
                id="en-US-Journey-F",
                name="Google en-US (Journey Female)",
                engine=self.engine_name,
                language="en-US",
                gender="Female",
            ),
            VoiceObject(
                id="en-US-Journey-D",
                name="Google en-US (Journey Male)",
                engine=self.engine_name,
                language="en-US",
                gender="Male",
            ),
            VoiceObject(
                id="en-US-Studio-O",
                name="Google en-US (Studio Female)",
                engine=self.engine_name,
                language="en-US",
                gender="Female",
            ),
            VoiceObject(
                id="en-US-Wavenet-D",
                name="Google en-US (WaveNet Male)",
                engine=self.engine_name,
                language="en-US",
                gender="Male",
            ),
        ]
        for v in fallback:
            self._voice_lang_map[v.id] = v.language
        return fallback

    async def get_voices(self) -> List[VoiceObject]:
        if self._cached_voices is not None:
            return self._cached_voices

        if self.client is None:
            fallback = self._get_fallback_voices()
            self._cached_voices = fallback
            return fallback

        try:
            response = await asyncio.to_thread(self.client.list_voices)
            voices: List[VoiceObject] = []
            for v in response.voices:
                lang = v.language_codes[0] if v.language_codes else "en-US"
                try:
                    gender_enum = texttospeech.SsmlVoiceGender(v.ssml_gender)
                    gender_str = gender_enum.name.capitalize()
                    if gender_str == "Ssml_voice_gender_unspecified":
                        gender = None
                    else:
                        gender = gender_str
                except Exception:
                    gender = None

                sample_rate = getattr(v, "natural_sample_rate_hertz", None)
                voices.append(
                    VoiceObject(
                        id=v.name,
                        name=v.name,
                        engine=self.engine_name,
                        language=lang,
                        gender=gender,
                        sample_rate=sample_rate,
                    )
                )
                self._voice_lang_map[v.name] = lang

            self._cached_voices = voices
            return voices
        except Exception as e:
            logger.warning("Failed to fetch Google Cloud voices online: %s", str(e))
            fallback = self._get_fallback_voices()
            self._cached_voices = fallback
            return fallback

    async def health_check(self) -> bool:
        return self.client is not None
