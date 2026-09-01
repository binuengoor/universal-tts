import asyncio
import logging
from typing import AsyncGenerator, Dict, List, Optional, Tuple

from gateway.config import AppSettings
from gateway.engines.base import BaseTTSEngine
from gateway.engines.edge_engine import EdgeTTSEngine
from gateway.engines.kokoro_engine import KokoroEngine
from gateway.engines.piper_engine import PiperEngine
from gateway.schemas import VoiceObject

logger = logging.getLogger(__name__)


class TTSRouter:
    """
    Dynamic Router and Circuit Breaker for TTS synthesis.
    """

    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.engines: Dict[str, BaseTTSEngine] = {}
        self._init_engines()

    def _init_engines(self) -> None:
        if self.settings.engines.edge_tts.enabled:
            self.engines["edge-tts"] = EdgeTTSEngine(
                default_voice=self.settings.engines.edge_tts.default_voice,
                timeout_seconds=self.settings.engines.edge_tts.timeout_seconds or 5.0,
            )

        if self.settings.engines.piper.enabled:
            self.engines["piper"] = PiperEngine(
                models_dir=self.settings.paths.piper_models_dir,
                default_voice=self.settings.engines.piper.default_voice,
            )

        if self.settings.engines.kokoro.enabled:
            self.engines["kokoro"] = KokoroEngine(
                model_path=self.settings.paths.kokoro_model_path,
                voices_path=self.settings.paths.kokoro_voices_path,
                default_voice=self.settings.engines.kokoro.default_voice,
            )

    def resolve_engine_and_voice(
        self,
        model: Optional[str],
        voice: Optional[str],
    ) -> Tuple[BaseTTSEngine, str]:
        req_model = (model or "").lower().strip()
        req_voice = (voice or "").strip()

        # Step 1: Map OpenAI voice names (alloy, echo, etc.)
        if req_voice.lower() in self.settings.openai_voice_map:
            req_voice = self.settings.openai_voice_map[req_voice.lower()]

        # Step 2: Map short aliases (ryan, heart, aria, etc.)
        if req_voice.lower() in self.settings.short_aliases:
            req_voice = self.settings.short_aliases[req_voice.lower()]

        # Step 3: Determine engine based on explicit model or voice heuristics
        target_engine_name = None

        if req_model in ("edge-tts", "edge_tts"):
            target_engine_name = "edge-tts"
        elif req_model in ("piper", "piper-tts"):
            target_engine_name = "piper"
        elif req_model in ("kokoro", "kokoro-tts"):
            target_engine_name = "kokoro"
        elif req_model in ("tts-1", "tts-1-hd"):
            # OpenAI standard model names map to default engine (edge-tts)
            target_engine_name = self.settings.defaults.engine
        else:
            # Infer from voice prefix/pattern
            if any(req_voice.startswith(p) for p in ("af_", "am_", "bf_", "bm_", "jf_", "jm_", "zf_", "zm_", "ef_", "ff_", "if_", "pf_", "hf_")):
                target_engine_name = "kokoro"
            elif "Neural" in req_voice or req_voice.count("-") >= 2 and not req_voice.endswith(("-low", "-medium", "-high")):
                target_engine_name = "edge-tts"
            elif req_voice.endswith(("-low", "-medium", "-high")) or "piper" in req_voice.lower():
                target_engine_name = "piper"
            else:
                target_engine_name = self.settings.defaults.engine

        # Fallback to available engine if target engine is not registered
        if target_engine_name not in self.engines:
            target_engine_name = next(iter(self.engines)) if self.engines else "edge-tts"

        engine = self.engines.get(target_engine_name)
        if not engine:
            raise RuntimeError(f"Engine '{target_engine_name}' is not configured or available")

        # Resolve voice default if empty
        resolved_voice = req_voice if req_voice else engine.default_voice
        return engine, resolved_voice

    def resolve_format(self, engine: BaseTTSEngine, requested_format: Optional[str]) -> str:
        if requested_format:
            return requested_format.lower()
        return engine.native_format

    async def synthesize(
        self,
        text: str,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        speed: float = 1.0,
        response_format: Optional[str] = None,
    ) -> Tuple[bytes, str, str]:
        """
        Synthesize audio with automatic circuit breaker fallback.
        Returns: (audio_bytes, format_name, resolved_engine_name)
        """
        primary_engine, resolved_voice = self.resolve_engine_and_voice(model, voice)
        target_fmt = self.resolve_format(primary_engine, response_format)

        try:
            audio_bytes, out_fmt = await primary_engine.synthesize_bytes(
                text=text,
                voice=resolved_voice,
                speed=speed,
                response_format=target_fmt,
            )
            return audio_bytes, out_fmt, primary_engine.engine_name

        except Exception as e:
            logger.warning(
                "Primary engine '%s' failed: %s",
                primary_engine.engine_name,
                str(e),
            )

            # Circuit breaker fallback
            if self.settings.circuit_breaker.enabled and primary_engine.engine_name == "edge-tts":
                fallback_name = self.settings.circuit_breaker.fallback_engine
                fallback_engine = self.engines.get(fallback_name)
                if fallback_engine:
                    fallback_voice = self.settings.circuit_breaker.fallback_voice
                    logger.info(
                        "⚡ Circuit breaker triggered: Falling back from Edge-TTS to %s (voice: %s)",
                        fallback_name,
                        fallback_voice,
                    )
                    try:
                        audio_bytes, out_fmt = await fallback_engine.synthesize_bytes(
                            text=text,
                            voice=fallback_voice,
                            speed=speed,
                            response_format=target_fmt,
                        )
                        return audio_bytes, out_fmt, fallback_name
                    except Exception as fb_err:
                        logger.error("Fallback engine '%s' also failed: %s", fallback_name, fb_err)

            raise e

    async def get_all_voices(self) -> List[VoiceObject]:
        all_voices: List[VoiceObject] = []
        for engine in self.engines.values():
            try:
                voices = await engine.get_voices()
                all_voices.extend(voices)
            except Exception as e:
                logger.warning("Error fetching voices from engine %s: %s", engine.engine_name, e)
        return all_voices

    async def get_engine_health(self) -> Dict[str, bool]:
        status = {}
        for name, engine in self.engines.items():
            try:
                status[name] = await engine.health_check()
            except Exception:
                status[name] = False
        return status
