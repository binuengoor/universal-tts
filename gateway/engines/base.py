from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Optional, Tuple
from gateway.schemas import VoiceObject


class BaseTTSEngine(ABC):
    """
    Abstract base class for all TTS engine drivers.
    """

    engine_name: str
    native_format: str
    default_voice: str

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        response_format: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        Synthesize text and stream chunks of audio bytes.
        """
        pass

    @abstractmethod
    async def synthesize_bytes(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        response_format: Optional[str] = None,
    ) -> Tuple[bytes, str]:
        """
        Synthesize text and return (full_audio_bytes, format_name).
        """
        pass

    @abstractmethod
    async def get_voices(self) -> List[VoiceObject]:
        """
        Return the list of voices provided by this engine.
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Return True if the engine is ready and operational.
        """
        pass
