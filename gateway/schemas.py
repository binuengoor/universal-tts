import time
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


AudioFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]


class SpeechRequest(BaseModel):
    model: str = Field(default="edge-tts", description="TTS model/engine name or alias ('edge-tts', 'piper', 'kokoro', 'tts-1', 'tts-1-hd')")
    input: str = Field(..., max_length=10000, description="The text to generate audio for")
    voice: str = Field(default="en-US-AriaNeural", description="Voice ID or alias")
    response_format: Optional[AudioFormat] = Field(default=None, description="Audio format: mp3, opus, aac, flac, wav, pcm")
    speed: Optional[float] = Field(default=1.0, ge=0.25, le=4.0, description="Speed of generated audio")
    stream: Optional[bool] = Field(default=False, description="Whether to stream audio chunks directly")


class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "universal-tts"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelObject]


class VoiceObject(BaseModel):
    id: str
    name: str
    engine: str
    language: str
    gender: Optional[str] = None
    sample_rate: Optional[int] = None


class VoiceListResponse(BaseModel):
    object: str = "list"
    voices: List[VoiceObject]


class HealthResponse(BaseModel):
    status: str
    version: str = "0.1.0"
    default_engine: str
    default_voice: str
    engines: Dict[str, Any]
    cache: Dict[str, Any]
