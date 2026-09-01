import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_OPENAI_VOICE_MAP = {
    "alloy": "en-US-AriaNeural",
    "echo": "en-US-GuyNeural",
    "fable": "en-GB-SoniaNeural",
    "onyx": "en-US-ChristopherNeural",
    "nova": "en-US-JennyNeural",
    "shimmer": "en-US-AnaNeural",
}

DEFAULT_SHORT_ALIASES = {
    "aria": "en-US-AriaNeural",
    "jenny": "en-US-JennyNeural",
    "guy": "en-US-GuyNeural",
    "sonia": "en-GB-SoniaNeural",
    "ryan": "en_US-ryan-medium",
    "amy": "en_US-amy-medium",
    "lessac": "en_US-lessac-medium",
    "heart": "af_heart",
    "bella": "af_bella",
    "sarah": "af_sarah",
    "nicole": "af_nicole",
    "adam": "am_adam",
    "michael": "am_michael",
}


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str = ""
    cors_origins: List[str] = ["*"]


class DefaultsConfig(BaseModel):
    engine: str = "edge-tts"
    voice: str = "en-US-AriaNeural"
    speed: float = 1.0
    response_format: str = "mp3"


class CacheConfig(BaseModel):
    enabled: bool = True
    max_entries: int = 500
    max_memory_mb: int = 50
    ttl_seconds: int = 86400


class CircuitBreakerConfig(BaseModel):
    enabled: bool = True
    timeout_seconds: float = 5.0
    fallback_engine: str = "piper"
    fallback_voice: str = "en_US-ryan-medium"


class PathsConfig(BaseModel):
    piper_models_dir: str = "models/piper"
    kokoro_model_path: str = "models/kokoro/kokoro-v1.0.onnx"
    kokoro_voices_path: str = "models/kokoro/voices-v1.0.bin"


class EngineDetailConfig(BaseModel):
    enabled: bool = True
    default_voice: str
    native_format: str = "wav"
    timeout_seconds: Optional[float] = None


class EnginesConfig(BaseModel):
    edge_tts: EngineDetailConfig = EngineDetailConfig(
        enabled=True, default_voice="en-US-AriaNeural", native_format="mp3", timeout_seconds=5.0
    )
    piper: EngineDetailConfig = EngineDetailConfig(
        enabled=True, default_voice="en_US-ryan-medium", native_format="wav"
    )
    kokoro: EngineDetailConfig = EngineDetailConfig(
        enabled=True, default_voice="af_heart", native_format="wav"
    )


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TTS_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    server: ServerConfig = Field(default_factory=ServerConfig)
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    engines: EnginesConfig = Field(default_factory=EnginesConfig)
    openai_voice_map: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_OPENAI_VOICE_MAP))
    short_aliases: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_SHORT_ALIASES))

    @classmethod
    def load_from_yaml(cls, config_path: Optional[str] = None) -> "AppSettings":
        path_to_try = config_path or os.getenv("TTS_CONFIG_PATH", "config.yaml")
        yaml_data: Dict[str, Any] = {}
        if Path(path_to_try).exists():
            with open(path_to_try, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    yaml_data = loaded
        return cls(**yaml_data)


settings = AppSettings.load_from_yaml()
