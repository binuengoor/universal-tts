from contextlib import asynccontextmanager
import logging
import os
from typing import Optional
from fastapi import Depends, FastAPI, Header, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from gateway.audio import get_mime_type
from gateway.cache import AudioCache
from gateway.config import AppSettings, settings
from gateway.router import TTSRouter
from gateway.schemas import (
    HealthResponse,
    ModelListResponse,
    ModelObject,
    SpeechRequest,
    VoiceListResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("universal-tts")

# Global instances
cache = AudioCache(
    enabled=settings.cache.enabled,
    max_entries=settings.cache.max_entries,
    max_memory_mb=settings.cache.max_memory_mb,
    ttl_seconds=settings.cache.ttl_seconds,
)
router = TTSRouter(settings=settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Universal TTS Gateway...")
    logger.info("Default Engine: %s | Default Voice: %s", settings.defaults.engine, settings.defaults.voice)
    logger.info("Cache Enabled: %s (Max: %s entries, %s MB)", settings.cache.enabled, settings.cache.max_entries, settings.cache.max_memory_mb)
    yield
    logger.info("Shutting down Universal TTS Gateway...")


app = FastAPI(
    title="Universal TTS Gateway",
    description="OpenAI-compatible drop-in Text-to-Speech API gateway with multi-engine drivers (Edge-TTS, Piper, Kokoro).",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_api_key(authorization: Optional[str] = Header(None)) -> None:
    required_key = settings.server.api_key
    if not required_key:
        return  # No auth required

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or token != required_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )


@app.get("/")
async def root():
    return {
        "service": "Universal TTS Gateway",
        "version": "0.1.0",
        "description": "1:1 drop-in replacement for OpenAI and Kokoro-FastAPI TTS endpoints",
        "endpoints": {
            "speech": "POST /v1/audio/speech",
            "models": "GET /v1/models",
            "voices": "GET /v1/audio/voices (or /v1/voices)",
            "health": "GET /health",
        },
    }


@app.get("/health", response_model=HealthResponse)
@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    engine_health = await router.get_engine_health()
    all_ok = any(engine_health.values())
    return HealthResponse(
        status="ok" if all_ok else "degraded",
        version="0.1.0",
        default_engine=settings.defaults.engine,
        default_voice=settings.defaults.voice,
        engines=engine_health,
        cache=cache.get_stats(),
    )


@app.get("/v1/models", response_model=ModelListResponse, dependencies=[Depends(verify_api_key)])
async def list_models():
    model_ids = ["edge-tts", "piper", "kokoro", "google-cloud", "google-tts", "tts-1", "tts-1-hd"]
    models = [ModelObject(id=m) for m in model_ids]
    return ModelListResponse(data=models)


@app.get("/v1/audio/voices", response_model=VoiceListResponse, dependencies=[Depends(verify_api_key)])
@app.get("/v1/voices", response_model=VoiceListResponse, dependencies=[Depends(verify_api_key)])
async def list_voices():
    voices = await router.get_all_voices()
    return VoiceListResponse(voices=voices)


@app.post("/v1/audio/speech", dependencies=[Depends(verify_api_key)])
async def create_speech(request: SpeechRequest):
    if not request.input.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Input text cannot be empty.")

    engine, resolved_voice = router.resolve_engine_and_voice(request.model, request.voice)
    target_format = router.resolve_format(engine, request.response_format)
    speed = request.speed or 1.0

    # Cache Lookup
    cache_key = cache.generate_key(
        engine=engine.engine_name,
        voice=resolved_voice,
        text=request.input,
        speed=speed,
        response_format=target_format,
    )

    cached = cache.get(cache_key)
    if cached is not None:
        audio_bytes, fmt = cached
        mime = get_mime_type(fmt)
        return Response(
            content=audio_bytes,
            media_type=mime,
            headers={
                "X-Cache": "HIT",
                "X-Engine": engine.engine_name,
                "Content-Disposition": f'attachment; filename="speech.{fmt}"',
            },
        )

    # Cache Miss -> Synthesize
    try:
        audio_bytes, out_fmt, used_engine = await router.synthesize(
            text=request.input,
            model=request.model,
            voice=request.voice,
            speed=speed,
            response_format=request.response_format,
        )

        # Store in cache
        cache.put(cache_key, audio_bytes, out_fmt)

        mime = get_mime_type(out_fmt)
        return Response(
            content=audio_bytes,
            media_type=mime,
            headers={
                "X-Cache": "MISS",
                "X-Engine": used_engine,
                "Content-Disposition": f'attachment; filename="speech.{out_fmt}"',
            },
        )
    except Exception as e:
        logger.error("TTS Synthesis error: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"TTS synthesis failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", str(settings.server.port)))
    uvicorn.run(
        "gateway.main:app",
        host=settings.server.host,
        port=port,
        reload=False,
    )
