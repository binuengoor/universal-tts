import pytest
from httpx import ASGITransport, AsyncClient

from gateway.config import AppSettings
from gateway.main import app
from gateway.router import TTSRouter


@pytest.mark.asyncio
async def test_curated_voices_default():
    settings = AppSettings.load_from_yaml()
    router = TTSRouter(settings=settings)

    # By default, should return the curated subset
    voices = await router.get_curated_voices(all=False)
    assert len(voices) > 0
    # All voices returned should be in the curated config
    curated_ids = {v.id for v in settings.curated_voices.voices}
    for v in voices:
        assert v.id in curated_ids


@pytest.mark.asyncio
async def test_curated_voices_filtering_locale():
    settings = AppSettings.load_from_yaml()
    router = TTSRouter(settings=settings)

    # Filter for ml-IN (Malayalam)
    ml_voices = await router.get_curated_voices(all=False, locale="ml-IN")
    assert len(ml_voices) > 0
    for v in ml_voices:
        assert v.language == "ml-IN"

    # Filter for hi-IN (Hindi)
    hi_voices = await router.get_curated_voices(all=False, locale="hi-IN")
    assert len(hi_voices) > 0
    for v in hi_voices:
        assert v.language == "hi-IN"

    # Filter for en-GB
    gb_voices = await router.get_curated_voices(all=False, locale="en-GB")
    assert len(gb_voices) > 0
    for v in gb_voices:
        assert v.language == "en-GB"


@pytest.mark.asyncio
async def test_curated_voices_filtering_engine():
    settings = AppSettings.load_from_yaml()
    router = TTSRouter(settings=settings)

    kokoro_voices = await router.get_curated_voices(all=False, engine="kokoro")
    assert len(kokoro_voices) > 0
    for v in kokoro_voices:
        assert v.engine == "kokoro"

    google_voices = await router.get_curated_voices(all=False, engine="google-cloud")
    assert len(google_voices) > 0
    for v in google_voices:
        assert v.engine == "google-cloud"


@pytest.mark.asyncio
async def test_curated_voices_endpoint_api():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # 1. Default curated list
        resp = await ac.get("/v1/voices")
        assert resp.status_code == 200
        data = resp.json()
        voices = data["voices"]
        assert len(voices) > 0
        # Should be filtered subset (~30-40 voices, not 2000+)
        assert len(voices) < 100

        # 2. Filter by locale
        resp_ml = await ac.get("/v1/voices?locale=ml-IN")
        assert resp_ml.status_code == 200
        ml_data = resp_ml.json()["voices"]
        assert len(ml_data) >= 2
        for v in ml_data:
            assert v["language"] == "ml-IN"

        # 3. Filter by engine
        resp_piper = await ac.get("/v1/voices?engine=piper")
        assert resp_piper.status_code == 200
        piper_data = resp_piper.json()["voices"]
        for v in piper_data:
            assert v["engine"] == "piper"

        # 4. Filter with both locale and engine
        resp_kokoro_gb = await ac.get("/v1/voices?engine=kokoro&locale=en-GB")
        assert resp_kokoro_gb.status_code == 200
        kokoro_gb_data = resp_kokoro_gb.json()["voices"]
        assert len(kokoro_gb_data) == 3
        for v in kokoro_gb_data:
            assert v["engine"] == "kokoro"
            assert v["language"] == "en-GB"
