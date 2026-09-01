import argparse
import os
from pathlib import Path
import urllib.request
import sys

MODELS = {
    "kokoro": [
        (
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
            "models/kokoro/kokoro-v1.0.onnx",
        ),
        (
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
            "models/kokoro/voices-v1.0.bin",
        ),
    ],
    "piper": [
        (
            "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx",
            "models/piper/en_US-ryan-medium.onnx",
        ),
        (
            "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json",
            "models/piper/en_US-ryan-medium.onnx.json",
        ),
    ],
}


def download_file(url: str, dest_path: str):
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"✓ Already exists: {dest_path}")
        return

    print(f"⬇ Downloading {url} -> {dest_path}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"✓ Downloaded: {dest_path} ({dest.stat().st_size / (1024*1024):.2f} MB)")
    except Exception as e:
        print(f"✗ Failed to download {url}: {e}", file=sys.stderr)
        if dest.exists():
            dest.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Download default TTS model files")
    parser.add_argument("--engine", choices=["all", "kokoro", "piper"], default="all")
    args = parser.parse_args()

    targets = ["kokoro", "piper"] if args.engine == "all" else [args.engine]
    for engine in targets:
        print(f"\n--- Setting up {engine.upper()} models ---")
        for url, path in MODELS[engine]:
            download_file(url, path)

    print("\n✓ Model setup complete.")


if __name__ == "__main__":
    main()
