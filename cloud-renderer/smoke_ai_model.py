from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "ai-model-smoke-test")
sys.path.insert(0, str(Path(__file__).parent))

import worker


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in worker.SUPPORTED_AI_MODELS:
        raise SystemExit("Usage: smoke_ai_model.py <espcn|edsr|swinir|realesrgan|basicvsr|vrt>")
    model = sys.argv[1]
    with tempfile.TemporaryDirectory(prefix="vyntra-ai-smoke-") as temporary:
        root = Path(temporary)
        source = root / "synthetic.mp4"
        output = root / f"{model}-output.mp4"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=24",
            "-frames:v", "6", "-pix_fmt", "yuv420p", "-c:v", "libx264",
            "-preset", "ultrafast", str(source),
        ], check=True, timeout=30)
        if model == "espcn":
            from espcn import upscale_video
            expected = (192, 192)
            upscale_video(source, output, 64, 64, 24.0, 0.25, root / "espcn.tflite")
        else:
            worker.ensure_ai_model_runtime(model, root / "ai-runtime")
            if model in ("edsr", "swinir", "realesrgan"):
                from ai_image_models import upscale_video
                expected = upscale_video(source, output, model, 64, 64, 24.0, 0.25, root / "model-cache")
            elif model == "vrt":
                from temporal_models import upscale_video
                expected = upscale_video(source, output, 64, 64, 24.0, 0.25, root / "model-cache" / "vrt")
            else:
                from temporal_models import upscale_basicvsr_video
                expected = upscale_basicvsr_video(source, output, 64, 64, 24.0, 0.25, root / "model-cache" / "basicvsr")
        if not output.is_file() or output.stat().st_size <= 0:
            raise RuntimeError(f"{model}_smoke_output_missing")
        metadata = worker.probe(output)
        stream = next((item for item in metadata.get("streams", []) if item.get("codec_type") == "video"), None)
        actual = (int(stream["width"]), int(stream["height"])) if stream else None
        if actual != expected:
            raise RuntimeError(f"{model}_smoke_dimensions_mismatch:{actual}:{expected}")
        subprocess.run(["ffmpeg", "-hide_banner", "-v", "error", "-i", str(output), "-f", "null", "-"], check=True, timeout=60)
        print(f"PASS {model}: {actual[0]}x{actual[1]}, {output.stat().st_size} bytes, full decode verified")


if __name__ == "__main__":
    main()
