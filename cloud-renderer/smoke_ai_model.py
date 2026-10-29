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
        choices = "|".join(worker.SUPPORTED_AI_MODELS)
        raise SystemExit(f"Usage: smoke_ai_model.py <{choices}>")
    model = sys.argv[1]
    with tempfile.TemporaryDirectory(prefix="vyntra-ai-smoke-") as temporary:
        root = Path(temporary)
        is_photo = model in worker.AI_PHOTO_ONLY_MODELS
        if is_photo:
            from PIL import Image
            source = root / "synthetic.png"
            Image.new("RGB", (64, 64), (32, 96, 160)).save(source)
            output = root / f"{model}-output.png"
            worker.ensure_ai_model_runtime(model, root / "ai-runtime")
            from ai_image_models import upscale_image
            expected = upscale_image(source, output, model, root / "model-cache")
        elif model == "espcn":
            source = root / "synthetic.mp4"
            output = root / f"{model}-output.mp4"
            subprocess.run([
                "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=24",
                "-frames:v", "6", "-pix_fmt", "yuv420p", "-c:v", "libx264",
                "-preset", "ultrafast", str(source),
            ], check=True, timeout=30)
            from espcn import upscale_video
            expected = (192, 192)
            upscale_video(source, output, 64, 64, 24.0, 0.25, root / "espcn.tflite")
        else:
            source = root / "synthetic.mp4"
            output = root / f"{model}-output.mp4"
            subprocess.run([
                "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=24",
                "-frames:v", "6", "-pix_fmt", "yuv420p", "-c:v", "libx264",
                "-preset", "ultrafast", str(source),
            ], check=True, timeout=30)
            worker.ensure_ai_model_runtime(model, root / "ai-runtime")
            if model in worker.AI_TEMPORAL_MODELS:
                from temporal_models import upscale_basicvsr_video, upscale_mmagic_video, upscale_nanovsr_video, upscale_video
                if model == "vrt":
                    expected = upscale_video(source, output, 64, 64, 24.0, 0.25, root / "model-cache" / "vrt")
                elif model == "basicvsr":
                    expected = upscale_basicvsr_video(source, output, 64, 64, 24.0, 0.25, root / "model-cache" / "basicvsr")
                elif model == "mobile_rrn":
                    from mobile_rrn import upscale_video as upscale_mobile_rrn_video
                    expected = upscale_mobile_rrn_video(source, output, 64, 64, 24.0, 0.25, root / "model-cache" / "mobile_rrn")
                elif model in {"edvr", "tdan", "basicvsr_v1", "iconvsr", "realbasicvsr"}:
                    expected = upscale_mmagic_video(model, source, output, 64, 64, 24.0, 0.25, root / "model-cache" / model)
                else:
                    expected = upscale_nanovsr_video(source, output, 64, 64, 24.0, 0.25, root / "model-cache" / "nanovsr")
            else:
                from ai_image_models import upscale_video
                expected = upscale_video(source, output, model, 64, 64, 24.0, 0.25, root / "model-cache")
        if not output.is_file() or output.stat().st_size <= 0:
            raise RuntimeError(f"{model}_smoke_output_missing")
        if is_photo:
            from PIL import Image
            with Image.open(output) as rendered:
                rendered.verify()
            with Image.open(output) as rendered:
                actual = rendered.size
        else:
            metadata = worker.probe(output)
            stream = next((item for item in metadata.get("streams", []) if item.get("codec_type") == "video"), None)
            actual = (int(stream["width"]), int(stream["height"])) if stream else None
        if actual != expected:
            raise RuntimeError(f"{model}_smoke_dimensions_mismatch:{actual}:{expected}")
        if not is_photo:
            subprocess.run(["ffmpeg", "-hide_banner", "-v", "error", "-i", str(output), "-f", "null", "-"], check=True, timeout=60)
        print(f"PASS {model}: {actual[0]}x{actual[1]}, {output.stat().st_size} bytes, full decode verified")


if __name__ == "__main__":
    main()
