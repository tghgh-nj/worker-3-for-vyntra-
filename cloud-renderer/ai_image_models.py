"""Pinned pretrained AI super-resolution backends for Status HD.

Image-first networks run on a still image or on every decoded video frame.
Temporal networks are deliberately not routed through this module.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

MAX_IMAGE_PIXELS = 8_000_000
MAX_IMAGE_OUTPUT_PIXELS = 33_177_600
MAX_VIDEO_SOURCE_LONG_EDGE = 1920
MAX_VIDEO_SOURCE_SHORT_EDGE = 1080
TILE = 128
OVERLAP = 16

WEIGHTS = {
    # Official BasicSR EDSR-M x4 checkpoint (Google Drive file ID from BasicSR's
    # official download_pretrained_models.py). The fixed file ID identifies the asset.
    "edsr": {
        "kind": "gdrive",
        "id": "1bCK6cFYU01uJudLgUUe-jgx-tZ3ikOWn",
        "filename": "EDSR_Mx4_f64b16_DIV2K_official-0c287733.pth",
        "sha256": None,
    },
    # Official SwinIR architecture checkpoint mirrored byte-for-byte with a published SHA-256.
    "swinir": {
        "url": "https://huggingface.co/uwg/upscaler/resolve/7aeaa0b9279e22605ccb41ebcbc90c5c7ebfdf33/SwinIR/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth",
        "filename": "SwinIR-M-x4.pth",
        "sha256": "4e78e33f22c1aa8a773db0cf4a7381bae97c2362c717f155439ebc690cbd9215",
    },
    # Official Real-ESRGAN release checkpoint.
    "realesrgan": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "filename": "RealESRGAN_x4plus.pth",
        "sha256": "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
    },
    # OmniSR X4 is an Apache-2.0 lightweight transformer checkpoint. Safetensors
    # avoids pickle-based model deserialization and the commit pins the asset.
    "omnisr": {
        "url": "https://huggingface.co/Acly/Omni-SR/resolve/b28a7caa0baba8669de3e49687e76e12448ea020/OmniSR_X4_DIV2K.safetensors",
        "filename": "OmniSR_X4_DIV2K.safetensors",
        "sha256": "dff25e4ed392cb5cbe534d920e292063a0555df9281c54c5ec321490a2a59832",
    },
    "omnisr2": {
        "url": "https://huggingface.co/Acly/Omni-SR/resolve/b28a7caa0baba8669de3e49687e76e12448ea020/OmniSR_X2_DIV2K.safetensors",
        "filename": "OmniSR_X2_DIV2K.safetensors",
        "sha256": "79408fc23203bf161faa957c4a602cc40521ed2235a72d976bd9d375e6644611",
    },
    # BSRGAN is a practical blind-restoration model for real-world photos.
    "bsrgan": {
        "url": "https://huggingface.co/kadirnar/bsrgan/resolve/e80840d7348b9bc61b67a570d29e70eaa9293d32/BSRGAN.pth",
        "filename": "BSRGAN.pth",
        "sha256": "5d505a0766160921e0388d76e1ddf08cb114303990f9080432bf2b1c988b1c54",
    },
    # HAT offers clean-image and perceptual real-photo checkpoints.
    "hat": {
        "url": "https://huggingface.co/Acly/hat/resolve/8403819bcbf5959d54c72383f0725f2525472d30/HAT_SRx4_ImageNet-pretrain.pth",
        "filename": "HAT_SRx4_ImageNet-pretrain.pth",
        "sha256": "4ee053c42461187846dc0e93aa5abd34591c0725a8e044a59000e92ee215e833",
    },
    "realhat": {
        "url": "https://huggingface.co/Acly/hat/resolve/8403819bcbf5959d54c72383f0725f2525472d30/Real_HAT_GAN_sharper.pth",
        "filename": "Real_HAT_GAN_sharper.pth",
        "sha256": "5800b67136006eb8cab3b4ed7c8d73b6a195bb18e6cc709b674f9aa069c00271",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_weight(model_id: str, cache: Path) -> Path:
    if model_id not in WEIGHTS:
        raise ValueError("image_model_not_available")
    spec = WEIGHTS[model_id]
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / spec["filename"]
    expected = spec.get("sha256")
    if path.is_file() and expected and _sha256(path) != expected:
        path.unlink()
    if path.is_file() and path.stat().st_size > 1024 * 1024:
        return path
    path.unlink(missing_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.unlink(missing_ok=True)
    if spec.get("kind") == "gdrive":
        try:
            import gdown
        except ImportError as exc:
            raise RuntimeError("edsr_downloader_unavailable") from exc
        downloaded = gdown.download(id=spec["id"], output=str(temporary), quiet=True, fuzzy=True)
        if not downloaded or not temporary.is_file():
            temporary.unlink(missing_ok=True)
            raise RuntimeError("edsr_weight_download_failed")
    else:
        request = urllib.request.Request(spec["url"], headers={"User-Agent": "Vyntra-StatusHD-Renderer/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    if temporary.stat().st_size < 1024 * 1024:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("ai_model_weight_incomplete")
    if expected and _sha256(temporary) != expected:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("ai_model_weight_hash_mismatch")
    temporary.replace(path)
    return path


def _load_model(model_id: str, cache: Path):
    try:
        import torch
        from spandrel import ModelLoader
        import spandrel_extra_arches  # noqa: F401 - registers additional architectures
    except ImportError as exc:
        raise RuntimeError("ai_image_runtime_unavailable") from exc
    weight_path = _download_weight(model_id, cache)
    descriptor = ModelLoader().load_from_file(weight_path)
    scale = int(descriptor.scale)
    if scale not in (2, 3, 4, 8):
        raise RuntimeError("ai_model_unsupported_scale")
    model = descriptor.model.to("cpu").eval()
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    return torch, model, scale


def _infer_image(torch, model, source: Image.Image, scale: int) -> Image.Image:
    if source.width * source.height > MAX_IMAGE_PIXELS:
        raise ValueError("ai_image_pixel_limit")
    if source.width * source.height * scale * scale > MAX_IMAGE_OUTPUT_PIXELS:
        raise ValueError("ai_image_output_pixel_limit")
    rgb = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    height, width = rgb.shape[:2]
    result = np.empty((height * scale, width * scale, 3), dtype=np.uint8)
    with torch.inference_mode():
        for core_y in range(0, height, TILE):
            for core_x in range(0, width, TILE):
                y0, x0 = max(0, core_y - OVERLAP), max(0, core_x - OVERLAP)
                y1, x1 = min(height, core_y + TILE + OVERLAP), min(width, core_x + TILE + OVERLAP)
                tile = torch.from_numpy(rgb[y0:y1, x0:x1].copy()).permute(2, 0, 1).unsqueeze(0)
                output = model(tile).squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                output = np.clip(output * 255.0 + 0.5, 0, 255).astype(np.uint8)
                core_x0, core_y0 = (core_x - x0) * scale, (core_y - y0) * scale
                core_x1 = core_x0 + min(TILE, width - core_x) * scale
                core_y1 = core_y0 + min(TILE, height - core_y) * scale
                result[core_y * scale:(core_y + min(TILE, height - core_y)) * scale,
                       core_x * scale:(core_x + min(TILE, width - core_x)) * scale] = output[core_y0:core_y1, core_x0:core_x1]
    return Image.fromarray(result, mode="RGB")


def upscale_image(source: Path, destination: Path, model_id: str, cache: Path) -> tuple[int, int]:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError("ai_image_pixel_limit")
    torch, model, scale = _load_model(model_id, cache)
    enhanced = _infer_image(torch, model, image, scale)
    enhanced.save(destination, format="PNG")
    return enhanced.size


def _read_exact(stream, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            if remaining == size:
                return b""
            raise RuntimeError("ai_video_decoder_truncated_frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def upscale_video(source: Path, destination: Path, model_id: str, width: int,
                  height: int, fps: float, duration: float, cache: Path) -> tuple[int, int]:
    """Enhance every decoded frame with an image model; preserve source audio."""
    if width < 1 or height < 1 or max(width, height) > MAX_VIDEO_SOURCE_LONG_EDGE or min(width, height) > MAX_VIDEO_SOURCE_SHORT_EDGE:
        raise ValueError("ai_video_source_must_be_1080p_or_lower")
    if fps <= 0 or fps > 60 or duration <= 0 or duration > 180:
        raise ValueError("ai_video_source_limit")
    torch, model, scale = _load_model(model_id, cache)
    output_width, output_height = width * scale, height * scale
    if output_width * output_height > 33_177_600:
        raise ValueError("ai_video_output_pixel_limit")
    decoder_cmd = [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-noautorotate",
        "-i", str(source), "-map", "0:v:0", "-an", "-sn", "-dn",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    encoder_cmd = [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{output_width}x{output_height}",
        "-r", str(fps), "-i", "pipe:0", "-i", str(source),
        "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart", str(destination),
    ]
    decoder = encoder = None
    try:
        decoder = subprocess.Popen(decoder_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        encoder = subprocess.Popen(encoder_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        while True:
            raw = _read_exact(decoder.stdout, width * height * 3)
            if not raw:
                break
            frame = Image.fromarray(np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3), mode="RGB")
            enhanced = _infer_image(torch, model, frame, scale)
            try:
                encoder.stdin.write(np.asarray(enhanced, dtype=np.uint8).tobytes())
            except (BrokenPipeError, OSError) as exc:
                raise RuntimeError("ai_video_encoder_failed") from exc
        encoder.stdin.close()
        encoder.stdin = None
        decoder_stdout, decoder_stderr = decoder.communicate(timeout=120)
        encoder_stdout, encoder_stderr = encoder.communicate(timeout=900)
        if decoder.returncode != 0:
            raise RuntimeError("ai_video_decoder_failed:" + decoder_stderr.decode("utf-8", "replace")[-500:])
        if encoder.returncode != 0:
            raise RuntimeError("ai_video_encoder_failed:" + encoder_stderr.decode("utf-8", "replace")[-500:])
    finally:
        for process in (decoder, encoder):
            if process and process.poll() is None:
                process.kill()
                process.wait()
    return output_width, output_height
