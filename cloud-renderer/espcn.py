"""Bounded CPU ESPCN preprocessing for Status HD cloud exports."""
from __future__ import annotations

import hashlib
import subprocess
import time
import urllib.request
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image
from ai_runtime import cpu_thread_count

MODEL_URL = "https://storage.googleapis.com/ailia-models-tflite/espcn/espcn.tflite"
MODEL_SHA256 = "e1607d2ce4f02b2f03b74fc5699cf709570e9d3d407dd9e69e9f16806bfd77d54"
TILE = 64
SCALE = 3
MAX_PHOTO_SOURCE_EDGE = 854
MAX_PHOTO_SOURCE_SHORT_EDGE = 480
MAX_VIDEO_SOURCE_EDGE = 1920
MAX_VIDEO_SOURCE_SHORT_EDGE = 1080
MAX_CLIP_SECONDS = 180.0
MAX_INPUT_FPS = 24.0


def validate_espcn_source(width: int, height: int, fps: float, duration: float) -> None:
    if width < 1 or height < 1 or max(width, height) > MAX_VIDEO_SOURCE_EDGE or min(width, height) > MAX_VIDEO_SOURCE_SHORT_EDGE:
        raise ValueError("espcn_source_must_be_1080p_or_lower")
    if fps <= 0 or fps > MAX_INPUT_FPS:
        raise ValueError("espcn_frame_rate_limit_24fps")
    if duration <= 0 or duration > MAX_CLIP_SECONDS:
        raise ValueError("espcn_clip_limit_180_seconds")


def load_model(model_path: Path, heartbeat=None):
    """Download the Apache-2.0 source model and verify its pinned SHA-256."""
    if not model_path.exists():
        request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Vyntra-StatusHD-Renderer/1.0"})
        last_heartbeat = time.monotonic()
        with urllib.request.urlopen(request, timeout=30) as response, model_path.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                if heartbeat and time.monotonic() - last_heartbeat >= 5:
                    heartbeat()
                    last_heartbeat = time.monotonic()
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        model_path.unlink(missing_ok=True)
        raise RuntimeError("espcn_model_hash_mismatch")
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError as exc:
        raise RuntimeError("espcn_runtime_unavailable") from exc
    interpreter = Interpreter(model_path=str(model_path), num_threads=cpu_thread_count())
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    if list(input_detail["shape"]) != [1, 3, 64, 64] or list(output_detail["shape"]) != [1, 3, 192, 192]:
        raise RuntimeError("espcn_model_shape_mismatch")
    if input_detail["dtype"] != np.float32 or output_detail["dtype"] != np.float32:
        raise RuntimeError("espcn_model_dtype_mismatch")
    return interpreter, input_detail["index"], output_detail["index"]


def upscale_image(source: Path, destination: Path, model_path: Path, heartbeat=None, progress=None) -> None:
    """Enhance one still image with ESPCN and save the actual 3x image result."""
    image = Image.open(source).convert("RGB")
    width, height = image.size
    if width < 1 or height < 1 or max(width, height) > MAX_PHOTO_SOURCE_EDGE or min(width, height) > MAX_PHOTO_SOURCE_SHORT_EDGE:
        raise ValueError("espcn_source_must_be_480p_or_lower")
    interpreter, input_index, output_index = load_model(model_path, heartbeat)
    try:
        frame = np.asarray(image, dtype=np.uint8)
        enlarged = np.empty((height * SCALE, width * SCALE, 3), dtype=np.uint8)
        tile_count = max(1, ((height + TILE - 1) // TILE) * ((width + TILE - 1) // TILE))
        completed_tiles = 0
        last_heartbeat = time.monotonic()
        for top in range(0, height, TILE):
            tile_h = min(TILE, height - top)
            for left in range(0, width, TILE):
                tile_w = min(TILE, width - left)
                tile = frame[top:top+tile_h, left:left+tile_w]
                if tile_h != TILE or tile_w != TILE:
                    tile = np.asarray(Image.fromarray(tile).resize((TILE, TILE), Image.Resampling.BICUBIC))
                tensor = np.transpose(tile.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
                interpreter.set_tensor(input_index, tensor)
                interpreter.invoke()
                result = interpreter.get_tensor(output_index)[0]
                rgb = np.transpose(result, (1, 2, 0))
                rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
                if tile_h != TILE or tile_w != TILE:
                    rgb = np.asarray(Image.fromarray(rgb).resize((tile_w*SCALE, tile_h*SCALE), Image.Resampling.BICUBIC))
                enlarged[top*SCALE:(top+tile_h)*SCALE, left*SCALE:(left+tile_w)*SCALE] = rgb
                completed_tiles += 1
                if progress: progress(completed_tiles / tile_count)
                if heartbeat and time.monotonic() - last_heartbeat >= 5:
                    heartbeat()
                    last_heartbeat = time.monotonic()
        Image.fromarray(enlarged, mode="RGB").save(destination, format="PNG")
    finally:
        interpreter = None


def _read_exact(stream, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            if remaining == size:
                return b""
            raise RuntimeError("espcn_decoder_truncated_frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def upscale_video(source: Path, destination: Path, width: int, height: int,
                  fps: float, duration: float, model_path: Path, progress=None, heartbeat=None) -> None:
    """Run ESPCN on every source frame, then retain source audio in an MP4."""
    validate_espcn_source(width, height, fps, duration)
    interpreter, input_index, output_index = load_model(model_path, heartbeat)
    rate = Fraction(str(fps)).limit_denominator(1001)
    raw_size = width * height * 3
    decoder_cmd = [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-noautorotate",
        "-i", str(source), "-map", "0:v:0", "-an", "-sn", "-dn",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    encoder_cmd = [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width*SCALE}x{height*SCALE}",
        "-r", str(rate), "-i", "pipe:0", "-i", str(source),
        "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart", str(destination),
    ]
    decoder = encoder = None
    try:
        decoder = subprocess.Popen(decoder_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        encoder = subprocess.Popen(encoder_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        frame_index = 0
        expected_frames = max(1, int(duration * fps))
        last_heartbeat = time.monotonic()
        while True:
            raw = _read_exact(decoder.stdout, raw_size)
            if not raw:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            enlarged = np.empty((height * SCALE, width * SCALE, 3), dtype=np.uint8)
            for top in range(0, height, TILE):
                tile_h = min(TILE, height - top)
                for left in range(0, width, TILE):
                    tile_w = min(TILE, width - left)
                    tile = frame[top:top+tile_h, left:left+tile_w]
                    if tile_h != TILE or tile_w != TILE:
                        tile = np.asarray(Image.fromarray(tile).resize((TILE, TILE), Image.Resampling.BICUBIC))
                    tensor = np.transpose(tile.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
                    interpreter.set_tensor(input_index, tensor)
                    interpreter.invoke()
                    result = interpreter.get_tensor(output_index)[0]
                    rgb = np.transpose(result, (1, 2, 0))
                    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
                    if tile_h != TILE or tile_w != TILE:
                        rgb = np.asarray(Image.fromarray(rgb).resize((tile_w*SCALE, tile_h*SCALE), Image.Resampling.BICUBIC))
                    enlarged[top*SCALE:(top+tile_h)*SCALE, left*SCALE:(left+tile_w)*SCALE] = rgb
            encoder.stdin.write(enlarged.tobytes())
            frame_index += 1
            if progress and (frame_index % 8 == 0 or frame_index == expected_frames):
                progress(min(10, max(1, int(frame_index * 10 / expected_frames))))
            if heartbeat and time.monotonic() - last_heartbeat >= 5:
                heartbeat()
                last_heartbeat = time.monotonic()
        decoder.stdout.close()
        decoder_code = decoder.wait(timeout=30)
        decoder_error = decoder.stderr.read().decode("utf-8", errors="replace")
        encoder.stdin.close()
        encoder_code = encoder.wait(timeout=120)
        encoder_error = encoder.stderr.read().decode("utf-8", errors="replace")
        if decoder_code != 0:
            raise RuntimeError("espcn_decode_failed:" + decoder_error[-800:])
        if encoder_code != 0 or frame_index == 0:
            raise RuntimeError("espcn_encode_failed:" + encoder_error[-800:])
    except Exception:
        for proc in (decoder, encoder):
            if proc and proc.poll() is None:
                proc.kill()
                proc.wait()
        destination.unlink(missing_ok=True)
        raise
    finally:
        interpreter = None
