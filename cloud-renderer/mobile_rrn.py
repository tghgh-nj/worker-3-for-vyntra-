"""Mobile RRN video super-resolution through the official LiteRT model."""
from __future__ import annotations

import subprocess
import time
import urllib.request
from fractions import Fraction
from pathlib import Path
from typing import Callable

import numpy as np

MODEL_URL = (
    "https://raw.githubusercontent.com/MediaTek-NeuroPilot/mai22-real-time-video-sr/"
    "main/snapshot/mobile_rrn_16/rrn_16.tflite"
)
MODEL_MIN_BYTES = 100_000
SCALE = 4
MAX_SOURCE = (1280, 720)
MAX_OUTPUT_PIXELS = 33_177_600


def _read_exact(stream, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            if remaining == size:
                return b""
            raise RuntimeError("mobile_rrn_decoder_truncated_frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _download_model(path: Path, heartbeat: Callable[[], None] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size >= MODEL_MIN_BYTES:
        with path.open("rb") as model_file:
            model_file.seek(4)
            if model_file.read(4) == b"TFL3":
                return path
        path.unlink(missing_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    partial.unlink(missing_ok=True)
    request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Vyntra-Cloud-Renderer/1.0"})
    last_heartbeat = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=30) as response, partial.open("wb") as output:
            while True:
                chunk = response.read(256 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                if heartbeat and time.monotonic() - last_heartbeat >= 5:
                    heartbeat()
                    last_heartbeat = time.monotonic()
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    with partial.open("rb") as model_file:
        model_file.seek(4)
        is_tflite = model_file.read(4) == b"TFL3"
    if partial.stat().st_size < MODEL_MIN_BYTES or not is_tflite:
        partial.unlink(missing_ok=True)
        raise RuntimeError("mobile_rrn_model_invalid")
    partial.replace(path)
    return path


def _load_interpreter(model_path: Path, width: int, height: int, threads: int, heartbeat=None):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError as exc:
        raise RuntimeError("mobile_rrn_runtime_unavailable") from exc
    model_path = _download_model(model_path, heartbeat)
    interpreter = Interpreter(model_path=str(model_path), num_threads=max(1, min(4, threads)))
    try:
        inputs = interpreter.get_input_details()
        if len(inputs) != 2:
            raise RuntimeError("mobile_rrn_input_signature_invalid")
        pair = next((item for item in inputs if item["shape"][-1] == 6), None)
        state = next((item for item in inputs if item["shape"][-1] == 16), None)
        if pair is None or state is None:
            raise RuntimeError("mobile_rrn_input_signature_invalid")
        interpreter.resize_tensor_input(pair["index"], [1, height, width, 6], strict=False)
        interpreter.resize_tensor_input(state["index"], [1, height, width, 16], strict=False)
        interpreter.allocate_tensors()
        inputs = interpreter.get_input_details()
        outputs = interpreter.get_output_details()
        pair = next((item for item in inputs if item["shape"][-1] == 6), None)
        state = next((item for item in inputs if item["shape"][-1] == 16), None)
        image = next((item for item in outputs if item["shape"][-1] == 3), None)
        hidden = next((item for item in outputs if item["shape"][-1] == 16), None)
        if not all((pair, state, image, hidden)):
            raise RuntimeError("mobile_rrn_tensor_signature_invalid")
        if any(item["dtype"] != np.float32 for item in (pair, state, image, hidden)):
            raise RuntimeError("mobile_rrn_tensor_dtype_invalid")
        if list(image["shape"]) != [1, height * SCALE, width * SCALE, 3]:
            raise RuntimeError("mobile_rrn_output_shape_invalid")
        return interpreter, pair["index"], state["index"], image["index"], hidden["index"]
    except Exception:
        interpreter = None
        raise


def upscale_video(
    source: Path,
    destination: Path,
    width: int,
    height: int,
    fps: float,
    duration: float,
    cache: Path,
    progress: Callable[[int, int], None] | None = None,
    heartbeat: Callable[[], None] | None = None,
) -> tuple[int, int]:
    """Run recurrent inference frame by frame, keeping one model instance per job."""
    if width < 1 or height < 1 or max(width, height) > MAX_SOURCE[0] or min(width, height) > MAX_SOURCE[1]:
        raise ValueError("mobile_rrn_source_must_be_720p_or_lower")
    if fps <= 0 or duration <= 0 or duration > 180:
        raise ValueError("mobile_rrn_clip_limit_180_seconds")
    out_width, out_height = width * SCALE, height * SCALE
    if out_width * out_height > MAX_OUTPUT_PIXELS:
        raise ValueError("mobile_rrn_output_dimensions_too_large")
    total_frames = max(1, round(fps * duration))
    try:
        from ai_runtime import cpu_thread_count
    except ImportError:
        cpu_thread_count = lambda: 2
    interpreter, pair_index, state_index, image_index, hidden_index = _load_interpreter(
        cache / "mobile_rrn_16.tflite", width, height, cpu_thread_count(), heartbeat,
    )
    frame_bytes = width * height * 3
    state = np.zeros((1, height, width, 16), dtype=np.float32)
    decoder_cmd = [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-noautorotate",
        "-i", str(source), "-map", "0:v:0", "-an", "-sn", "-dn", "-vsync", "0",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    encoder_cmd = [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{out_width}x{out_height}",
        "-r", str(Fraction(str(fps)).limit_denominator(1001)), "-i", "pipe:0", "-i", str(source),
        "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart", str(destination),
    ]
    decoder = encoder = None
    previous = None
    completed = 0
    last_heartbeat = time.monotonic()
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        decoder = subprocess.Popen(decoder_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        encoder = subprocess.Popen(encoder_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        while True:
            raw = _read_exact(decoder.stdout, frame_bytes)
            if not raw:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
            current = frame.astype(np.float32)
            if previous is None:
                previous = current
            pair = np.concatenate((previous, current), axis=-1)[None, ...]
            interpreter.set_tensor(pair_index, pair)
            interpreter.set_tensor(state_index, state)
            interpreter.invoke()
            enhanced = interpreter.get_tensor(image_index)[0]
            next_state = interpreter.get_tensor(hidden_index)
            encoder.stdin.write(np.clip(enhanced, 0, 255).astype(np.uint8).tobytes())
            # Carry recurrent state only within this job; do not use global state
            # across jobs or worker threads.
            state = next_state.copy()
            previous = current
            completed += 1
            if progress:
                progress(completed, total_frames)
            if heartbeat and time.monotonic() - last_heartbeat >= 5:
                heartbeat()
                last_heartbeat = time.monotonic()
        if completed == 0:
            raise RuntimeError("mobile_rrn_decoder_no_frames")
        encoder.stdin.close()
        encoder.stdin = None
        _, decoder_error = decoder.communicate(timeout=120)
        _, encoder_error = encoder.communicate(timeout=900)
        if decoder.returncode:
            raise RuntimeError("mobile_rrn_decoder_failed:" + decoder_error.decode("utf-8", "replace")[-400:])
        if encoder.returncode:
            raise RuntimeError("mobile_rrn_encoder_failed:" + encoder_error.decode("utf-8", "replace")[-400:])
    finally:
        for process in (decoder, encoder):
            if process and process.poll() is None:
                process.kill()
                process.wait()
        interpreter = None
    if progress:
        progress(completed, total_frames)
    return out_width, out_height
