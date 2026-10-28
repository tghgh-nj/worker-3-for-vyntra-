"""Sequence-aware VRT video super-resolution adapter.

VRT is kept separate from image-only upscalers: it consumes overlapping six-frame
windows so neighbouring-frame information is used for every output frame.
"""
from __future__ import annotations

import importlib.util
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np
from ai_runtime import configure_torch, select_torch_device

VRT_SOURCE_COMMIT = "ed9498a9de21d80cde744b9ab5d3986e1a3bac83"
VRT_SOURCE_URL = f"https://raw.githubusercontent.com/JingyunLiang/VRT/{VRT_SOURCE_COMMIT}/models/network_vrt.py"
VRT_WEIGHT_URL = "https://github.com/JingyunLiang/VRT/releases/download/v0.0/001_VRT_videosr_bi_REDS_6frames.pth"
WINDOW = 6
OVERLAP = 2
SCALE = 4


class _Heartbeat:
    def __init__(self, callback: Callable[[], None] | None):
        self.callback = callback
        self.stop = threading.Event()
        self.thread = None

    def __enter__(self):
        if self.callback:
            def run():
                while not self.stop.wait(5):
                    try:
                        self.callback()
                    except Exception:
                        pass
            self.thread = threading.Thread(target=run, name="vyntra-temporal-heartbeat", daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=1)


def _download(url: str, destination: Path, minimum_bytes: int,
              heartbeat: Callable[[], None] | None = None) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size >= minimum_bytes:
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Vyntra-StatusHD-Renderer/1.0"})
    try:
        last_heartbeat = time.monotonic()
        with urllib.request.urlopen(request, timeout=30) as response, temporary.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                if heartbeat and time.monotonic() - last_heartbeat >= 5:
                    heartbeat()
                    last_heartbeat = time.monotonic()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if temporary.stat().st_size < minimum_bytes:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("vrt_asset_incomplete")
    temporary.replace(destination)
    return destination


def _load_vrt(cache: Path, heartbeat: Callable[[], None] | None = None):
    try:
        import torch
        import torch.nn.functional as F
        import torchvision  # noqa: F401 - required by the official VRT network
        from einops import rearrange  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("vrt_runtime_unavailable") from exc
    source_path = _download(VRT_SOURCE_URL, cache / "network_vrt.py", 10_000, heartbeat)
    source = source_path.read_text(encoding="utf-8")
    source = source.replace(
        "from distutils.version import LooseVersion",
        "from setuptools._distutils.version import LooseVersion",
    )
    source_path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("vyntra_vrt_network", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("vrt_network_loader_unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.VRT(
        upscale=4,
        img_size=[6, 64, 64],
        window_size=[6, 8, 8],
        depths=[8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4],
        indep_reconsts=[11, 12],
        embed_dims=[120, 120, 120, 120, 120, 120, 120, 180, 180, 180, 180, 180, 180],
        num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        pa_frames=2,
        deformable_groups=12,
    )
    checkpoint = _download(VRT_WEIGHT_URL, cache / "001_VRT_videosr_bi_REDS_6frames.pth", 100_000_000, heartbeat)
    try:
        with _Heartbeat(heartbeat):
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        with _Heartbeat(heartbeat):
            state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "params" in state:
        state = state["params"]
    if not isinstance(state, dict):
        raise RuntimeError("vrt_checkpoint_invalid")
    model.load_state_dict(state, strict=True)
    configure_torch(torch)
    device = select_torch_device(torch)
    model.eval().to(device)
    return torch, F, model, device


def _read_exact(stream, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            if remaining == size:
                return b""
            raise RuntimeError("vrt_decoder_truncated_frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def upscale_video(source: Path, destination: Path, width: int, height: int,
                  fps: float, duration: float, cache: Path,
                  progress: Callable[[float], None] | None = None,
                  heartbeat: Callable[[], None] | None = None) -> tuple[int, int]:
    """Run VRT over all frames in six-frame windows, keeping overlap for context."""
    if width < 1 or height < 1 or max(width, height) > 854 or min(width, height) > 480:
        raise ValueError("vrt_source_must_be_480p_or_lower")
    if fps <= 0 or fps > 30 or duration <= 0 or duration > 180:
        raise ValueError("vrt_video_limit")
    with _Heartbeat(heartbeat):
        torch, F, model, device = _load_vrt(cache, heartbeat)
    out_width, out_height = width * SCALE, height * SCALE
    decoder_cmd = [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-noautorotate",
        "-i", str(source), "-map", "0:v:0", "-an", "-sn", "-dn",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    encoder_cmd = [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{out_width}x{out_height}",
        "-r", str(fps), "-i", "pipe:0", "-i", str(source),
        "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart", str(destination),
    ]
    decoder = encoder = None
    frames: list[np.ndarray] = []
    emitted = 0
    frame_bytes = width * height * 3

    def infer_window(window_frames: list[np.ndarray]) -> list[bytes]:
        array = np.stack(window_frames, axis=0)
        tensor = torch.from_numpy(array.copy()).permute(0, 3, 1, 2).float().div_(255.0).unsqueeze(0).to(device)
        h_pad = (8 - height % 8) % 8
        w_pad = (8 - width % 8) % 8
        if h_pad:
            tensor = torch.cat([tensor, torch.flip(tensor[:, :, :, -h_pad:, :], [3])], dim=3)
        if w_pad:
            tensor = torch.cat([tensor, torch.flip(tensor[:, :, :, :, -w_pad:], [4])], dim=4)
        with _Heartbeat(heartbeat):
            with torch.inference_mode():
                result = model(tensor).detach().cpu()[0, :, :, :height*SCALE, :width*SCALE]
        result = result.permute(0, 2, 3, 1).clamp(0, 1).numpy()
        return [np.clip(frame * 255.0 + 0.5, 0, 255).astype(np.uint8).tobytes() for frame in result]

    try:
        decoder = subprocess.Popen(decoder_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        encoder = subprocess.Popen(encoder_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        while True:
            raw = _read_exact(decoder.stdout, frame_bytes)
            if not raw:
                break
            frames.append(np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy())
            if len(frames) < WINDOW:
                continue
            if emitted == 0:
                output_frames = infer_window(frames[:WINDOW])
                for frame in output_frames:
                    encoder.stdin.write(frame)
                emitted += WINDOW
                if progress: progress(min(1.0, emitted / max(1, round(fps * duration))))
                frames = frames[WINDOW-OVERLAP:]
            else:
                output_frames = infer_window(frames[:WINDOW])
                for frame in output_frames[OVERLAP:]:
                    encoder.stdin.write(frame)
                emitted += WINDOW - OVERLAP
                if progress: progress(min(1.0, emitted / max(1, round(fps * duration))))
                frames = frames[WINDOW-OVERLAP:]

        if frames:
            if emitted == 0:
                new_count = len(frames)
                context = frames
            else:
                new_count = max(0, len(frames) - OVERLAP)
                context = frames
            if new_count:
                missing = WINDOW - len(context)
                if missing <= len(context):
                    padded = context + list(reversed(context[-missing:])) if missing else context
                else:
                    padded = context + [context[-1]] * missing
                output_frames = infer_window(padded[:WINDOW])
                first_output = 0 if emitted == 0 else OVERLAP
                for frame in output_frames[first_output:first_output + new_count]:
                    encoder.stdin.write(frame)
                emitted += new_count
                if progress: progress(min(1.0, emitted / max(1, round(fps * duration))))
        encoder.stdin.close()
        encoder.stdin = None
        _, decoder_stderr = decoder.communicate(timeout=120)
        _, encoder_stderr = encoder.communicate(timeout=900)
        if decoder.returncode != 0:
            raise RuntimeError("vrt_decoder_failed:" + decoder_stderr.decode("utf-8", "replace")[-500:])
        if encoder.returncode != 0:
            raise RuntimeError("vrt_encoder_failed:" + encoder_stderr.decode("utf-8", "replace")[-500:])
    finally:
        for process in (decoder, encoder):
            if process and process.poll() is None:
                process.kill()
                process.wait()
    if progress: progress(1.0)
    return out_width, out_height


MMAGIC_SOURCE_COMMIT = "0a560bba9b79ebe78574e1d4cbbdd0e798e63568"
MMAGIC_SOURCE_URL = f"https://github.com/open-mmlab/mmagic/archive/{MMAGIC_SOURCE_COMMIT}.zip"
BASICVSR_WEIGHT_URL = "https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth"


def _ensure_basicvsr(cache: Path, heartbeat: Callable[[], None] | None = None):
    """Load BasicVSR++ through the official MMagic implementation on CPU."""
    import zipfile

    try:
        import mmcv  # noqa: F401
        import mmengine  # noqa: F401
        import mmagic  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("basicvsr_runtime_unavailable") from exc

    source_root = cache / "mmagic-source"
    config_path = source_root / "configs" / "basicvsr_pp" / "basicvsr-pp_c64n7_8xb1-600k_reds4.py"
    if not config_path.is_file():
        archive = _download(MMAGIC_SOURCE_URL, cache / "mmagic-source.zip", 1_000_000, heartbeat)
        source_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
            root_prefix = next((name.split("/")[0] + "/" for name in names if name.endswith("/configs/basicvsr_pp/basicvsr-pp_c64n7_8xb1-600k_reds4.py")), None)
            if root_prefix is None:
                raise RuntimeError("basicvsr_config_missing")
            for name in names:
                if not name.startswith(root_prefix) or not (name.startswith(root_prefix + "configs/") or name.startswith(root_prefix + "LICENSE")):
                    continue
                relative = name[len(root_prefix):]
                if not relative or ".." in Path(relative).parts:
                    continue
                target = source_root / relative
                if name.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(name) as incoming, target.open("wb") as outgoing:
                    outgoing.write(incoming.read())
    checkpoint = _download(BASICVSR_WEIGHT_URL, cache / "basicvsr_plusplus_reds4.pth", 10_000_000, heartbeat)
    try:
        import torch
        from mmagic.apis import init_model
    except ImportError as exc:
        raise RuntimeError("basicvsr_runtime_unavailable") from exc
    configure_torch(torch)
    device = select_torch_device(torch)
    with _Heartbeat(heartbeat):
        model = init_model(str(config_path), str(checkpoint), device=str(device))
    return torch, model, device


def upscale_basicvsr_video(source: Path, destination: Path, width: int, height: int,
                           fps: float, duration: float, cache: Path,
                           progress: Callable[[float], None] | None = None,
                           heartbeat: Callable[[], None] | None = None) -> tuple[int, int]:
    """Run BasicVSR++ on overlapping frame sequences; do not process frames independently."""
    if width < 1 or height < 1 or max(width, height) > 854 or min(width, height) > 480:
        raise ValueError("basicvsr_source_must_be_480p_or_lower")
    if fps <= 0 or fps > 30 or duration <= 0 or duration > 180:
        raise ValueError("basicvsr_video_limit")
    with _Heartbeat(heartbeat):
        torch, model, device = _ensure_basicvsr(cache, heartbeat)
    scale = 4
    out_width, out_height = width * scale, height * scale
    chunk_size, overlap = 8, 2
    frame_bytes = width * height * 3
    decoder_cmd = [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-noautorotate",
        "-i", str(source), "-map", "0:v:0", "-an", "-sn", "-dn",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    encoder_cmd = [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{out_width}x{out_height}",
        "-r", str(fps), "-i", "pipe:0", "-i", str(source),
        "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart", str(destination),
    ]
    decoder = encoder = None
    frames: list[np.ndarray] = []
    emitted = 0

    def infer_window(window_frames: list[np.ndarray]) -> list[bytes]:
        array = np.stack(window_frames, axis=0)
        tensor = torch.from_numpy(array.copy()).permute(0, 3, 1, 2).float().div_(255.0).unsqueeze(0).to(device)
        h_pad = (32 - height % 32) % 32
        w_pad = (32 - width % 32) % 32
        if h_pad:
            tensor = torch.cat([tensor, torch.flip(tensor[:, :, :, -h_pad:, :], [3])], dim=3)
        if w_pad:
            tensor = torch.cat([tensor, torch.flip(tensor[:, :, :, :, -w_pad:], [4])], dim=4)
        with _Heartbeat(heartbeat):
            with torch.inference_mode():
                result = model(inputs=tensor, mode="tensor").detach().cpu()[0, :, :, :height*scale, :width*scale]
        if result.ndim != 4 or result.shape[0] != chunk_size:
            raise RuntimeError("basicvsr_output_shape_mismatch")
        result = result.permute(0, 2, 3, 1).clamp(0, 1).numpy()
        return [np.clip(frame * 255.0 + 0.5, 0, 255).astype(np.uint8).tobytes() for frame in result]

    try:
        decoder = subprocess.Popen(decoder_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        encoder = subprocess.Popen(encoder_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        while True:
            raw = _read_exact(decoder.stdout, frame_bytes)
            if not raw:
                break
            frames.append(np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy())
            if len(frames) < chunk_size:
                continue
            output_frames = infer_window(frames[:chunk_size])
            first_output = 0 if emitted == 0 else overlap
            count = chunk_size if emitted == 0 else chunk_size - overlap
            for frame in output_frames[first_output:first_output + count]:
                encoder.stdin.write(frame)
            emitted += count
            if progress: progress(min(1.0, emitted / max(1, round(fps * duration))))
            frames = frames[chunk_size-overlap:]

        if frames:
            new_count = len(frames) if emitted == 0 else max(0, len(frames) - overlap)
            if new_count:
                missing = chunk_size - len(frames)
                padded = frames + (list(reversed(frames[-missing:])) if missing <= len(frames) else [frames[-1]] * missing)
                output_frames = infer_window(padded[:chunk_size])
                first_output = 0 if emitted == 0 else overlap
                for frame in output_frames[first_output:first_output + new_count]:
                    encoder.stdin.write(frame)
                emitted += new_count
                if progress: progress(min(1.0, emitted / max(1, round(fps * duration))))
        encoder.stdin.close()
        encoder.stdin = None
        _, decoder_stderr = decoder.communicate(timeout=120)
        _, encoder_stderr = encoder.communicate(timeout=900)
        if decoder.returncode != 0:
            raise RuntimeError("basicvsr_decoder_failed:" + decoder_stderr.decode("utf-8", "replace")[-500:])
        if encoder.returncode != 0:
            raise RuntimeError("basicvsr_encoder_failed:" + encoder_stderr.decode("utf-8", "replace")[-500:])
    finally:
        for process in (decoder, encoder):
            if process and process.poll() is None:
                process.kill()
                process.wait()
    if progress: progress(1.0)
    return out_width, out_height
