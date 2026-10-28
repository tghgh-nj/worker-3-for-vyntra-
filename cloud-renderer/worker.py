#!/usr/bin/env python3
"""Vyntra cloud render worker. Runs outside Supabase Edge Functions."""
from __future__ import annotations

import hashlib, hmac, importlib, json, os, shutil, subprocess, sys, tempfile, time, urllib.error, urllib.parse, urllib.request, uuid
import http.client
from pathlib import Path
import re
from fractions import Fraction

SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SERVICE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
WORKER_ID = os.getenv("VYNTRA_WORKER_ID", str(uuid.uuid4()))
BUCKET = os.getenv("VYNTRA_STORAGE_BUCKET", "vyntra-media")
POLL_SECONDS = max(2, int(os.getenv("VYNTRA_POLL_SECONDS", "5")))
MAX_DURATION = min(180, max(1, int(os.getenv("VYNTRA_MAX_DURATION_SECONDS", "180"))))
MAX_INPUT_BYTES = min(50 * 1024 * 1024, max(1, int(os.getenv("VYNTRA_MAX_INPUT_BYTES", str(50 * 1024 * 1024)))))
SOURCE_PART_BYTES = 48_000_000
MAX_SOURCE_FILE_BYTES = 200 * 1024 * 1024
MAX_PROJECT_SOURCE_BYTES = 200 * 1024 * 1024
OUTPUT_PART_BYTES = 48_000_000
MAX_OUTPUT_BYTES = max(OUTPUT_PART_BYTES, int(os.getenv("VYNTRA_MAX_OUTPUT_BYTES", "200000000")))
RELEASE_TTL_HOURS = min(168, max(1, int(os.getenv("VYNTRA_RELEASE_TTL_HOURS", "6"))))
RELEASE_REPOSITORY = os.getenv("VYNTRA_RELEASE_REPOSITORY", "fasmora-continue/here")
RELEASE_TOKEN = os.getenv("VYNTRA_RELEASE_TOKEN", "")
SUPPORTED_AI_MODELS = (
    "espcn", "edsr", "swinir", "realesrgan", "omnisr", "omnisr2", "bsrgan", "hat", "realhat",
    "basicvsr", "vrt",
)
AI_PHOTO_ONLY_MODELS = {"hat", "realhat"}
AI_TEMPORAL_MODELS = {"basicvsr", "vrt"}
AI_FRAME_MODEL_MAX_SOURCE = (1920, 1080)
AI_TEMPORAL_MAX_SOURCE = (854, 480)
AI_IMAGE_MAX_PIXELS = 8_000_000
AI_OUTPUT_PIXEL_LIMIT = 33_177_600
AI_MODEL_SCALES = {
    "espcn": 3,
    "omnisr2": 2,
    "edsr": 4,
    "swinir": 4,
    "realesrgan": 4,
    "omnisr": 4,
    "bsrgan": 4,
    "hat": 4,
    "realhat": 4,
    "basicvsr": 4,
    "vrt": 4,
}
AI_REQUIREMENTS_BY_MODEL = {
    "edsr": "requirements-ai-image.txt",
    "swinir": "requirements-ai-image.txt",
    "realesrgan": "requirements-ai-image.txt",
    "omnisr": "requirements-ai-image.txt",
    "omnisr2": "requirements-ai-image.txt",
    "bsrgan": "requirements-ai-image.txt",
    "hat": "requirements-ai-image.txt",
    "realhat": "requirements-ai-image.txt",
    "vrt": "requirements-vrt.txt",
    "basicvsr": "requirements-temporal.txt",
}
AI_BUILD_CONSTRAINTS = Path(__file__).with_name("requirements-ai-build-constraints.txt")
ONESHOT = os.getenv("VYNTRA_ONESHOT", "false").lower() == "true"
ENABLE_8K = os.getenv("VYNTRA_ENABLE_8K", "false").lower() == "true"
MAX_JOBS = min(100, max(1, int(os.getenv("VYNTRA_MAX_CONCURRENT_JOBS", "100"))))
MAX_JOBS_PER_REPOSITORY = min(20, max(1, int(os.getenv("VYNTRA_MAX_JOBS_PER_REPOSITORY", "20"))))

def safe_ffmpeg_diagnostic(stderr: str, work_dir: Path | None = None) -> str:
    """Return a bounded FFmpeg diagnostic with secrets, URLs, and temp paths removed."""
    text = stderr.replace("\x00", "")
    if work_dir:
        for work_path in {str(work_dir), str(work_dir).replace("\\", "/")}:
            text = text.replace(work_path, "[render-workdir]")
    text = re.sub(r"https?://\S+", "[url-redacted]", text, flags=re.IGNORECASE)
    text = re.sub(r"(?i)\bbearer\s+[^\s,;]+", "Bearer [redacted]", text)
    text = re.sub(
        r"(?i)\b(authorization|bearer|apikey|service[_ -]?role|access[_ -]?token|refresh[_ -]?token|password|secret)\b\s*[:=]?\s*[^\s,;]+",
        r"\1=[redacted]", text,
    )
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # FFmpeg often ends with the unhelpful "Conversion failed!"; keep the
    # preceding filter/stream errors and omit that generic footer.
    while lines and lines[-1].lower() == "conversion failed!":
        lines.pop()
    return " | ".join(line[-600:] for line in lines[-8:])[-3600:]


def ffmpeg_error_code(diagnostic: str) -> str:
    """Fit a useful, already-sanitized summary into the existing 100-char DB field."""
    if not diagnostic:
        return "ffmpeg_failed"
    summary = diagnostic.split(" | ")[-1]
    summary = re.sub(r"\s+", " ", summary).strip()
    return ("ffmpeg_failed:" + summary)[:100]

def api(path: str, method="GET", payload=None, headers=None, raw=False):
    data = payload if raw else (json.dumps(payload).encode() if payload is not None else None)
    h = {"apikey": SERVICE_KEY, "Authorization": f"Bearer {SERVICE_KEY}"}
    if not raw: h["Content-Type"] = "application/json"
    if headers: h.update(headers)
    req = urllib.request.Request(SUPABASE_URL + path, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            body = response.read()
            return json.loads(body) if body and not raw else body
    except urllib.error.HTTPError as exc:
        # PostgREST's response body is the only reliable way to distinguish a
        # schema-cache/column mismatch from an auth or constraint failure.
        # Keep only its bounded, redacted diagnostic in runner logs.
        try:
            response_body = exc.read(4096).decode("utf-8", "replace")
        except Exception:
            response_body = ""
        diagnostic = safe_ffmpeg_diagnostic(response_body)
        exc.vyntra_diagnostic = diagnostic
        print("Supabase REST request failed", exc.code, diagnostic or "[no response detail]", flush=True)
        raise


def is_retryable_job_error(job, exc, raw_code: str) -> bool:
    """Jobs whose receipt deleted their remote inputs cannot be requeued safely."""
    if job.get("source_cleanup_completed_at"):
        return False
    http_status = getattr(getattr(exc, "response", None), "status_code", None)
    transient_http = http_status is None or http_status >= 500 or http_status == 429
    return (
        isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError))
        or (exc.__class__.__module__.startswith("requests.") and transient_http)
        or raw_code == "ffmpeg_failed"
    )

def update_job(job_id, **values):
    # Refresh the lease on progress updates so a live FFmpeg render is not marked dead.
    values.setdefault("updated_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    global STAGE_COLUMN_AVAILABLE
    path = "/rest/v1/vyntra_render_jobs?id=eq." + urllib.parse.quote(job_id)
    payload = dict(values)
    if STAGE_COLUMN_AVAILABLE is False:
        payload.pop("stage", None)
    try:
        api(path, "PATCH", payload, {"Prefer": "return=minimal"})
    except urllib.error.HTTPError as exc:
        diagnostic = getattr(exc, "vyntra_diagnostic", "")
        missing_stage_column = (
            "stage" in payload
            and exc.code == 400
            and "stage" in diagnostic.lower()
            and (
                ("column" in diagnostic.lower()
                 and ("does not exist" in diagnostic.lower() or "could not find" in diagnostic.lower()))
                or "pgrst204" in diagnostic.lower()
            )
        )
        if not missing_stage_column:
            raise
        # Older databases may have the job table but not the optional progress
        # stage migration yet. Status/progress updates must still work so a
        # render (including AI inference) is not discarded before processing.
        STAGE_COLUMN_AVAILABLE = False
        payload.pop("stage", None)
        print("Vyntra stage column is unavailable; continuing without detailed stage updates", flush=True)
        api(path, "PATCH", payload, {"Prefer": "return=minimal"})


def check_job_cancelled(job_id):
    path = "/rest/v1/vyntra_render_jobs?id=eq." + urllib.parse.quote(job_id) + "&select=status"
    rows = api(path)
    if rows and rows[0].get("status") == "cancel_requested":
        raise InterruptedError("cancelled")


def ai_job_progress_callback(job_id, base: int = 11, span: int = 38, stage: str = "ai_enhancement"):
    """Throttle tile/frame progress writes while preserving monotonic progress."""
    last_progress = [base - 1]
    last_update = [0.0]
    last_cancel_check = [time.monotonic()]

    def report(value):
        fraction = max(0.0, min(1.0, float(value)))
        progress = base + int(span * fraction)
        now = time.monotonic()
        if progress > last_progress[0] or fraction >= 1.0 or now - last_update[0] >= 10:
            update_job(job_id, stage=stage, progress=progress)
            last_progress[0] = max(last_progress[0], progress)
            last_update[0] = now
        if now - last_cancel_check[0] >= 5:
            check_job_cancelled(job_id)
            last_cancel_check[0] = now

    return report


def ai_percent_progress_callback(job_id, base: int = 10, span: int = 40, divisor: int = 100):
    normalized = ai_job_progress_callback(job_id, base=base, span=span)
    return lambda percentage: normalized(float(percentage) / divisor)


STAGE_COLUMN_AVAILABLE: bool | None = None


COMPLETION_ERROR_CODES = {
    "method_not_allowed", "server_not_configured", "worker_unauthorized", "invalid_json",
    "output_manifest_invalid", "completion_unauthorized", "job_not_found",
    "job_already_completed", "job_not_rendering", "source_receipt_required",
    "release_repository_not_private", "release_not_found", "release_asset_verification_failed",
    "output_media_type_mismatch",
    "release_validation_unavailable", "job_completion_record_failed", "job_completion_conflict",
}
COMPLETION_MANIFEST_FIELDS = {
    "job_id", "asset_id", "release_id", "bytes", "sha256", "provider",
    "repository", "release_tag", "asset_name", "content_type", "output_path",
}


def safe_completion_http_error(exc: urllib.error.HTTPError) -> str:
    """Keep a known endpoint error code without logging its raw response body."""
    try:
        response = json.loads(exc.read(4096).decode("utf-8", errors="replace"))
        code = response.get("error") if isinstance(response, dict) else None
    except Exception:
        code = None
    if not isinstance(code, str) or code not in COMPLETION_ERROR_CODES:
        code = "unrecognized_response"
    elif code == "output_manifest_invalid":
        fields = response.get("fields", [])
        if isinstance(fields, list):
            safe_fields = sorted({field for field in fields if isinstance(field, str) and field in COMPLETION_MANIFEST_FIELDS})
            if safe_fields:
                code += ":" + ",".join(safe_fields)
    return f"release_completion_http_{exc.code}:{code}"


def report_render_completed(job, output_path: str, metadata: dict):
    """Let Supabase validate the private release asset before exposing completion."""
    job_id = str(uuid.UUID(str(job["id"])))
    asset_id = int(metadata.get("assetId", 0))
    output_bytes = int(metadata.get("bytes", 0))
    digest = str(metadata.get("sha256", "")).lower()
    if asset_id < 1 or output_bytes < 1 or not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("release_output_manifest_invalid")
    signature_payload = f"vyntra-output-complete:v1:{job_id}:{asset_id}:{output_bytes}:{digest}".encode()
    signature = hmac.new(SERVICE_KEY.encode(), signature_payload, hashlib.sha256).hexdigest()
    payload = json.dumps({"jobId": job_id, "outputPath": output_path, "outputMetadata": metadata}).encode()
    request = urllib.request.Request(
        SUPABASE_URL + "/functions/v1/vyntra-render-complete",
        data=payload,
        headers={
            "apikey": SERVICE_KEY,
            "Authorization": f"Bearer {SERVICE_KEY}",
            "x-vyntra-completion-signature": signature,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                result = json.loads(response.read())
                if result.get("completed") is not True:
                    raise RuntimeError("release_completion_rejected")
                job.update(status="completed", progress=100, output_path=output_path, output_metadata=metadata)
                return
        except urllib.error.HTTPError as exc:
            completion_error = safe_completion_http_error(exc)
            if exc.code < 500 and exc.code != 429:
                raise RuntimeError(completion_error) from None
            if attempt == 2:
                raise RuntimeError(completion_error) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise RuntimeError("release_completion_delivery_failed") from None
        if attempt < 2:
            time.sleep(1 + attempt)
    try:
        current = api(
            "/rest/v1/vyntra_render_jobs?id=eq." + urllib.parse.quote(job_id) +
            "&select=status,output_path,output_metadata"
        )
        if current and current[0].get("status") == "completed" and current[0].get("output_path") == output_path:
            job.update(status="completed", progress=100, output_path=output_path, output_metadata=metadata)
            return
    except Exception:
        pass

def claim_render_job(target_job_id=None):
    if target_job_id:
        try:
            target_job_id = str(uuid.UUID(target_job_id))
        except (ValueError, TypeError, AttributeError):
            raise ValueError("invalid_dispatched_job_id")
    claimed = api("/rest/v1/rpc/vyntra_claim_render_job", "POST", {
        "p_worker_id": WORKER_ID,
        "p_target_job_id": target_job_id,
    })
    return claimed[0] if isinstance(claimed, list) and claimed else None

def storage_download(object_path: str, destination: Path):
    if not object_path or object_path.startswith("/") or ".." in Path(object_path).parts:
        raise ValueError("invalid_media_path")
    storage_url, storage_key = source_storage_config(object_path)
    url = storage_url + "/storage/v1/object/" + BUCKET + "/" + urllib.parse.quote(object_path, safe="/")
    req = urllib.request.Request(url, headers={"apikey": storage_key, "Authorization": f"Bearer {storage_key}"})
    total = 0
    with urllib.request.urlopen(req, timeout=120) as response, destination.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk: break
            total += len(chunk)
            if total > MAX_INPUT_BYTES: raise ValueError("input_too_large")
            out.write(chunk)


def source_storage_tier(object_path: str) -> str:
    """Read the trusted storage tier encoded in newly uploaded source paths.

    Old jobs have paths like <owner>/sources/<uuid>-<name> and remain on the
    primary project. New jobs use <owner>/sources/{secondary|fallback}/... .
    """
    parts = object_path.split("/")
    if len(parts) >= 4 and parts[1] == "sources" and parts[2] in ("secondary", "tertiary", "fallback"):
        return parts[2]
    return "primary"


def source_storage_config(object_path: str) -> tuple[str, str]:
    """Resolve the Supabase project for a source object without trusting URLs."""
    tier = source_storage_tier(object_path)
    if tier == "secondary":
        url = (os.getenv("VYNTRA_SOURCE_SECONDARY_SUPABASE_URL") or "https://vfaakmxoeflzedwidlcl.supabase.co").rstrip("/")
        key = os.getenv("VYNTRA_SOURCE_SECONDARY_SERVICE_ROLE_KEY") or os.getenv("VYNTRA_OUTPUT_SUPABASE_SERVICE_ROLE_KEY", "")
    elif tier == "tertiary":
        url = (os.getenv("VYNTRA_SOURCE_FALLBACK_SUPABASE_URL") or "https://vbtreilxmstadnyotodb.supabase.co").rstrip("/")
        key = os.getenv("VYNTRA_SOURCE_FALLBACK_SERVICE_ROLE_KEY", "")
    elif tier == "fallback":
        url = (os.getenv("VYNTRA_SOURCE_FINAL_FALLBACK_SUPABASE_URL") or "https://lrnugbbyrazzeculrqwe.supabase.co").rstrip("/")
        key = os.getenv("VYNTRA_SOURCE_FINAL_FALLBACK_SERVICE_ROLE_KEY", "")
    else:
        url, key = SUPABASE_URL, SERVICE_KEY
    if not url or not key:
        raise RuntimeError(f"source_storage_{tier}_not_configured")
    return url, key


def source_object_groups(object_paths: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for path in dict.fromkeys(object_paths):
        groups.setdefault(source_storage_tier(path), []).append(path)
    return groups


def source_storage_remove(object_paths: list[str]) -> None:
    """Delete verified source objects from their owning private project."""
    for tier, paths in source_object_groups(object_paths).items():
        representative = paths[0]
        storage_url, storage_key = source_storage_config(representative)
        payload = json.dumps({"prefixes": paths}).encode()
        request = urllib.request.Request(
            storage_url + "/storage/v1/object/" + BUCKET,
            data=payload,
            headers={"apikey": storage_key, "Authorization": f"Bearer {storage_key}", "Content-Type": "application/json"},
            method="DELETE",
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            response.read()

def _verify_source_file(path: Path, object_path: str, expected_bytes: int | None, expected_sha256: str | None, manifest: dict, verified_parts: dict):
    expected = manifest.get(object_path)
    if expected is not None:
        if expected_bytes is not None and expected.get("bytes") != expected_bytes:
            raise ValueError("source_part_manifest_mismatch")
        expected_bytes = expected.get("bytes")
        manifest_sha = expected.get("sha256")
        if expected_sha256 and manifest_sha and expected_sha256 != manifest_sha:
            raise ValueError("source_part_manifest_mismatch")
        expected_sha256 = manifest_sha or expected_sha256
    actual_bytes = path.stat().st_size
    if expected_bytes is not None and actual_bytes != expected_bytes:
        raise ValueError("source_part_size_mismatch")
    if expected_sha256:
        if not re.fullmatch(r"[a-f0-9]{64}", expected_sha256):
            raise ValueError("source_checksum_invalid")
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError("source_checksum_mismatch")
    else:
        actual_sha256 = _sha256_file(path)
    verified_parts[object_path] = {"path": object_path, "bytes": actual_bytes, "sha256": actual_sha256}


def download_source_asset(object_path: str, parts, owner_id: str, destination: Path, manifest=None, verified_parts=None):
    """Download a normal source object or reconstruct a byte-split upload."""
    manifest = manifest if isinstance(manifest, dict) else {}
    verified_parts = verified_parts if isinstance(verified_parts, dict) else {}
    if not isinstance(object_path, str) or not object_path.startswith(owner_id + "/"):
        raise ValueError("media_not_owned")
    if parts is None:
        storage_download(object_path, destination)
        if destination.stat().st_size > MAX_SOURCE_FILE_BYTES:
            raise ValueError("source_file_too_large")
        _verify_source_file(destination, object_path, None, None, manifest, verified_parts)
        return
    if not isinstance(parts, list) or not 2 <= len(parts) <= 6:
        raise ValueError("invalid_source_parts")
    if any(not isinstance(part, dict) for part in parts):
        raise ValueError("invalid_source_parts")
    if parts[0].get("path") != object_path:
        raise ValueError("invalid_source_parts")
    total = 0
    destination.unlink(missing_ok=True)
    try:
        for index, part in enumerate(parts):
            if not isinstance(part, dict):
                raise ValueError("invalid_source_parts")
            part_path = part.get("path")
            expected_bytes = part.get("bytes")
            if (not isinstance(part_path, str) or not part_path.startswith(owner_id + "/") or
                    not isinstance(expected_bytes, int) or expected_bytes < 1 or expected_bytes > SOURCE_PART_BYTES):
                raise ValueError("invalid_source_parts")
            total += expected_bytes
            if total > MAX_SOURCE_FILE_BYTES:
                raise ValueError("source_file_too_large")
            temp_part = destination.with_name(destination.name + f".part{index:03d}")
            storage_download(part_path, temp_part)
            _verify_source_file(temp_part, part_path, expected_bytes, part.get("sha256"), manifest, verified_parts)
            with temp_part.open("rb") as incoming, destination.open("ab") as assembled:
                shutil.copyfileobj(incoming, assembled, 1024 * 1024)
            temp_part.unlink(missing_ok=True)
        if destination.stat().st_size != total:
            raise ValueError("source_reassembly_size_mismatch")
    except Exception:
        destination.unlink(missing_ok=True)
        for index in range(len(parts)):
            destination.with_name(destination.name + f".part{index:03d}").unlink(missing_ok=True)
        raise

def validate_source_manifest(plan, owner_id: str):
    manifest = plan.get("uploadedSourceParts")
    if manifest is None:
        return
    if not isinstance(manifest, list) or len(manifest) > 128:
        raise ValueError("invalid_source_parts")
    sizes = {}
    for part in manifest:
        if not isinstance(part, dict):
            raise ValueError("invalid_source_parts")
        path, size = part.get("path"), part.get("bytes")
        checksum = part.get("sha256")
        if (not isinstance(path, str) or not path.startswith(owner_id + "/") or
                not isinstance(size, int) or size < 1 or size > SOURCE_PART_BYTES):
            raise ValueError("invalid_source_parts")
        if checksum is not None and (not isinstance(checksum, str) or not re.fullmatch(r"[a-f0-9]{64}", checksum)):
            raise ValueError("source_checksum_invalid")
        item = {"bytes": size, "sha256": checksum}
        if path in sizes and sizes[path] != item:
            raise ValueError("source_part_manifest_mismatch")
        sizes[path] = item
    if sum(item["bytes"] for item in sizes.values()) > MAX_PROJECT_SOURCE_BYTES:
        raise ValueError("project_source_size_limit")
    return sizes


def acknowledge_source_receipt(job, expected_manifest: dict, verified_parts: dict):
    """Ask the control plane to delete inputs only after every part is hash-verified."""
    expected_paths = set(expected_manifest)
    if not expected_paths or set(verified_parts) != expected_paths:
        raise ValueError("source_receipt_manifest_incomplete")
    if any(not expected_manifest[path].get("sha256") for path in expected_paths):
        return False  # Legacy jobs keep source objects until the rendered output is safely stored.
    expected = [
        {"path": path, "bytes": expected_manifest[path]["bytes"], "sha256": expected_manifest[path]["sha256"]}
        for path in sorted(expected_paths)
    ]
    actual = [verified_parts[path] for path in sorted(expected_paths)]
    if actual != expected:
        raise ValueError("source_receipt_manifest_mismatch")
    job_paths = set(job.get("source_object_paths") or [])
    if job_paths != expected_paths:
        raise ValueError("source_receipt_path_mismatch")
    manifest_digest = hashlib.sha256(json.dumps(expected, separators=(",", ":")).encode()).hexdigest()
    signature_payload = f"vyntra-source-receipt:v1:{job['id']}:{manifest_digest}".encode()
    signature = hmac.new(SERVICE_KEY.encode(), signature_payload, hashlib.sha256).hexdigest()
    payload = json.dumps({"jobId": job["id"], "receivedParts": actual, "receiptSha256": manifest_digest}).encode()
    request = urllib.request.Request(
        SUPABASE_URL + "/functions/v1/vyntra-render-receipt",
        data=payload,
        headers={
            "apikey": SERVICE_KEY,
            "Authorization": f"Bearer {SERVICE_KEY}",
            "x-vyntra-receipt-signature": signature,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                result = json.loads(response.read())
                if result.get("acknowledged") is not True:
                    raise RuntimeError("source_receipt_rejected")
                job["source_object_paths"] = []
                job["source_cleanup_completed_at"] = result.get("cleanedAt")
                return True
        except urllib.error.HTTPError as exc:
            if exc.code < 500 and exc.code != 429:
                raise RuntimeError(f"source_receipt_http_{exc.code}") from None
            if attempt == 2:
                raise RuntimeError(f"source_receipt_http_{exc.code}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise RuntimeError("source_receipt_delivery_failed") from None
        if attempt < 2:
            time.sleep(1 + attempt)
    return False

def output_storage_config(job):
    """Return the storage endpoint for this job's rendered output."""
    snapshot = job.get("project_snapshot") if isinstance(job, dict) else None
    if isinstance(snapshot, dict) and snapshot.get("vyntraOutputStore") == "secondary":
        url = os.getenv("VYNTRA_OUTPUT_SUPABASE_URL", "").rstrip("/")
        key = os.getenv("VYNTRA_OUTPUT_SUPABASE_SERVICE_ROLE_KEY", "")
        if not url or not key:
            raise ValueError("output_storage_not_configured")
        return url, key, os.getenv("VYNTRA_OUTPUT_BUCKET", "vyntra-exports")
    return SUPABASE_URL, SERVICE_KEY, BUCKET


def storage_upload(source: Path, object_path: str, job=None):
    import requests
    # Stream the output from disk; do not load large rendered files into memory.
    storage_url, storage_key, storage_bucket = output_storage_config(job or {})
    url = storage_url + "/storage/v1/object/" + storage_bucket + "/" + urllib.parse.quote(object_path, safe="/")
    with source.open("rb") as stream:
        response = requests.post(url, data=stream, timeout=(30, 900), headers={
            "apikey": storage_key, "Authorization": f"Bearer {storage_key}",
            "Content-Type": "video/mp4", "x-upsert": "true",
        })
    if not response.ok:
        diagnostic = safe_ffmpeg_diagnostic(response.text)
        detail = diagnostic or response.reason or "no response details"
        error = requests.HTTPError(
            f"storage_upload_failed:http_{response.status_code}:{detail}", response=response
        )
        error.vyntra_storage_upload = True
        raise error


def is_storage_capacity_error(exc):
    """Only quota/object-cap failures should move an output to primary storage."""
    if not getattr(exc, "vyntra_storage_upload", False):
        return False
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    # Supabase reports object limits as 413 and project quota restrictions as 402.
    # 507 is the standard insufficient-storage response for compatible gateways.
    if status in (402, 413, 507):
        return True
    body = str(getattr(response, "text", "")).lower()
    return any(code in body for code in (
        "exceed_storage_size_quota", "exceeded_storage_quota",
        "storage_quota_exceeded", "insufficient_storage", "storage is full",
    ))


def storage_remove(object_paths: list[str], job=None):
    if not object_paths:
        return
    storage_url, storage_key, storage_bucket = output_storage_config(job or {})
    payload = json.dumps({"prefixes": object_paths}).encode()
    request = urllib.request.Request(
        storage_url + "/storage/v1/object/" + storage_bucket,
        data=payload,
        headers={"apikey": storage_key, "Authorization": f"Bearer {storage_key}", "Content-Type": "application/json"},
        method="DELETE",
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        response.read()


def cleanup_source_objects(job):
    """Free primary-bucket inputs after a completed output is safely recorded.

    This is deliberately best-effort: a cleanup outage must not turn a valid
    completed render into a failed job. The source paths remain in the row if
    deletion or the follow-up row update fails, allowing client acknowledgement
    to retry cleanup after delivery.
    """
    paths = job.get("source_object_paths")
    paths = list(dict.fromkeys(path for path in paths if isinstance(path, str) and path)) if isinstance(paths, list) else []
    if not paths:
        return False
    try:
        source_storage_remove(paths)
        update_job(job["id"], source_object_paths=[])
        print(f"removed {len(paths)} completed-job source objects", flush=True)
        return True
    except Exception as cleanup_error:
        print("completed-job source cleanup deferred", type(cleanup_error).__name__, flush=True)
        return False


def job_output_paths(job) -> list[str]:
    metadata = job.get("output_metadata")
    chunks = metadata.get("chunks", []) if isinstance(metadata, dict) else []
    paths = [chunk.get("path") for chunk in chunks if isinstance(chunk, dict)] if isinstance(chunks, list) else []
    if job.get("output_path"):
        paths.append(job["output_path"])
    return list(dict.fromkeys(path for path in paths if isinstance(path, str) and path))


def set_output_storage_target(job, target: str):
    """Persist the actual output route so signing and cleanup use the same project."""
    snapshot = job.get("project_snapshot") if isinstance(job, dict) else None
    snapshot = dict(snapshot) if isinstance(snapshot, dict) else {}
    snapshot["vyntraOutputStore"] = target
    job["project_snapshot"] = snapshot
    update_job(job["id"], project_snapshot=snapshot, output_path=None, output_metadata=None)


def _upload_output_parts_on_current_store(source: Path, job, metadata: dict, work: Path) -> tuple[str, dict]:
    total_bytes = source.stat().st_size
    if total_bytes <= 0:
        raise ValueError("empty_render_output")
    if total_bytes > MAX_OUTPUT_BYTES:
        raise ValueError("output_exceeds_free_storage_limit")
    if total_bytes <= OUTPUT_PART_BYTES:
        object_path = f'{job["owner_id"]}/exports/{job["id"]}.mp4'
        try:
            storage_upload(source, object_path, job)
            output_metadata = {**metadata, "bytes": total_bytes}
            update_job(job["id"], status="finalizing", progress=99,
                       output_path=object_path, output_metadata=output_metadata)
            current = api("/rest/v1/vyntra_render_jobs?id=eq." + urllib.parse.quote(job["id"]) + "&select=status")
            if current and current[0]["status"] == "cancel_requested":
                raise InterruptedError("cancelled")
            return object_path, output_metadata
        except Exception:
            try:
                storage_remove([object_path], job)
            except Exception as cleanup_error:
                print("partial output cleanup failed", type(cleanup_error).__name__, flush=True)
            try:
                update_job(job["id"], output_path=None, output_metadata=None)
            except Exception as update_error:
                print("partial output record cleanup failed", type(update_error).__name__, flush=True)
            raise
    part_count = (total_bytes + OUTPUT_PART_BYTES - 1) // OUTPUT_PART_BYTES
    uploaded = []
    attempted_paths = []
    try:
        with source.open("rb") as rendered:
            for index in range(part_count):
                current = api("/rest/v1/vyntra_render_jobs?id=eq." + urllib.parse.quote(job["id"]) + "&select=status")
                if current and current[0]["status"] == "cancel_requested":
                    raise InterruptedError("cancelled")
                data = rendered.read(OUTPUT_PART_BYTES)
                if not data:
                    raise RuntimeError("render_output_truncated")
                part_path = work / f"output_part_{index:03d}.mp4"
                part_path.write_bytes(data)
                object_path = f'{job["owner_id"]}/exports/{job["id"]}.part{index:03d}'
                attempted_paths.append(object_path)
                storage_upload(part_path, object_path, job)
                part_path.unlink(missing_ok=True)
                uploaded.append({"path": object_path, "bytes": len(data)})
                output_metadata = {**metadata, "bytes": total_bytes, "chunked": True, "chunks": uploaded.copy()}
                progress = min(99, 96 + int(3 * len(uploaded) / part_count))
                update_job(job["id"], status="finalizing", progress=progress,
                           output_path=uploaded[0]["path"], output_metadata=output_metadata)
        current = api("/rest/v1/vyntra_render_jobs?id=eq." + urllib.parse.quote(job["id"]) + "&select=status")
        if current and current[0]["status"] == "cancel_requested":
            raise InterruptedError("cancelled")
        return uploaded[0]["path"], {**metadata, "bytes": total_bytes, "chunked": True, "chunks": uploaded}
    except Exception:
        try:
            storage_remove(attempted_paths, job)
        except Exception as cleanup_error:
            print("partial output cleanup failed", type(cleanup_error).__name__, flush=True)
        try:
            update_job(job["id"], output_path=None, output_metadata=None)
        except Exception as update_error:
            print("partial output record cleanup failed", type(update_error).__name__, flush=True)
        raise


def upload_output_parts(source: Path, job, metadata: dict, work: Path) -> tuple[str, dict]:
    snapshot = job.get("project_snapshot") if isinstance(job, dict) else None
    storage_target = "secondary" if isinstance(snapshot, dict) and snapshot.get("vyntraOutputStore") == "secondary" else "primary"
    total_bytes = source.stat().st_size
    print(f"uploading {total_bytes} output bytes to {storage_target} storage", flush=True)
    try:
        return _upload_output_parts_on_current_store(source, job, metadata, work)
    except Exception as exc:
        if storage_target != "secondary" or not is_storage_capacity_error(exc):
            raise
        status = getattr(getattr(exc, "response", None), "status_code", "unknown")
        print(f"secondary output storage rejected an object for capacity/size (HTTP {status}); falling back to primary", flush=True)
        set_output_storage_target(job, "primary")
        print("retrying the complete output in primary storage", flush=True)
        return _upload_output_parts_on_current_store(source, job, metadata, work)


def _release_request(path: str, method="GET", payload=None):
    if not RELEASE_TOKEN:
        raise RuntimeError("github_release_not_configured")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", RELEASE_REPOSITORY):
        raise RuntimeError("github_release_repository_invalid")
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        "https://api.github.com/repos/" + RELEASE_REPOSITORY + path,
        data=body,
        headers={
            "Authorization": "Bearer " + RELEASE_TOKEN,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            **({"Content-Type": "application/json"} if body is not None else {}),
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise RuntimeError(f"github_release_http_{exc.code}") from None


def _release_upload(source: Path, upload_url: str, asset_name: str, content_type="video/mp4", heartbeat=None):
    parsed = urllib.parse.urlsplit(upload_url.split("{", 1)[0])
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    query.append(("name", asset_name))
    target = parsed.path + "?" + urllib.parse.urlencode(query)
    connection = http.client.HTTPSConnection(parsed.hostname, timeout=600)
    try:
        connection.putrequest("POST", target)
        connection.putheader("Authorization", "Bearer " + RELEASE_TOKEN)
        connection.putheader("Accept", "application/vnd.github+json")
        connection.putheader("X-GitHub-Api-Version", "2022-11-28")
        connection.putheader("Content-Type", content_type)
        connection.putheader("Content-Length", str(source.stat().st_size))
        connection.endheaders()
        last_heartbeat = time.monotonic()
        sent_bytes = 0
        total_bytes = source.stat().st_size
        with source.open("rb") as rendered:
            while chunk := rendered.read(1024 * 1024):
                connection.send(chunk)
                sent_bytes += len(chunk)
                if heartbeat and time.monotonic() - last_heartbeat >= 15:
                    heartbeat(sent_bytes, total_bytes)
                    last_heartbeat = time.monotonic()
        response = connection.getresponse()
        response_body = response.read()
        if response.status not in (201, 200):
            raise RuntimeError(f"github_release_upload_http_{response.status}")
        return json.loads(response_body)
    finally:
        connection.close()


def _sha256_file(source: Path) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as rendered:
        for chunk in iter(lambda: rendered.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upload_output_release(source: Path, job, metadata: dict, work: Path) -> tuple[str, dict]:
    """Store one verified, retry-safe MP4 or PNG asset in the private release repo."""
    total_bytes = source.stat().st_size
    if total_bytes <= 0:
        raise ValueError("empty_render_output")
    if total_bytes > MAX_OUTPUT_BYTES:
        raise ValueError("output_exceeds_free_storage_limit")
    if not RELEASE_TOKEN:
        raise RuntimeError("github_release_not_configured")
    content_type = metadata.get("contentType", "video/mp4")
    extension = {"video/mp4": "mp4", "image/png": "png"}.get(content_type)
    if extension is None:
        raise ValueError("unsupported_output_content_type")

    repository = _release_request("")
    if not isinstance(repository, dict) or repository.get("private") is not True:
        raise RuntimeError("github_release_repository_must_be_private")

    job_id = str(uuid.UUID(str(job["id"])))
    tag = "vyntra-render-" + job_id
    release = _release_request("/releases/tags/" + urllib.parse.quote(tag, safe=""))
    if release is None:
        release = _release_request("/releases", "POST", {
            "tag_name": tag,
            "name": "Vyntra render " + job_id,
            "body": "Temporary Vyntra cloud-render output. Deleted after verified delivery.",
            "draft": False,
            "prerelease": True,
            "make_latest": "false",
        })
    if not isinstance(release, dict) or not isinstance(release.get("id"), int):
        raise RuntimeError("github_release_create_failed")

    digest = _sha256_file(source)
    asset_name = f"vyntra-{job_id}-{digest[:16]}.{extension}"
    assets = _release_request(f"/releases/{release['id']}/assets") or []
    if not isinstance(assets, list):
        raise RuntimeError("github_release_assets_invalid")
    asset = next((item for item in assets if item.get("name") == asset_name and item.get("size") == total_bytes and item.get("state") == "uploaded"), None)
    if asset is None:
        for incomplete in assets:
            if incomplete.get("name") == asset_name:
                _release_request(f"/releases/assets/{incomplete['id']}", "DELETE")
        try:
            asset = _release_upload(
                source,
                release["upload_url"],
                asset_name,
                content_type=content_type,
                heartbeat=lambda sent, total: update_job(
                    job_id, stage="uploading_release",
                    progress=96 + int(2 * sent / max(total, 1)),
                ),
            )
        except Exception:
            leftovers = _release_request(f"/releases/{release['id']}/assets") or []
            for leftover in leftovers if isinstance(leftovers, list) else []:
                if str(leftover.get("name", "")).startswith(f"vyntra-{job_id}-"):
                    _release_request(f"/releases/assets/{leftover['id']}", "DELETE")
            remaining = _release_request(f"/releases/{release['id']}/assets") or []
            if not remaining:
                _release_request(f"/releases/{release['id']}", "DELETE")
                _release_request(f"/git/refs/tags/{urllib.parse.quote(tag, safe='')}", "DELETE")
            raise
    if not isinstance(asset, dict) or not isinstance(asset.get("id"), int) or asset.get("size") != total_bytes or asset.get("state") != "uploaded" or asset.get("content_type") != content_type:
        raise RuntimeError("github_release_asset_verification_failed")

    # A prior attempt may have uploaded a different output for this job. Keep the
    # previous asset until its replacement is fully uploaded and verified.
    for old_asset in assets:
        if old_asset.get("id") != asset["id"] and str(old_asset.get("name", "")).startswith(f"vyntra-{job_id}-"):
            _release_request(f"/releases/assets/{old_asset['id']}", "DELETE")

    output_metadata = {
        **metadata,
        "provider": "github_release",
        "repository": RELEASE_REPOSITORY,
        "releaseId": release["id"],
        "releaseTag": tag,
        "assetId": asset["id"],
        "assetName": asset_name,
        "contentType": content_type,
        "bytes": total_bytes,
        "sha256": digest,
    }
    output_path = f"github-release:{release['id']}:{asset['id']}"
    update_job(job_id, status="finalizing", stage="finalizing", progress=99, output_path=output_path, output_metadata=output_metadata)
    job["output_path"] = output_path
    job["output_metadata"] = output_metadata
    current = api("/rest/v1/vyntra_render_jobs?id=eq." + urllib.parse.quote(job_id) + "&select=status")
    if current and current[0]["status"] == "cancel_requested":
        _release_request(f"/releases/assets/{asset['id']}", "DELETE")
        _release_request(f"/releases/{release['id']}", "DELETE")
        _release_request(f"/git/refs/tags/{urllib.parse.quote(tag, safe='')}", "DELETE")
        raise InterruptedError("cancelled")
    return output_path, output_metadata


def cleanup_release_output(job):
    metadata = job.get("output_metadata") if isinstance(job, dict) else None
    if not isinstance(metadata, dict) or metadata.get("provider") != "github_release":
        return False
    if metadata.get("repository") != RELEASE_REPOSITORY:
        raise RuntimeError("github_release_metadata_invalid")
    release_id = int(metadata.get("releaseId", 0))
    asset_id = int(metadata.get("assetId", 0))
    tag = str(metadata.get("releaseTag", ""))
    if release_id < 1 or asset_id < 1 or not re.fullmatch(r"vyntra-render-[0-9a-f-]{36}", tag):
        raise RuntimeError("github_release_metadata_invalid")
    _release_request(f"/releases/assets/{asset_id}", "DELETE")
    assets = _release_request(f"/releases/{release_id}/assets") or []
    if not isinstance(assets, list):
        raise RuntimeError("github_release_assets_invalid")
    if not assets:
        _release_request(f"/releases/{release_id}", "DELETE")
        _release_request(f"/git/refs/tags/{urllib.parse.quote(tag, safe='')}", "DELETE")
    update_job(job["id"], output_path=None, output_metadata=None)
    return True


def cleanup_expired_release_outputs():
    cutoff = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ",
        time.gmtime(time.time() - RELEASE_TTL_HOURS * 3600),
    )
    query = urllib.parse.urlencode({
        "status": "in.(completed,failed)",
        "output_metadata->>provider": "eq.github_release",
        "completed_at": "lt." + cutoff,
        "select": "id,status,output_metadata",
        "limit": "100",
    })
    expired = api("/rest/v1/vyntra_render_jobs?" + query)
    for job in expired if isinstance(expired, list) else []:
        try:
            cleanup_release_output(job)
            if job.get("status") == "completed":
                update_job(job["id"], status="failed", progress=0, error_code="output_expired")
            print("removed expired cloud render release", job.get("id"), flush=True)
        except Exception as exc:
            print("expired release cleanup deferred", job.get("id"), type(exc).__name__, flush=True)

def probe(path: Path):
    raw = subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)], timeout=45)
    return json.loads(raw)

def safe_number(value, low, high, name):
    try: number = float(value)
    except (TypeError, ValueError): raise ValueError("invalid_" + name)
    if number < low or number > high: raise ValueError("invalid_" + name)
    return number

def clip_speed_segments(clip, source_ms):
    points = clip.get("speedCurve", [])
    if not isinstance(points, list) or len(points) > 12:
        raise ValueError("invalid_speed_curve")
    segments = []
    cursor = 0
    current_speed = safe_number(clip.get("speed", 1.0), .25, 4.0, "speed")
    previous_point = -1
    for point in points:
        if not isinstance(point, dict): raise ValueError("invalid_speed_curve_point")
        raw_at = safe_number(point.get("timeMs", -1), 0, max(0, source_ms - 1), "speed_curve_time")
        if not raw_at.is_integer(): raise ValueError("invalid_speed_curve_time")
        at = int(raw_at)
        next_speed = safe_number(point.get("speed", current_speed), .25, 4.0, "speed_curve_speed")
        if at <= previous_point: raise ValueError("invalid_speed_curve_order")
        if at > cursor: segments.append((cursor / 1000, at / 1000, current_speed))
        cursor = at
        current_speed = next_speed
        previous_point = at
    if cursor < source_ms: segments.append((cursor / 1000, source_ms / 1000, current_speed))
    return segments or [(0, source_ms / 1000, current_speed)]

def speed_curve_duration(segments):
    return sum((end - start) / speed for start, end, speed in segments)

def speed_curve_setpts(segments):
    """Map source-time into the cumulative piecewise-constant curve time."""
    elapsed = 0.0
    expression = ""
    for index, (start, end, speed) in enumerate(segments):
        value = f"({elapsed:.9f}+((PTS-STARTPTS)*TB-{start:.9f})/{speed:.9f})/TB"
        if index == len(segments) - 1:
            expression = value
        else:
            expression = f"if(lt((PTS-STARTPTS)*TB\\,{end:.9f})\\,{value}\\,{expression})"
        elapsed += (end - start) / speed
    return f"setpts='{expression}'"

def atempo_chain(speed):
    filters = []
    while speed > 2.0:
        filters.append("atempo=2.0"); speed /= 2.0
    while speed < .5:
        filters.append("atempo=0.5"); speed /= .5
    filters.append(f"atempo={speed:.6f}")
    return filters

def clip_trim_range(clip, duration_seconds):
    # The editor saves duration from the platform player, while FFprobe reads
    # the decoded container duration. Treat FFprobe as authoritative and clamp
    # a saved end to that duration instead of rejecting a valid trim because
    # those metadata sources differ.
    actual_duration_ms = duration_seconds * 1000
    declared_duration_ms = safe_number(
        clip.get("durationMs", int(actual_duration_ms)), 1, MAX_DURATION * 1000, "clip_duration"
    )
    start_ms = safe_number(clip.get("startMs", 0), 0, actual_duration_ms, "trim_start")
    requested_end_ms = safe_number(clip.get("endMs", int(actual_duration_ms)), 1, declared_duration_ms, "trim_end")
    end_ms = min(requested_end_ms, actual_duration_ms)
    start, end = start_ms / 1000, end_ms / 1000
    if end <= start:
        raise ValueError("invalid_trim_range")
    return start, end

def output_dimensions(source_width: int, source_height: int, rotation: int, aspect_ratio: str, resolution: str):
    edge = {"480p": 480, "540p": 540, "720p": 720, "1080p": 1080, "1440p": 1440, "1620p": 1620, "2160p": 2160, "2880p": 2880, "3456p": 3456, "4320p": 4320}.get(resolution)
    if edge is None: raise ValueError("unsupported_resolution")
    ratios = {"9:16": 9 / 16, "16:9": 16 / 9, "1:1": 1.0, "4:5": 4 / 5, "3:4": 3 / 4, "4:3": 4 / 3, "21:9": 21 / 9}
    if aspect_ratio == "Original":
        if rotation % 180 == 90: source_width, source_height = source_height, source_width
        target_ratio = source_width / max(1, source_height)
    elif aspect_ratio in ratios:
        target_ratio = ratios[aspect_ratio]
    else:
        raise ValueError("invalid_aspect_ratio")
    if target_ratio >= 1:
        width, height = round(edge * target_ratio / 2) * 2, edge
    else:
        width, height = edge, round(edge / target_ratio / 2) * 2
    if width * height > 7680 * 4320: raise ValueError("output_dimensions_too_large")
    return width, height


def ensure_ai_model_runtime(
    ai_model: str,
    runtime_root: Path | None = None,
    heartbeat=None,
) -> None:
    """Install heavyweight CPU inference packages only for jobs that need them."""
    requirement_name = AI_REQUIREMENTS_BY_MODEL.get(ai_model)
    if requirement_name is None:
        if ai_model != "espcn":
            raise ValueError("ai_model_unavailable")
        return
    requirements = Path(__file__).with_name(requirement_name)
    if not requirements.is_file():
        raise RuntimeError("ai_runtime_requirements_missing")
    if not AI_BUILD_CONSTRAINTS.is_file():
        raise RuntimeError("ai_runtime_build_constraints_missing")
    runtime_dir = (runtime_root or Path(tempfile.gettempdir()) / "vyntra-ai-runtime") / requirements.stem
    requirements_hash = hashlib.sha256(
        requirements.read_bytes() + AI_BUILD_CONSTRAINTS.read_bytes()
    ).hexdigest()[:16]
    marker = runtime_dir / f".requirements-installed-{requirements_hash}"
    if runtime_dir.exists() and not marker.is_file():
        shutil.rmtree(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    if not marker.is_file():
        command = [
            sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
            "--no-cache-dir", "--build-constraint", str(AI_BUILD_CONSTRAINTS),
            "--target", str(runtime_dir), "-r", str(requirements),
        ]
        install_log_path = runtime_dir / ".install.log"
        try:
            with install_log_path.open("w+b") as install_log:
                process = subprocess.Popen(command, stdout=install_log, stderr=subprocess.STDOUT)
                started_at = time.monotonic()
                while process.poll() is None:
                    if time.monotonic() - started_at >= 1200:
                        process.kill()
                        process.wait()
                        raise TimeoutError("ai_runtime_install_timeout")
                    if heartbeat:
                        heartbeat()
                    time.sleep(10)
                if process.returncode != 0:
                    install_log.flush()
                    install_log.seek(max(0, install_log.tell() - 400))
                    detail = install_log.read().decode("utf-8", errors="replace")[-400:]
                    raise RuntimeError("ai_runtime_install_failed:" + detail)
        except Exception as exc:
            shutil.rmtree(runtime_dir, ignore_errors=True)
            raise
        install_log_path.unlink(missing_ok=True)
        marker.touch()
    runtime_path = str(runtime_dir)
    if runtime_path not in sys.path:
        sys.path.insert(0, runtime_path)
    importlib.invalidate_caches()


def append_clip_input(input_args: list[str], path: Path, start: float, end: float, is_image: bool) -> None:
    """Append the selected (possibly AI-processed) clip as a real FFmpeg input."""
    if is_image:
        input_args.extend(["-loop", "1", "-t", str(end - start), "-i", str(path)])
    else:
        input_args.extend(["-ss", str(start), "-t", str(end - start), "-i", str(path)])


def resolve_ai_selection(plan: dict, clips: list, resolution: str) -> tuple[str, str]:
    legacy_flag = plan.get("aiEnhancement", False)
    if not isinstance(legacy_flag, bool):
        raise ValueError("invalid_ai_enhancement_flag")
    mode = plan.get("enhancementMode")
    if mode is None:
        mode = "ai" if legacy_flag else "normal"
    model = plan.get("aiModel", "espcn")
    if mode not in ("normal", "ai", "both"):
        raise ValueError("unsupported_enhancement_mode")
    if mode != "normal" and model not in SUPPORTED_AI_MODELS:
        raise ValueError("ai_model_unavailable")
    supported_resolutions = {"480p", "540p", "720p", "1080p", "1440p", "1620p", "2160p", "2880p", "3456p"}
    if ENABLE_8K:
        supported_resolutions.add("4320p")
    if resolution not in supported_resolutions:
        raise ValueError("ai_resolution_unavailable")
    if mode != "normal" and len(clips) != 1:
        raise ValueError("ai_requires_single_clip")
    if mode != "normal" and clips[0].get("mediaType", "video") == "image" and model in AI_TEMPORAL_MODELS:
        raise ValueError("ai_temporal_model_requires_video")
    if mode != "normal" and clips[0].get("mediaType", "video") == "video" and model in AI_PHOTO_ONLY_MODELS:
        raise ValueError("ai_model_requires_photo")
    return mode, model


def validate_ai_source_dimensions(model: str, width: int, height: int) -> None:
    """Use a higher source ceiling for tiled frame models than temporal models."""
    max_long, max_short = AI_TEMPORAL_MAX_SOURCE if model in AI_TEMPORAL_MODELS else AI_FRAME_MODEL_MAX_SOURCE
    if width < 1 or height < 1 or max(width, height) > max_long or min(width, height) > max_short:
        limit = "480p" if model in AI_TEMPORAL_MODELS else "1080p"
        raise ValueError(f"{model}_source_must_be_{limit}_or_lower")
    if width * height * AI_MODEL_SCALES[model] ** 2 > AI_OUTPUT_PIXEL_LIMIT:
        raise ValueError("ai_video_output_pixel_limit")


def validate_ai_photo_dimensions(model: str, width: int, height: int) -> None:
    """Use still-image limits, independent of the video decoder limits."""
    if width < 1 or height < 1:
        raise ValueError("ai_image_dimensions_invalid")
    if model in AI_TEMPORAL_MODELS:
        raise ValueError("ai_temporal_model_requires_video")
    if model == "espcn" and (max(width, height) > 854 or min(width, height) > 480):
        raise ValueError("espcn_photo_source_must_be_480p_or_lower")
    if model != "espcn" and width * height > AI_IMAGE_MAX_PIXELS:
        raise ValueError("ai_image_pixel_limit")
    if width * height * AI_MODEL_SCALES[model] ** 2 > AI_OUTPUT_PIXEL_LIMIT:
        raise ValueError("ai_image_output_pixel_limit")


def resolve_clip_ai_model(clip: dict, media_type: str) -> str:
    """Validate an editor clip's model and its image/video compatibility."""
    model = clip.get("aiUpscaleModel", "none")
    if model not in ("none", *SUPPORTED_AI_MODELS):
        raise ValueError("ai_model_unavailable")
    if model in AI_PHOTO_ONLY_MODELS and media_type != "image":
        raise ValueError("ai_model_requires_photo")
    if model in AI_TEMPORAL_MODELS and media_type != "video":
        raise ValueError("ai_temporal_model_requires_video")
    return model


def visual_effect_filters(effect: str, width: int, height: int) -> list[str]:
    """Map editor effect IDs to FFmpeg filters; reject unknown IDs explicitly."""
    filters = {
        "none": [],
        "flash": ["eq=brightness=0.10:enable='lt(mod(t,0.4),0.08)'"],
        "shake": [f"crop=iw-24:ih-24:x=12+8*sin(2*PI*12*t):y=12+8*cos(2*PI*10*t),scale={width}:{height}"],
        "blur": ["boxblur=4:1"],
        "glitch": ["rgbashift=rh=8:bh=-8:enable='lt(mod(t,0.5),0.08)'"],
        "phonk": ["eq=contrast=1.25:saturation=1.35"],
        "grayscale": ["hue=s=0"],
        "negative": ["negate"],
        "soft": ["gblur=sigma=1.5"],
        "cinematic": ["eq=brightness=0.015:contrast=1.13:saturation=1.08"],
        "moody": ["eq=brightness=-0.06:contrast=1.20:saturation=0.97"],
        "vivid": ["eq=brightness=0.025:contrast=1.20:saturation=1.20"],
        "dramatic": ["eq=contrast=1.30:saturation=1.10"],
        "vintage": ["colorbalance=rs=0.08:gs=0.02:bs=-0.05", "eq=saturation=0.85:contrast=0.92"],
        "fade": ["eq=contrast=0.84:saturation=0.94:brightness=0.025"],
        "night": ["eq=brightness=-0.10:contrast=1.08:saturation=0.90", "colorbalance=bs=0.08"],
        "sunset": ["colorbalance=rs=0.08:gs=0.02:bs=-0.08", "eq=saturation=1.13"],
        "noir": ["hue=s=0", "eq=contrast=1.25:brightness=-0.02"],
        "dreamy": ["gblur=sigma=2", "eq=brightness=0.03:saturation=1.10"],
    }
    if effect not in filters:
        raise ValueError("unsupported_effect")
    return filters[effect]


def render_still_image(job, source: Path, work: Path, model: str, enhancement_mode: str):
    """Render a Status photo as PNG, preserving still-image delivery semantics."""
    from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

    plan = job["project_snapshot"]
    clips = plan["clips"]
    if len(clips) != 1 or clips[0].get("mediaType") != "image":
        raise ValueError("still_image_export_requires_one_photo")
    if plan.get("musicPath") or plan.get("imageOverlays"):
        raise ValueError("still_image_export_has_unsupported_overlays")
    clip = clips[0]
    normalized_source = work / "status_photo_normalized.png"
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image.save(normalized_source, format="PNG")

    width, height = image.size
    clip_model = resolve_clip_ai_model(clip, "image")
    active_model = clip_model if clip_model != "none" else model if enhancement_mode != "normal" else "none"
    generated_dimensions = None
    models_used = []
    if active_model != "none":
        validate_ai_photo_dimensions(active_model, width, height)
        update_job(job["id"], stage="ai_enhancement", progress=10)
        if active_model == "espcn":
            enhanced = work / "status_photo_espcn.png"
            from espcn import upscale_image
            upscale_image(
                normalized_source, enhanced, work / "espcn_x3.tflite",
                heartbeat=lambda: update_job(job["id"], stage="ai_enhancement"),
                progress=ai_job_progress_callback(job["id"], base=10, span=40),
            )
            generated_dimensions = (width * 3, height * 3)
        else:
            update_job(job["id"], stage="loading_ai_model", progress=10)
            ensure_ai_model_runtime(active_model, work / "ai-runtime", heartbeat=lambda: update_job(
                job["id"], stage="loading_ai_model",
            ))
            update_job(job["id"], stage="ai_enhancement", progress=11)
            from ai_image_models import upscale_image
            enhanced = work / f"status_photo_{active_model}.png"
            generated_dimensions = upscale_image(
                normalized_source, enhanced, active_model, work / "model-cache",
                heartbeat=lambda: update_job(job["id"], stage="ai_enhancement"),
                progress=ai_job_progress_callback(job["id"]),
            )
        with Image.open(enhanced) as processed:
            image = processed.convert("RGB")
        models_used.append(active_model)

    rotation = int(clip.get("rotation", 0)) % 360
    if rotation not in (0, 90, 180, 270):
        raise ValueError("unsupported_rotation")
    if rotation:
        image = image.rotate(-rotation, expand=True)

    aspect = plan.get("aspectRatio", "Original")
    ratios = {"9:16": 9/16, "16:9": 16/9, "1:1": 1.0, "4:5": 4/5, "3:4": 3/4, "4:3": 4/3, "21:9": 21/9}
    if aspect != "Original" and aspect not in ratios:
        raise ValueError("invalid_aspect_ratio")
    target_ratio = image.width / image.height if aspect == "Original" else ratios[aspect]
    if image.width / image.height > target_ratio:
        crop_width = max(1, round(image.height * target_ratio))
        left = (image.width - crop_width) // 2
        image = image.crop((left, 0, left + crop_width, image.height))
    elif image.width / image.height < target_ratio:
        crop_height = max(1, round(image.width / target_ratio))
        top = (image.height - crop_height) // 2
        image = image.crop((0, top, image.width, top + crop_height))

    output_width, output_height = output_dimensions(image.width, image.height, 0, aspect, job["resolution"])
    if output_width * output_height > AI_OUTPUT_PIXEL_LIMIT:
        raise ValueError("ai_image_output_pixel_limit")
    image = image.resize((output_width, output_height), Image.Resampling.LANCZOS)
    image = ImageEnhance.Brightness(image).enhance(1 + safe_number(clip.get("brightness", 0), -.25, .25, "brightness"))
    image = ImageEnhance.Contrast(image).enhance(1 + safe_number(clip.get("contrast", 0), -.3, .3, "contrast"))
    image = ImageEnhance.Color(image).enhance(1 + safe_number(clip.get("saturation", 0), -20, 30, "saturation") / 100)
    sharpness = safe_number(clip.get("sharpness", 0), 0, .5, "sharpness")
    if sharpness:
        image = ImageEnhance.Sharpness(image).enhance(1 + sharpness * 2)

    overlays = plan.get("overlays", [])
    if not isinstance(overlays, list) or len(overlays) > 16:
        raise ValueError("invalid_text_overlay")
    if overlays:
        image = image.convert("RGBA")
        draw = ImageDraw.Draw(image)
        for overlay in overlays:
            text = str(overlay.get("text", ""))
            start = safe_number(overlay.get("startMs", 0), 0, MAX_DURATION * 1000, "overlay_start")
            end = safe_number(overlay.get("endMs", 1000), start + 1, MAX_DURATION * 1000, "overlay_end")
            if not text or len(text) > 160 or not start <= 0 < end:
                continue
            font_size = int(safe_number(overlay.get("fontSize", 34), 16, 320, "font_size"))
            x = safe_number(overlay.get("x", 0), -1, 1, "overlay_x")
            y = safe_number(overlay.get("y", .55), -1, 1, "overlay_y")
            argb = int(overlay.get("color", 0xFFFFFFFF))
            color = (argb >> 16 & 255, argb >> 8 & 255, argb & 255)
            alpha = (argb >> 24 & 255) if argb > 0xFFFFFF else 255
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except OSError:
                font = ImageFont.load_default()
            draw.text(
                (output_width * (0.5 + x / 2), output_height * (0.5 + y / 2)), text,
                anchor="mm", font=font, fill=(*color, alpha), stroke_width=max(2, round(font_size * .035)),
                stroke_fill=(0, 0, 0, 210),
            )
        image = image.convert("RGB")

    out = work / "render.png"
    image.save(out, format="PNG", optimize=True)
    with Image.open(out) as verified:
        verified.verify()
    update_job(job["id"], stage="verifying_output", progress=95)
    metadata = {
        "width": output_width, "height": output_height, "imageFormat": "png",
        "contentType": "image/png", "bytes": out.stat().st_size,
        "enhancementMode": "ai" if models_used and enhancement_mode == "normal" else enhancement_mode,
        "aiModel": models_used[0] if models_used else None, "aiModels": models_used,
        "aiGeneratedWidth": generated_dimensions[0] if generated_dimensions else None,
        "aiGeneratedHeight": generated_dimensions[1] if generated_dimensions else None,
    }
    return out, metadata


def render(job, work: Path):
    plan = job["project_snapshot"]
    source_manifest = validate_source_manifest(plan, job["owner_id"])
    clips = plan.get("clips")
    if not isinstance(clips, list) or not 1 <= len(clips) <= 24:
        raise ValueError("unsupported_project_clip_count")
    resolution = job["resolution"]
    enhancement_mode, ai_model = resolve_ai_selection(plan, clips, resolution)
    print(
        "Vyntra render enhancement selection",
        json.dumps({
            "job_id": str(job.get("id", "")),
            "mode": enhancement_mode,
            "model": ai_model if enhancement_mode != "normal" else None,
            "resolution": resolution,
        }, separators=(",", ":")),
        flush=True,
    )
    edge = {"480p": 480, "540p": 540, "720p": 720, "1080p": 1080, "1440p": 1440, "1620p": 1620, "2160p": 2160, "2880p": 2880, "3456p": 3456, "4320p": 4320}.get(resolution)
    if edge is None: raise ValueError("unsupported_resolution")
    if resolution in ("2880p", "3456p", "4320p"):
        try:
            total_ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError, AttributeError):
            total_ram = 0
        free_disk = shutil.disk_usage(tempfile.gettempdir()).free
        if resolution == "2880p" and (total_ram < 4 * 1024**3 or free_disk < 6 * 1024**3):
            raise ValueError("5k_worker_resources_insufficient")
        if resolution == "3456p" and (total_ram < 6 * 1024**3 or free_disk < 8 * 1024**3):
            raise ValueError("6k_worker_resources_insufficient")
        if resolution == "4320p":
            if not ENABLE_8K: raise ValueError("8k_not_enabled_on_worker")
            if total_ram < 8 * 1024**3 or free_disk < 10 * 1024**3:
                raise ValueError("8k_worker_resources_insufficient")
    codec = job.get("codec", "h264")
    encoders = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], stderr=subprocess.STDOUT, text=True)
    if codec == "h265" and "libx265" not in encoders: raise ValueError("encoder_unavailable:h265")
    if codec not in ("h264", "h265"): raise ValueError("unsupported_codec")

    # Download and verify every source before acknowledging receipt. Rendering,
    # AI enhancement, and FFmpeg work begin only after the control plane has
    # accepted the receipt and deleted the now-unneeded Storage objects.
    source_dir = work / "verified_sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    verified_parts = {}
    update_job(job["id"], stage="downloading_sources", progress=1)
    source_total_bytes = sum(item["bytes"] for item in (source_manifest or {}).values())

    def report_source_progress():
        if source_total_bytes <= 0:
            return
        verified_bytes = sum(item["bytes"] for item in verified_parts.values())
        progress = 1 + int(7 * min(verified_bytes, source_total_bytes) / source_total_bytes)
        update_job(job["id"], stage="downloading_sources", progress=min(8, progress))

    clip_sources = {}
    for index, clip in enumerate(clips):
        key = clip.get("path", "")
        if not key.startswith(job["owner_id"] + "/"):
            raise ValueError("media_not_owned")
        path = source_dir / f"clip_{index}.source"
        download_source_asset(key, clip.get("sourceParts"), job["owner_id"], path, source_manifest, verified_parts)
        clip_sources[index] = path
        report_source_progress()
    music_key = plan.get("musicPath")
    music_source = None
    if music_key:
        if not isinstance(music_key, str) or not music_key.startswith(job["owner_id"] + "/"):
            raise ValueError("media_not_owned")
        music_source = source_dir / "music.source"
        download_source_asset(music_key, plan.get("musicSourceParts"), job["owner_id"], music_source, source_manifest, verified_parts)
        report_source_progress()
    image_sources = {}
    for index, overlay in enumerate(plan.get("imageOverlays", [])):
        key = overlay.get("path", "")
        if not isinstance(key, str) or not key.startswith(job["owner_id"] + "/"):
            raise ValueError("media_not_owned")
        suffix = Path(key).suffix.lower()
        if suffix not in (".png", ".jpg", ".jpeg", ".webp"):
            raise ValueError("unsupported_image_format")
        image_path = source_dir / f"image_overlay_{index}{suffix}"
        download_source_asset(key, overlay.get("sourceParts"), job["owner_id"], image_path, source_manifest, verified_parts)
        image_sources[index] = image_path
        report_source_progress()

    update_job(job["id"], stage="verifying_sources", progress=8)
    if source_manifest is not None:
        acknowledge_source_receipt(job, source_manifest, verified_parts)
    source_complete_stage = "source_received" if source_manifest is not None else "sources_downloaded"
    update_job(job["id"], stage=source_complete_stage, progress=9)

    if plan.get("outputMediaType") == "image/png":
        if len(clips) != 1 or clips[0].get("mediaType") != "image":
            raise ValueError("still_image_export_requires_one_photo")
        return render_still_image(job, clip_sources[0], work, ai_model, enhancement_mode)

    local = []
    clip_lengths = []
    trim_ranges = []
    input_args = ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-progress", "pipe:1", "-loglevel", "error"]
    probe_info = []
    ai_generated_dimensions = None
    ai_models_used = []
    for i, clip in enumerate(clips):
        key = clip.get("path", "")
        if not key.startswith(job["owner_id"] + "/"):
            raise ValueError("media_not_owned")
        path = clip_sources[i]
        meta = probe(path)
        v = next((s for s in meta.get("streams", []) if s.get("codec_type") == "video"), None)
        if not v: raise ValueError("source_has_no_video")
        media_type = clip.get("mediaType", "video")
        if media_type not in ("video", "image"): raise ValueError("unsupported_media_type")
        duration = float(meta.get("format", {}).get("duration", 0))
        if media_type == "image":
            if Path(key).suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"): raise ValueError("unsupported_image_format")
            duration = safe_number(clip.get("durationMs", 0), 1, MAX_DURATION * 1000, "image_duration") / 1000
        if duration <= 0 or duration > MAX_DURATION:
            raise ValueError("source_duration_limit")
        start, end = clip_trim_range(clip, duration)
        if end <= start or end - start > MAX_DURATION: raise ValueError("invalid_trim_range")
        speed = safe_number(clip.get("speed", 1.0), .25, 4.0, "speed")
        clip_ai_model = resolve_clip_ai_model(clip, media_type)
        active_ai_model = clip_ai_model if clip_ai_model != "none" else ai_model
        if enhancement_mode != "normal" or clip_ai_model != "none":
            if active_ai_model not in ai_models_used:
                ai_models_used.append(active_ai_model)
            update_job(job["id"], stage="ai_enhancement", progress=10)
            source_width, source_height = int(v.get("width", 0)), int(v.get("height", 0))
            if media_type == "image":
                validate_ai_photo_dimensions(active_ai_model, source_width, source_height)
            else:
                validate_ai_source_dimensions(active_ai_model, source_width, source_height)
            if duration <= 0 or duration > MAX_DURATION:
                raise ValueError("ai_clip_limit_180_seconds")
            if media_type == "image":
                enhanced_path = work / f"clip_{i}_{active_ai_model}_enhanced.png"
                if active_ai_model == "espcn":
                    from espcn import upscale_image
                    upscale_image(
                        path, enhanced_path, work / "espcn_x3.tflite",
                        heartbeat=lambda jid=job["id"]: update_job(jid, stage="ai_enhancement"),
                        progress=ai_job_progress_callback(job["id"], base=10, span=40),
                    )
                    ai_generated_dimensions = (source_width * 3, source_height * 3)
                elif active_ai_model in AI_TEMPORAL_MODELS:
                    raise ValueError("ai_temporal_model_requires_video")
                else:
                    update_job(job["id"], stage="loading_ai_model", progress=10)
                    ensure_ai_model_runtime(active_ai_model, work / "ai-runtime", heartbeat=lambda: update_job(
                        job["id"], stage="loading_ai_model",
                    ))
                    update_job(job["id"], stage="ai_enhancement", progress=11)
                    from ai_image_models import upscale_image
                    ai_generated_dimensions = upscale_image(
                        path, enhanced_path, active_ai_model, work / "model-cache",
                        heartbeat=lambda jid=job["id"]: update_job(jid, stage="ai_enhancement"),
                        progress=ai_job_progress_callback(job["id"]),
                    )
            else:
                try:
                    source_fps = float(Fraction(str(v.get("avg_frame_rate") or v.get("r_frame_rate") or "24/1")))
                except (ValueError, ZeroDivisionError):
                    source_fps = 24.0
                if active_ai_model == "espcn":
                    from espcn import validate_espcn_source, upscale_video
                    validate_espcn_source(source_width, source_height, source_fps, duration)
                    enhanced_path = work / f"clip_{i}_espcn.mp4"
                    upscale_video(
                        path, enhanced_path, source_width, source_height, source_fps, duration,
                        work / "espcn_x3.tflite",
                        progress=ai_percent_progress_callback(job["id"], divisor=10),
                        heartbeat=lambda jid=job["id"]: update_job(jid, stage="ai_enhancement"),
                    )
                    ai_generated_dimensions = (source_width * 3, source_height * 3)
                else:
                    update_job(job["id"], stage="loading_ai_model", progress=10)
                    ensure_ai_model_runtime(active_ai_model, work / "ai-runtime", heartbeat=lambda: update_job(
                        job["id"], stage="loading_ai_model",
                    ))
                    update_job(job["id"], stage="ai_enhancement", progress=11)
                    if active_ai_model == "vrt":
                        from temporal_models import upscale_video
                        enhanced_path = work / "vrt_enhanced.mp4"
                        ai_generated_dimensions = upscale_video(
                            path, enhanced_path, source_width, source_height, source_fps, duration,
                            work / "model-cache" / "vrt",
                            progress=ai_job_progress_callback(job["id"]),
                            heartbeat=lambda jid=job["id"]: update_job(
                                jid, stage="ai_enhancement",
                            ),
                        )
                    elif active_ai_model == "basicvsr":
                        from temporal_models import upscale_basicvsr_video
                        enhanced_path = work / "basicvsr_enhanced.mp4"
                        ai_generated_dimensions = upscale_basicvsr_video(
                            path, enhanced_path, source_width, source_height, source_fps, duration,
                            work / "model-cache" / "basicvsr",
                            progress=ai_job_progress_callback(job["id"]),
                            heartbeat=lambda jid=job["id"]: update_job(
                                jid, stage="ai_enhancement",
                            ),
                        )
                    else:
                        from ai_image_models import upscale_video
                        enhanced_path = work / f"{active_ai_model}_enhanced.mp4"
                        ai_generated_dimensions = upscale_video(
                            path, enhanced_path, active_ai_model, source_width, source_height, source_fps, duration,
                            work / "model-cache",
                            progress=ai_job_progress_callback(job["id"]),
                            heartbeat=lambda jid=job["id"]: update_job(
                                jid, stage="ai_enhancement",
                            ),
                        )
            path = enhanced_path
            meta = probe(path)
            v = next((s for s in meta.get("streams", []) if s.get("codec_type") == "video"), None)
            if not v:
                raise RuntimeError("ai_output_has_no_video")
        reverse = clip.get("reverse", False)
        if not isinstance(reverse, bool): raise ValueError("invalid_reverse")
        speed_segments = clip_speed_segments(clip, (end-start)*1000)
        if reverse and clip.get("speedCurve"):
            raise ValueError("reverse_with_speed_curve_unsupported")
        append_clip_input(input_args, path, start, end, media_type == "image")
        freeze_ms = safe_number(clip.get("freezeFrameMs", 0), 0, 10000, "freeze_frame")
        local.append(path); clip_lengths.append(speed_curve_duration(speed_segments) + freeze_ms/1000); probe_info.append((v, media_type == "video" and any(s.get("codec_type") == "audio" for s in meta.get("streams", [])))); trim_ranges.append((start, end))
    if sum(clip_lengths) > MAX_DURATION: raise ValueError("project_duration_limit")
    music_path = music_source
    if music_path:
        input_args += ["-i", str(music_path)]

    first = probe_info[0][0]
    width, height = output_dimensions(
        int(first.get("width", 16)), int(first.get("height", 9)),
        int(clips[0].get("rotation", 0)), plan.get("aspectRatio", "Original"), resolution,
    )

    filters, video_labels, audio_labels = [], [], []
    for i, (clip, (v, has_audio), (trim_start, trim_end)) in enumerate(zip(clips, probe_info, trim_ranges)):
        rot = int(clip.get("rotation", 0)) % 360
        if rot not in (0, 90, 180, 270): raise ValueError("unsupported_rotation")
        vf = []
        if rot == 90: vf.append("transpose=clock")
        elif rot == 180: vf.extend(["hflip", "vflip"])
        elif rot == 270: vf.append("transpose=cclock")
        speed = safe_number(clip.get("speed", 1.0), .25, 4.0, "speed")
        reverse = clip.get("reverse", False)
        volume = safe_number(clip.get("volume", 1), 0, 1, "volume")
        effect = clip.get("effect", "none")
        effect_vf = visual_effect_filters(effect, width, height)
        if reverse: vf.append("reverse")
        frame_rate = int(safe_number(plan.get("exportFrameRate", 30), 24, 60, "frame_rate"))
        if frame_rate not in (24, 25, 30, 50, 60): raise ValueError("unsupported_frame_rate")
        vf += [f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos", f"crop={width}:{height}", "setsar=1", f"fps={frame_rate}"]
        vf += effect_vf
        speed_segments = clip_speed_segments(clip, (trim_end-trim_start)*1000)
        if clip.get("speedCurve"):
            vf.append(speed_curve_setpts(speed_segments))
        elif speed != 1.0: vf.append(f"setpts=(PTS-STARTPTS)/{speed}")
        freeze_ms = safe_number(clip.get("freezeFrameMs", 0), 0, 10000, "freeze_frame")
        if freeze_ms: vf.append(f"tpad=stop_mode=clone:stop_duration={freeze_ms/1000:.6f}")
        brightness = safe_number(clip.get("brightness", 0), -.25, .25, "brightness")
        contrast = 1 + safe_number(clip.get("contrast", 0), -.3, .3, "contrast")
        saturation = 1 + safe_number(clip.get("saturation", 0), -20, 30, "saturation") / 100
        sharpness = safe_number(clip.get("sharpness", 0), 0, .5, "sharpness")
        keyframes = clip.get("keyframes", [])
        if not isinstance(keyframes, list) or len(keyframes) > 100:
            raise ValueError("invalid_keyframes")
        normalized = []
        for keyframe in keyframes:
            if not isinstance(keyframe, dict):
                raise ValueError("invalid_keyframe")
            time_ms = safe_number(keyframe.get("timeMs", 0), 0, clip_lengths[i] - safe_number(clip.get("freezeFrameMs", 0), 0, 10000, "freeze_frame")/1000, "keyframe_time")
            kb = safe_number(keyframe.get("brightness", brightness), -.25, .25, "keyframe_brightness")
            kc = 1 + safe_number(keyframe.get("contrast", contrast-1), -.3, .3, "keyframe_contrast")
            ks = 1 + safe_number(keyframe.get("saturation", (saturation-1)*100), -20, 30, "keyframe_saturation") / 100
            kv = safe_number(keyframe.get("volume", volume), 0, 1, "keyframe_volume")
            kx = safe_number(keyframe.get("x", clip.get("positionX", 0)), -1, 1, "keyframe_x")
            ky = safe_number(keyframe.get("y", clip.get("positionY", 0)), -1, 1, "keyframe_y")
            kz = safe_number(keyframe.get("scale", clip.get("scale", 1)), .25, 4, "keyframe_scale")
            kr = safe_number(keyframe.get("rotation", 0), -360, 360, "keyframe_rotation")
            ko = safe_number(keyframe.get("opacity", clip.get("opacity", 1)), 0, 1, "keyframe_opacity")
            normalized.append((time_ms/1000, kb, kc, ks, kv, kx, ky, kz, kr, ko))
        normalized.sort(key=lambda item: item[0])
        def keyframe_expr(column, fallback, variable="t", points=None):
            frames = normalized if points is None else points
            value = f"{frames[-1][column] if frames else fallback:.6f}"
            for left, right in reversed(list(zip(frames, frames[1:]))):
                t0, t1 = left[0], right[0]
                v0, v1 = left[column], right[column]
                if t1 > t0:
                    value = f"if(lt({variable}\\,{t1:.6f})\\,{v0:.6f}+({v1-v0:.6f})*({variable}-{t0:.6f})/{t1-t0:.6f}\\,{value})"
            if frames:
                value = f"if(lt({variable}\\,{frames[0][0]:.6f})\\,{frames[0][column]:.6f}\\,{value})"
            return value
        if normalized:
            vf.append(f"eq=eval=frame:brightness='{keyframe_expr(1, brightness)}':contrast='{keyframe_expr(2, contrast)}':saturation='{keyframe_expr(3, saturation)}'")
        elif brightness or contrast != 1 or saturation != 1:
            vf.append(f"eq=brightness={brightness}:contrast={contrast}:saturation={saturation}")
        if sharpness > 0:
            vf.append(f"unsharp=5:5:{sharpness:.3f}:5:5:0")
        mask = clip.get("mask", "none")
        if mask not in ("none", "circle", "rectangle"): raise ValueError("unsupported_mask")
        feather = safe_number(clip.get("maskFeather", 0), 0, .5, "mask_feather")
        key_enabled = clip.get("chromaKeyEnabled", False)
        if not isinstance(key_enabled, bool): raise ValueError("invalid_chroma_key")
        if key_enabled:
            color = int(safe_number(clip.get("chromaKeyColor", 0xFF00FF00), 0, 0xFFFFFFFF, "chroma_key_color")) & 0xFFFFFF
            similarity = safe_number(clip.get("chromaSimilarity", .25), 0, 1, "chroma_similarity")
            blend = safe_number(clip.get("chromaBlend", .08), 0, 1, "chroma_blend")
            # colorkey is the RGB variant and accepts RGBA frames; chromakey is
            # a YUV filter and can reject the editor's alpha-capable stream.
            vf.append(f"format=rgba,colorkey=0x{color:06X}:{similarity:.4f}:{blend:.4f}")
        transform_rotation = safe_number(clip.get("transformRotation", 0), -360, 360, "transform_rotation")
        raw_transform_keyframes = clip.get("transformKeyframes", [])
        if not isinstance(raw_transform_keyframes, list) or len(raw_transform_keyframes) > 100:
            raise ValueError("invalid_transform_keyframes")
        transform_normalized = []
        for point in raw_transform_keyframes:
            if not isinstance(point, dict): raise ValueError("invalid_transform_keyframe")
            at = safe_number(point.get("timeMs", 0), 0, max(0, clip_lengths[i] * 1000 - 1), "transform_keyframe_time") / 1000
            tx = safe_number(point.get("x", clip.get("positionX", 0)), -1, 1, "transform_keyframe_x")
            ty = safe_number(point.get("y", clip.get("positionY", 0)), -1, 1, "transform_keyframe_y")
            ts = safe_number(point.get("scale", clip.get("scale", 1)), .5, 2.5, "transform_keyframe_scale")
            tr = safe_number(point.get("rotation", clip.get("transformRotation", 0)), -360, 360, "transform_keyframe_rotation")
            to = safe_number(point.get("opacity", clip.get("opacity", 1)), 0, 1, "transform_keyframe_opacity")
            transform_normalized.append((at, 0, 1, 1, 1, tx, ty, ts, tr, to))
        transform_normalized.sort(key=lambda item: item[0])
        transform_points = bool(transform_normalized) or any(abs(float(clip.get(name, default))-default) > .0001 for name, default in (("positionX",0), ("positionY",0), ("scale",1), ("opacity",1))) or abs(transform_rotation) > .0001
        if transform_points:
            fps_for_transform = int(safe_number(plan.get("exportFrameRate", 30), 24, 60, "frame_rate"))
            x_expr = keyframe_expr(5, safe_number(clip.get("positionX", 0), -1, 1, "position_x"), f"on/{fps_for_transform}", transform_normalized)
            y_expr = keyframe_expr(6, safe_number(clip.get("positionY", 0), -1, 1, "position_y"), f"on/{fps_for_transform}", transform_normalized)
            zoom_expr = keyframe_expr(7, safe_number(clip.get("scale", 1), .5, 2.5, "scale"), f"on/{fps_for_transform}", transform_normalized)
            angle_expr = keyframe_expr(8, transform_rotation, "t", transform_normalized)
            opacity_expr = keyframe_expr(9, safe_number(clip.get("opacity", 1), 0, 1, "opacity"), "T", transform_normalized)
            # A 4x transparent overscan canvas lets zoompan represent 0.5x to
            # 2.5x scale while staying inside zoompan's supported 1x..10x range.
            zoom_expr = f"max(1,min(10,({zoom_expr})*4))"
            vf += ["format=rgba,pad=iw*4:ih*4:iw*3/2:ih*3/2:color=black@0", f"rotate=a='({angle_expr})*PI/180':ow=iw:oh=ih:c=black@0", f"zoompan=z='{zoom_expr}':x='clip(iw/2-iw/zoom/2-({x_expr})*iw/(2*zoom),0,iw-iw/zoom)':y='clip(ih/2-ih/zoom/2-({y_expr})*ih/(2*zoom),0,ih-ih/zoom)':d=1:s={width}x{height}:fps={fps_for_transform}", f"format=rgba,geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='alpha(X,Y)*({opacity_expr})'"]
        elif mask != "none":
            vf.append("format=rgba")
        if mask != "none":
            shape = "pow((X-W/2)/(min(W,H)/2),2)+pow((Y-H/2)/(min(W,H)/2),2)" if mask == "circle" else "max(abs((X-W/2)/(W*0.42)),abs((Y-H/2)/(H*0.42)))"
            if feather:
                alpha_expr = f"255*clip((1-({shape}))*min(W,H)/({feather*min(width,height):.6f}),0,1)" if mask == "circle" else f"255*clip((1-({shape}))*min(W,H)/({feather*min(width,height):.6f}),0,1)"
            else:
                alpha_expr = f"if(lte({shape},1),255,0)"
            if key_enabled or transform_points:
                alpha_expr = f"alpha(X,Y)*({alpha_expr})/255"
            vf.append(f"geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='{alpha_expr}'")
        # Some effects (notably shake's crop/scale chain) can adjust SAR after
        # the initial canvas normalization. Reassert square pixels immediately
        # before concat/xfade so mixed clips have matching stream parameters.
        has_alpha = mask != "none" or key_enabled or transform_points
        if has_alpha:
            background = int(safe_number(clip.get("chromaBackgroundColor", 0xFF000000), 0, 0xFFFFFFFF, "chroma_background_color")) & 0xFFFFFF
            filters.append(f"[{i}:v]{','.join(vf)},setsar=1,setpts=PTS-STARTPTS[vraw{i}]")
            filters.append(f"color=c=0x{background:06X}:s={width}x{height}:r={frame_rate}:d={clip_lengths[i]:.6f}[vbg{i}]")
            filters.append(f"[vbg{i}][vraw{i}]overlay=shortest=1:format=auto,format=yuv420p[v{i}]")
        else:
            filters.append(f"[{i}:v]{','.join(vf)},setsar=1,format=yuv420p,setpts=PTS-STARTPTS[v{i}]")
        video_labels.append(f"[v{i}]")
        if has_audio:
            af = "areverse," if reverse else ""
            if clip.get("speedCurve"):
                # Split the trimmed source audio at each curve point, retime each
                # segment by its own factor, and concatenate in source order.
                if reverse: raise ValueError("reverse_with_speed_curve_unsupported")
                parts = []
                for part_index, (part_start, part_end, part_speed) in enumerate(speed_segments):
                    label = f"[as{i}_{part_index}]"
                    part_filters = [f"atrim=start={part_start:.6f}:end={part_end:.6f}", "asetpts=PTS-STARTPTS", *atempo_chain(part_speed)]
                    filters.append(f"[{i}:a]{','.join(part_filters)}{label}")
                    parts.append(label)
                filters.append("".join(parts) + f"concat=n={len(parts)}:v=0:a=1[aspeed{i}]")
                af = ""
            elif speed != 1.0:
                af += ",".join(atempo_chain(speed)) + ","
            audio_fx = []
            if clip.get("audioDenoise") is True: audio_fx.append("afftdn=nf=-25")
            highpass = int(safe_number(clip.get("audioHighPassHz", 0), 0, 300, "audio_highpass"))
            if highpass: audio_fx.append(f"highpass=f={highpass}")
            if clip.get("audioCompressor") is True: audio_fx.append("acompressor=threshold=0.125:ratio=3:attack=20:release=250")
            if clip.get("audioNormalize") is True: audio_fx.append("loudnorm=I=-16:TP=-1.5:LRA=11")
            if normalized:
                audio_fx.append(f"volume='{keyframe_expr(4, volume)}':eval=frame")
            else:
                audio_fx.append(f"volume={volume}")
            if clip.get("speedCurve"):
                if audio_fx:
                    filters.append(f"[aspeed{i}]{','.join(audio_fx)}[abase{i}]")
                else:
                    filters.append(f"[aspeed{i}]anull[abase{i}]")
            else:
                filters.append(f"[{i}:a]{af}asetpts=PTS-STARTPTS,{','.join(audio_fx)}[abase{i}]")
            fade_in = safe_number(clip.get("audioFadeInMs", 0), 0, 10000, "audio_fade_in") / 1000
            fade_out = safe_number(clip.get("audioFadeOutMs", 0), 0, 10000, "audio_fade_out") / 1000
            audio_filters = []
            if fade_in: audio_filters.append(f"afade=t=in:st=0:d={min(fade_in, clip_lengths[i]):.6f}")
            if fade_out: audio_filters.append(f"afade=t=out:st={max(0, clip_lengths[i]-fade_out):.6f}:d={min(fade_out, clip_lengths[i]):.6f}")
            filters.append(f"[abase{i}]{','.join(audio_filters)}[a{i}]" if audio_filters else f"[abase{i}]anull[a{i}]")
        else:
            filters.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={clip_lengths[i]:.6f},asetpts=PTS-STARTPTS[a{i}]")
        audio_labels.append(f"[a{i}]")
    # Real transitions are rendered by FFmpeg, not simulated only in the editor preview.
    transition_map = {name: name for name in (
        "fade", "fadeblack", "fadewhite", "dissolve", "pixelize", "radial", "distance", "rectcrop",
        "wipeleft", "wiperight", "wipeup", "wipedown", "slideleft", "slideright", "slideup", "slidedown",
        "smoothleft", "smoothright", "smoothup", "smoothdown", "circleopen", "circleclose",
        "vertopen", "vertclose", "horzopen", "horzclose", "diagtl", "diagtr", "diagbl", "diagbr",
        "hlslice", "hrslice", "vuslice", "vdslice", "coverleft", "coverright", "coverup", "coverdown",
        "revealleft", "revealright", "revealup", "revealdown",
    )}
    active_transitions = []
    for i in range(1, len(clips)):
        kind = clips[i].get("transition", "cut")
        if kind != "cut" and kind not in transition_map:
            raise ValueError("unsupported_transition")
        duration = safe_number(clips[i].get("transitionMs", 0), 0, 2000, "transition_duration") / 1000
        if kind != "cut" and duration > 0:
            duration = min(duration, clip_lengths[i-1] * .45, clip_lengths[i] * .45)
        active_transitions.append((transition_map.get(kind, "fade"), duration if kind != "cut" else 0))
    if any(duration > 0 for _, duration in active_transitions):
        current = video_labels[0]
        elapsed = clip_lengths[0]
        for i, (kind, duration) in enumerate(active_transitions, start=1):
            next_label = f"[vjoin{i}]"
            if duration <= 0:
                filters.append(f"{current}[v{i}]concat=n=2:v=1:a=0{next_label}")
                elapsed += clip_lengths[i]
            else:
                filters.append(f"{current}[v{i}]xfade=transition={kind}:duration={duration:.6f}:offset={max(0, elapsed-duration):.6f}{next_label}")
                elapsed += clip_lengths[i] - duration
            current = next_label
        video_label = current
    else:
        filters.append("".join(video_labels) + f"concat=n={len(clips)}:v=1:a=0[vcat]")
        video_label = "[vcat]"
    if any(duration > 0 for _, duration in active_transitions):
        current_audio = audio_labels[0]
        for i, (_, duration) in enumerate(active_transitions, start=1):
            next_audio = f"[ajoin{i}]"
            if duration > 0:
                filters.append(f"{current_audio}{audio_labels[i]}acrossfade=d={duration:.6f}:c1=tri:c2=tri{next_audio}")
            else:
                filters.append(f"{current_audio}{audio_labels[i]}concat=n=2:v=0:a=1{next_audio}")
            current_audio = next_audio
        audio_label = current_audio
    else:
        filters.append("".join(audio_labels) + f"concat=n={len(clips)}:v=0:a=1[acat]")
        audio_label = "[acat]"
    if music_path:
        music_index = len(clips)
        music_volume = safe_number(plan.get("musicVolume", .35), 0, 1, "music_volume")
        offset = int(safe_number(plan.get("musicOffsetMs", 0), 0, MAX_DURATION * 1000, "music_offset"))
        filters.append(f"[{music_index}:a]asetpts=PTS-STARTPTS,volume={music_volume},adelay={offset}|{offset}[music]")
        filters.append(f"{audio_label}[music]amix=inputs=2:duration=first:dropout_transition=2[aout]")
        audio_label = "[aout]"

    # Add user-scoped still-image overlays as looped inputs. The base video controls output duration.
    image_inputs = []
    for image_index, overlay in enumerate(plan.get("imageOverlays", [])):
        key = overlay.get("path", "")
        if not key.startswith(job["owner_id"] + "/"): raise ValueError("media_not_owned")
        suffix = Path(key).suffix.lower()
        if suffix not in (".png", ".jpg", ".jpeg", ".webp"): raise ValueError("unsupported_image_format")
        image_path = image_sources[image_index]
        input_args += ["-loop", "1", "-i", str(image_path)]
        image_inputs.append((len(clips) + (1 if music_path else 0) + image_index, overlay))

    # Timed text and image overlays are supported; unknown effects fail explicitly.
    for i, overlay in enumerate(plan.get("overlays", [])):
        text = str(overlay.get("text", ""))
        if not text or len(text) > 160: raise ValueError("invalid_text_overlay")
        # Use a text file rather than interpolating user text into a filter expression.
        text_file = work / f"text_{i}.txt"
        text_file.write_text(text, encoding="utf-8")
        start = safe_number(overlay.get("startMs", 0), 0, MAX_DURATION*1000, "overlay_start") / 1000
        end = safe_number(overlay.get("endMs", 1000), start*1000+1, MAX_DURATION*1000, "overlay_end") / 1000
        font_size = int(safe_number(overlay.get("fontSize", 34), 16, 160, "font_size"))
        x = safe_number(overlay.get("x", 0), -1, 1, "overlay_x")
        y = safe_number(overlay.get("y", .55), -1, 1, "overlay_y")
        argb = int(overlay.get("color", 0xFFFFFFFF))
        color = argb & 0xFFFFFF
        alpha = ((argb >> 24) & 0xFF) / 255 if argb > 0xFFFFFF else 1.0
        next_label = f"[vt{i}]"
        border_width = max(2, round(font_size * .035))
        filters.append(f"{video_label}drawtext=textfile='{text_file}':fontsize={font_size}:fontcolor=0x{color:06X}@{alpha:.3f}:borderw={border_width}:bordercolor=black@0.82:x=(w-text_w)/2+({x})*w/2:y=(h-text_h)/2+({y})*h/2:enable='between(t,{start},{end})'{next_label}")
        video_label = next_label

    for i, (image_index, overlay) in enumerate(image_inputs):
        start = safe_number(overlay.get("startMs", 0), 0, MAX_DURATION * 1000, "image_start") / 1000
        end = safe_number(overlay.get("endMs", 1000), start * 1000 + 1, MAX_DURATION * 1000, "image_end") / 1000
        scale = safe_number(overlay.get("scale", .28), .05, 1, "image_scale")
        x = safe_number(overlay.get("x", .65), -1, 1, "image_x")
        y = safe_number(overlay.get("y", .55), -1, 1, "image_y")
        next_label = f"[vi{i}]"
        filters.append(f"[{image_index}:v]scale={max(2, int(width * scale))}:-1,format=rgba[img{i}]")
        filters.append(f"{video_label}[img{i}]overlay=x=(W-w)/2+({x})*W/2:y=(H-h)/2+({y})*H/2:enable='between(t,{start},{end})':eof_action=pass{next_label}")
        video_label = next_label
    out = work / "render.mp4"
    bitrate = plan.get("exportBitrateKbps")
    encoder_args = ["-c:v", "libx264" if codec == "h264" else "libx265", "-preset", "veryfast"]
    if bitrate is None:
        encoder_args += ["-crf", str(int(safe_number(job.get("quality", 23), 18, 32, "quality")))]
    else:
        bitrate = int(safe_number(bitrate, 1000, 80000, "bitrate"))
        encoder_args += ["-b:v", f"{bitrate}k", "-maxrate", f"{int(bitrate * 1.25)}k", "-bufsize", f"{bitrate * 2}k"]
    cmd = input_args + ["-filter_complex", ";".join(filters), "-map", video_label, "-map", audio_label] + encoder_args + [
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)]
    encode_start_progress = 50 if enhancement_mode != "normal" else 10
    update_job(job["id"], status="processing", stage="encoding", progress=encode_start_progress, started_at=job.get("started_at") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    stderr_path = work / "ffmpeg.stderr"
    with stderr_path.open("w+b") as stderr_file:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=stderr_file, text=True, bufsize=1)
        duration_ms = int(sum(clip_lengths) * 1000)
        while proc.poll() is None:
            line = proc.stdout.readline() if proc.stdout else ""
            if line.startswith("out_time_ms="):
                try:
                    done = int(line.split("=", 1)[1].strip()) // 1000
                    progress = min(94, max(encode_start_progress, int(encode_start_progress + done * (94 - encode_start_progress) / max(1, duration_ms))))
                    update_job(job["id"], stage="encoding", progress=progress)
                except ValueError: pass
            # Worker restarts and cancellation are observed through the job record.
            current = api("/rest/v1/vyntra_render_jobs?id=eq." + urllib.parse.quote(job["id"]) + "&select=status")
            if current and current[0]["status"] == "cancel_requested":
                proc.terminate(); proc.wait(timeout=10); raise InterruptedError("cancelled")
            time.sleep(.1)
        stderr_file.flush()
        stderr_file.seek(0)
        stderr = stderr_file.read().decode("utf-8", errors="replace")
    if proc.returncode != 0:
        diagnostic = safe_ffmpeg_diagnostic(stderr, work)
        print("FFmpeg failed with exit code", proc.returncode, flush=True)
        print("FFmpeg diagnostic:", diagnostic or "[no diagnostic output]", flush=True)
        raise RuntimeError("ffmpeg_failed:" + diagnostic)
    update_job(job["id"], stage="verifying_output", progress=95)
    meta = probe(out)
    video = next((s for s in meta.get("streams", []) if s.get("codec_type") == "video"), None)
    if not video or int(video.get("width", 0)) != width or int(video.get("height", 0)) != height:
        raise RuntimeError("output_verification_failed:dimensions")
    # Full decode verification, not just successful container creation.
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(out), "-f", "null", "-"], check=True, timeout=MAX_DURATION*2)
    output_duration = float(meta.get("format", {}).get("duration", 0))
    expected_duration = sum(clip_lengths) - sum(duration for _, duration in active_transitions)
    if output_duration <= 0 or abs(output_duration - expected_duration) > max(1.0, len(clips) * 0.15):
        raise RuntimeError("output_verification_failed:duration")
    return out, {"width": width, "height": height, "videoCodec": video.get("codec_name"), "durationSeconds": output_duration, "bytes": out.stat().st_size, "enhancementMode": "ai" if ai_models_used and enhancement_mode == "normal" else enhancement_mode, "aiModel": ai_models_used[0] if ai_models_used else None, "aiModels": ai_models_used, "aiGeneratedWidth": ai_generated_dimensions[0] if ai_generated_dimensions else None, "aiGeneratedHeight": ai_generated_dimensions[1] if ai_generated_dimensions else None}

def main():
    print("Vyntra worker ready", WORKER_ID, flush=True)
    while True:
        try:
            cleanup_expired_release_outputs()
            target_job_id = os.getenv("VYNTRA_TARGET_JOB_ID") if ONESHOT else None
            job = claim_render_job(target_job_id)
            if not job:
                if ONESHOT:
                    print("No queued Vyntra render jobs", flush=True)
                    return
                time.sleep(POLL_SECONDS); continue
            stage = "prepare"
            try:
                update_job(job["id"], status="processing", stage="assigned", progress=0, worker_id=WORKER_ID)
                old_output_metadata = job.get("output_metadata")
                if isinstance(old_output_metadata, dict) and old_output_metadata.get("provider") == "github_release":
                    cleanup_release_output(job)
                else:
                    stale_output_paths = job_output_paths(job)
                    if stale_output_paths:
                        storage_remove(stale_output_paths, job)
                        update_job(job["id"], output_path=None, output_metadata=None)
                with tempfile.TemporaryDirectory(prefix="vyntra-render-") as directory:
                    stage = "render_and_verify"
                    output, metadata = render(job, Path(directory))
                    current = api("/rest/v1/vyntra_render_jobs?id=eq." + urllib.parse.quote(job["id"]) + "&select=status")
                    if current and current[0]["status"] == "cancel_requested":
                        raise InterruptedError("cancelled")
                    update_job(job["id"], status="finalizing", stage="uploading_release", progress=96)
                    stage = "upload_output_parts"
                    output_path, metadata = upload_output_release(output, job, metadata, Path(directory))
                    stage = "complete_job_record"
                    report_render_completed(job, output_path, metadata)
                    # Inputs are no longer needed once the verified output and
                    # completed state are persisted. Keep cleanup best-effort
                    # so an output remains deliverable during storage outages.
                    cleanup_source_objects(job)
            except InterruptedError:
                update_job(job["id"], status="cancelled", stage="cancelled", progress=0, completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            except Exception as exc:
                message = str(exc)
                raw_code = message.split(":",1)[0][:80] or "render_failed"
                if raw_code == "ffmpeg_failed":
                    code = ffmpeg_error_code(message.partition(":")[2])
                elif raw_code == "storage_upload_failed":
                    code = ("storage_upload_failed:" + ":".join(message.split(":")[1:]))[:100]
                elif raw_code.startswith("release_completion_http_"):
                    detail = message.partition(":")[2]
                    detail_parts = detail.split(":", 1)
                    valid_detail = detail_parts[0] in COMPLETION_ERROR_CODES
                    if valid_detail and len(detail_parts) == 2:
                        fields = detail_parts[1].split(",")
                        valid_detail = detail_parts[0] == "output_manifest_invalid" and all(
                            field in COMPLETION_MANIFEST_FIELDS for field in fields
                        )
                    code = f"{raw_code}:{detail}"[:100] if valid_detail else raw_code
                else:
                    code = raw_code
                safe_message = safe_ffmpeg_diagnostic(message)
                print("job failed", job.get("id"), "at", stage, type(exc).__name__, flush=True)
                print("job error summary:", safe_message or "[no diagnostic output]", flush=True)
                retryable = is_retryable_job_error(job, exc, raw_code)
                if int(job.get("attempts", 0)) < 3 and retryable:
                    update_job(job["id"], status="queued", stage="queued", error_code=code, progress=0)
                else:
                    update_job(job["id"], status="failed", stage="failed", error_code=code, progress=0, completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            if ONESHOT:
                return
        except Exception as exc:
            print("worker loop error", type(exc).__name__, flush=True)
            if ONESHOT:
                raise
            time.sleep(POLL_SECONDS)

if __name__ == "__main__": main()
