"""Hugging Face Video Annotation Studio: segment-level video annotations.

Run: HF_REPO_ID=your-name/your-dataset uvicorn app:app --reload
Modes: hub_files (MP4/AVI files), video_column (HF datasets Video column), local.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import mimetypes
import os
import secrets
import shutil
import subprocess
import sqlite3
import threading
import uuid
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static" / "index.html"
DATA_DIR = Path(os.getenv("ANNOTATION_DIR", str(ROOT / "storage"))).expanduser().resolve()
CACHE_DIR = DATA_DIR / "video_cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "annotations.sqlite3"

HF_REPO_ID = os.getenv("HF_REPO_ID", "franciszzj/SCAND").strip()
HF_VIDEO_DIR = os.getenv("HF_VIDEO_DIR", "random_mdps").strip().strip("/")
HF_TOKEN = os.getenv("HF_TOKEN") or None
HF_SPLIT = os.getenv("HF_SPLIT", "train").strip()
HF_VIDEO_COLUMN = os.getenv("HF_VIDEO_COLUMN", "video").strip()
SOURCE_MODE = os.getenv("SOURCE_MODE", "browser_upload").strip().lower()
LOCAL_VIDEO_DIR = Path(os.getenv("LOCAL_VIDEO_DIR", str(ROOT / "videos"))).expanduser().resolve()
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
APP_USERNAME = os.getenv("APP_USERNAME", "annotator")
EXTS = {".mp4", ".webm", ".mov", ".m4v", ".avi", ".mkv", ".ogv"}

app = FastAPI(title="Video Annotation Studio", version="1.0")
state_lock = threading.RLock()
manifest: list[dict[str, Any]] | None = None
video_iterator: Any = None
stream_finished = False
cached_paths: dict[int, Path] = {}
converted_local_paths: dict[str, Path] = {}


def db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    return conn


def setup_db() -> None:
    with db_conn() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS annotations (
            id TEXT PRIMARY KEY,
            repo_id TEXT NOT NULL,
            video_id TEXT NOT NULL,
            video_index INTEGER NOT NULL,
            video_path TEXT NOT NULL,
            start_sec REAL NOT NULL,
            end_sec REAL NOT NULL,
            label TEXT NOT NULL,
            pedestrian_type TEXT NOT NULL DEFAULT '',
            behavior TEXT NOT NULL DEFAULT '',
            notes TEXT NOT NULL DEFAULT '',
            annotator TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS annotations_video ON annotations(video_id)")


setup_db()


@app.middleware("http")
async def optional_basic_auth(request: Request, call_next):
    if APP_PASSWORD:
        import base64
        auth = request.headers.get("Authorization", "")
        supplied_user, supplied_password = "", ""
        if auth.lower().startswith("basic "):
            try:
                supplied_user, supplied_password = base64.b64decode(auth[6:]).decode().split(":", 1)
            except (ValueError, UnicodeDecodeError):
                pass
        if not (secrets.compare_digest(supplied_user, APP_USERNAME)
                and secrets.compare_digest(supplied_password, APP_PASSWORD)):
            return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="Video Annotation Studio"'})
    return await call_next(request)


def repository_label() -> str:
    return "Local browser videos (video bytes stay on your computer)" if SOURCE_MODE == "browser_upload" else str(LOCAL_VIDEO_DIR) if SOURCE_MODE == "local" else HF_REPO_ID


def video_record(index: int, path: str, size: int | None = None) -> dict[str, Any]:
    prefix = f"{HF_SPLIT}:" if SOURCE_MODE == "video_column" else ""
    if SOURCE_MODE == "browser_upload":
        key = f"upload:{Path(path).name.casefold()}"
    elif SOURCE_MODE == "local":
        key = f"local:{path}"
    elif SOURCE_MODE == "video_column":
        key = f"{HF_REPO_ID}:{HF_SPLIT}:row-{index}"
    else:
        key = f"{HF_REPO_ID}:{path}"
    return {"index": index, "id": key, "path": path, "name": Path(path).name or f"Video {index+1}",
            "repo_id": repository_label(), "size_bytes": size}


def initialize_source() -> None:
    global manifest, video_iterator, stream_finished
    if manifest is not None:
        return
    if SOURCE_MODE == "browser_upload":
        # User chooses files in the browser; never transfer their video bytes.
        manifest = []
        stream_finished = True
    elif SOURCE_MODE == "local":
        if not LOCAL_VIDEO_DIR.is_dir():
            raise HTTPException(503, f"Local directory not found: {LOCAL_VIDEO_DIR}")
        paths = sorted(str(p.relative_to(LOCAL_VIDEO_DIR)) for p in LOCAL_VIDEO_DIR.rglob("*")
                       if p.is_file() and p.suffix.lower() in EXTS)
        manifest = [video_record(i, p) for i, p in enumerate(paths)]
        stream_finished = True
    elif SOURCE_MODE == "hub_files":
        if not HF_REPO_ID:
            raise HTTPException(503, "Set HF_REPO_ID before starting the server (e.g., owner/dataset-name).")
        try:
            api = HfApi(token=HF_TOKEN)
            # Listing metadata does not download the video bytes. The file-size
            # ordering makes the first SCAND sample inexpensive to try.
            entries = api.list_repo_tree(repo_id=HF_REPO_ID, repo_type="dataset", recursive=True)
            candidates = [(entry.path, entry.size) for entry in entries
                          if isinstance(entry, RepoFile)
                          and Path(entry.path).suffix.lower() in EXTS
                          and (not HF_VIDEO_DIR or entry.path.startswith(HF_VIDEO_DIR + "/"))]
        except Exception as exc:
            raise HTTPException(503, f"Could not list dataset files: {exc.__class__.__name__}: {exc}") from exc
        # Prefer browser-friendly MP4 files; show smaller files first.
        candidates.sort(key=lambda pair: (Path(pair[0]).suffix.lower() not in {".mp4", ".webm", ".m4v"},
                                          pair[1] if pair[1] is not None else float("inf"),
                                          pair[0].lower()))
        manifest = [video_record(i, path, size) for i, (path, size) in enumerate(candidates)]
        stream_finished = True
    elif SOURCE_MODE == "video_column":
        if not HF_REPO_ID:
            raise HTTPException(503, "Set HF_REPO_ID before starting the server.")
        try:
            from datasets import load_dataset
            ds = load_dataset(HF_REPO_ID, split=HF_SPLIT, streaming=True, token=HF_TOKEN)
            # This avoids constructing decoded torchcodec VideoDecoder objects.
            if hasattr(ds, "decode"):
                ds = ds.decode(False)
            else:
                from datasets import Video
                ds = ds.cast_column(HF_VIDEO_COLUMN, Video(decode=False))
            video_iterator = iter(ds)
        except Exception as exc:
            raise HTTPException(503, f"Could not open Video column: {exc.__class__.__name__}: {exc}") from exc
        manifest = []
        stream_finished = False
    else:
        raise HTTPException(503, "SOURCE_MODE must be browser_upload, hub_files, video_column, or local.")


def ensure_index(index: int) -> dict[str, Any]:
    global stream_finished
    with state_lock:
        initialize_source()
        assert manifest is not None
        if index < 0:
            raise HTTPException(404, "Video does not exist.")
        if SOURCE_MODE == "video_column":
            while len(manifest) <= index and not stream_finished:
                try:
                    row = next(video_iterator)
                except StopIteration:
                    stream_finished = True
                    break
                except Exception as exc:
                    raise HTTPException(503, f"Could not read dataset row: {exc}") from exc
                current_index = len(manifest)
                item = row.get(HF_VIDEO_COLUMN)
                if not isinstance(item, (dict, str, bytes, bytearray)):
                    raise HTTPException(503, f"Column {HF_VIDEO_COLUMN!r} was not an undecoded Video column.")
                source_path = item.get("path") if isinstance(item, dict) else item if isinstance(item, str) else None
                display_name = source_path or str(row.get("id") or f"row_{current_index:05d}.mp4")
                record = video_record(current_index, str(display_name))
                manifest.append(record)
                # A streaming row is consumed once, so cache its video immediately.
                cached_paths[current_index] = cache_column_video(item, current_index)
        if index >= len(manifest):
            raise HTTPException(404, "No more videos.")
        return manifest[index]


def cache_column_video(item: Any, index: int) -> Path:
    if isinstance(item, dict):
        content, filename = item.get("bytes"), item.get("path")
    elif isinstance(item, (bytes, bytearray)):
        content, filename = item, None
    else:
        content, filename = None, item
    if content is not None:
        ext = Path(str(filename or "video.mp4")).suffix.lower()
        if ext not in EXTS:
            ext = ".mp4"
        name = hashlib.sha256(f"{HF_REPO_ID}:{HF_SPLIT}:{index}".encode()).hexdigest() + ext
        cached_file = CACHE_DIR / name
        if not cached_file.is_file():
            tmp = CACHE_DIR / (name + ".tmp")
            tmp.write_bytes(bytes(content))
            tmp.replace(cached_file)
        return cached_file
    if not filename:
        raise HTTPException(503, f"Row {index} has no video bytes or path.")
    filepath = Path(str(filename))
    if filepath.is_file():
        return filepath
    try:
        return Path(hf_hub_download(repo_id=HF_REPO_ID, filename=str(filename),
                                    repo_type="dataset", token=HF_TOKEN))
    except Exception as exc:
        raise HTTPException(503, f"Could not fetch video from row {index}: {exc}") from exc


def local_video_path(index: int) -> Path:
    record = ensure_index(index)
    with state_lock:
        if index in cached_paths:
            return cached_paths[index]
    if SOURCE_MODE == "browser_upload":
        raise HTTPException(400, "Browser-upload videos are played directly from the local file selector, not /api/video.")
    if SOURCE_MODE == "local":
        path = (LOCAL_VIDEO_DIR / record["path"]).resolve()
        if not path.is_relative_to(LOCAL_VIDEO_DIR) or not path.is_file():
            raise HTTPException(404, "Local video missing.")
    elif SOURCE_MODE == "hub_files":
        try:
            path = Path(hf_hub_download(repo_id=HF_REPO_ID, filename=record["path"],
                                        repo_type="dataset", token=HF_TOKEN))
        except Exception as exc:
            raise HTTPException(503, f"Could not download video: {exc}") from exc
    else:
        raise HTTPException(503, "Streaming row was not cached.")
    # Browsers cannot reliably decode AVI/MKV. Convert just this requested
    # file to a streamable MP4 when FFmpeg is available; retain source cache.
    if path.suffix.lower() in {".avi", ".mkv", ".mov", ".ogv", ".mp4", ".m4v"}:
        converted_name = hashlib.sha256(record["id"].encode()).hexdigest() + ".mp4"
        converted = CACHE_DIR / converted_name
        if not converted.is_file():
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg is None:
                raise HTTPException(503, "Video normalization needs FFmpeg on your PATH. Install FFmpeg and restart the server.")
            temporary = CACHE_DIR / (converted_name + ".tmp.mp4")
            try:
                subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                                "-i", str(path), "-c:v", "libx264", "-preset", "veryfast",
                                "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac",
                                "-movflags", "+faststart", str(temporary)],
                               capture_output=True, text=True, timeout=3600, check=True)
                temporary.replace(converted)
            except (subprocess.SubprocessError, OSError) as exc:
                temporary.unlink(missing_ok=True)
                raise HTTPException(503, f"FFmpeg could not convert this video: {exc}") from exc
        path = converted
    with state_lock:
        cached_paths[index] = path
    return path


class RegisterVideoInput(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(default=0, ge=0)


@app.post("/api/register-video")
def register_video(item: RegisterVideoInput):
    if SOURCE_MODE != "browser_upload":
        raise HTTPException(400, "Set SOURCE_MODE=browser_upload to register local browser files.")
    name = item.name.replace("\\", "/").split("/")[-1].strip()
    if not name or name in {".", ".."} or Path(name).suffix.lower() not in EXTS or "\x00" in name:
        raise HTTPException(422, "Choose a video file (MP4, WebM, MOV, M4V, AVI, MKV, OGV).")
    with state_lock:
        initialize_source()
        assert manifest is not None
        key = f"upload:{name.casefold()}"
        for record in manifest:
            if record["id"] == key:
                record["size_bytes"] = item.size_bytes
                return {"video": record, "existing": True}
        record = video_record(len(manifest), name, item.size_bytes)
        manifest.append(record)
        # If this is a replacement for an earlier Hugging Face-based Studio,
        # bring across records from the old SQLite database by matching the
        # basename. Retain the original rows for safety, but export only local
        # annotations in browser-upload mode to avoid duplicates.
        migrated = 0
        with db_conn() as conn:
            already = conn.execute("SELECT 1 FROM annotations WHERE video_id=? LIMIT 1", (key,)).fetchone()
            if not already:
                old = conn.execute("SELECT * FROM annotations WHERE video_id NOT LIKE 'upload:%'").fetchall()
                for source in old:
                    if str(source["video_path"]).replace("\\", "/").split("/")[-1].casefold() != name.casefold():
                        continue
                    row = dict(source)
                    row.update(id=str(uuid.uuid4()), repo_id=repository_label(),
                               video_id=key, video_index=record["index"], video_path=name)
                    columns = ", ".join(row)
                    values = ", ".join("?" for _ in row)
                    conn.execute(f"INSERT INTO annotations ({columns}) VALUES ({values})", list(row.values()))
                    migrated += 1
        return {"video": record, "existing": False, "migrated": migrated}


class AnnotationInput(BaseModel):
    start_sec: float = Field(ge=0, allow_inf_nan=False)
    end_sec: float = Field(gt=0, allow_inf_nan=False)
    label: str = Field(min_length=1, max_length=120)
    pedestrian_type: str = Field(default="", max_length=120)
    behavior: str = Field(default="", max_length=200)
    notes: str = Field(default="", max_length=5000)
    annotator: str = Field(default="", max_length=120)


def validate_interval(annotation: AnnotationInput) -> None:
    if annotation.end_sec <= annotation.start_sec:
        raise HTTPException(422, "End time must be after start time.")


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(STATIC.read_text(encoding="utf-8"))


@app.get("/api/videos")
def list_videos(upto: int = 0):
    if upto < 0 or upto > 100000:
        raise HTTPException(422, "Invalid video index.")
    with state_lock:
        initialize_source()
        if SOURCE_MODE == "video_column" and not stream_finished:
            try:
                ensure_index(upto)
            except HTTPException as exc:
                if exc.status_code != 404:
                    raise
        assert manifest is not None
        with db_conn() as conn:
            counts = {r["video_id"]: r["cnt"] for r in conn.execute(
                "SELECT video_id, COUNT(*) AS cnt FROM annotations GROUP BY video_id")}
        return {"videos": [{**v, "annotation_count": counts.get(v["id"], 0)} for v in manifest],
                "has_more": not stream_finished, "mode": SOURCE_MODE, "dataset": repository_label(),
                "split": HF_SPLIT if SOURCE_MODE == "video_column" else None}


@app.get("/api/video/{index}")
def play_video(index: int):
    path = local_video_path(index)
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=None, content_disposition_type="inline",
                        )


@app.post("/api/repair-video/{index}")
def repair_local_video(index: int, video: UploadFile = File(...)):
    """Optional local-only upload and FFmpeg conversion for incompatible browser video.

    This does not change the saved video ID/filename or the annotation database.
    The Windows launcher binds to 127.0.0.1, so raw bytes never leave the PC.
    """
    if SOURCE_MODE != "browser_upload":
        raise HTTPException(400, "Fix playback is available in browser-upload mode only.")
    record = ensure_index(index)
    name = (video.filename or "").replace("\\", "/").split("/")[-1]
    if name.casefold() != record["name"].casefold():
        raise HTTPException(400, "The selected filename does not match the video being annotated.")
    if Path(name).suffix.lower() not in EXTS:
        raise HTTPException(400, "Unsupported video extension.")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise HTTPException(503, "FFmpeg is not on the Python server PATH. Close the server window and restart it after installing FFmpeg.")
    # Stream uploaded bytes to disk in bounded chunks; no in-memory video copy.
    import_hash = hashlib.sha256()
    import_hash.update(record["id"].encode())
    nbytes = 0
    source_tmp = None
    output_tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=CACHE_DIR, suffix=Path(name).suffix,
                                         prefix="repair-source-", delete=False) as incoming:
            source_tmp = Path(incoming.name)
            while True:
                chunk = video.file.read(4 * 1024 * 1024)
                if not chunk:
                    break
                nbytes += len(chunk)
                if nbytes > 10 * 1024**3:
                    raise HTTPException(413, "Video exceeds the 10 GB local repair limit.")
                import_hash.update(chunk)
                incoming.write(chunk)
        if nbytes == 0:
            raise HTTPException(422, "The uploaded video is empty.")
        digest = import_hash.hexdigest()
        ready = CACHE_DIR / ("browser-fixed-" + digest + ".mp4")
        if not ready.is_file() or ready.stat().st_size < 10000:
            output_tmp = CACHE_DIR / ("repair-output-" + uuid.uuid4().hex + ".mp4")
            command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                       "-i", str(source_tmp), "-map", "0:v:0", "-map", "0:a:0?",
                       "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
                       "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
                       "-sn", "-dn", "-movflags", "+faststart", str(output_tmp)]
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=7200)
            except subprocess.TimeoutExpired as exc:
                raise HTTPException(503, "FFmpeg conversion exceeded two hours.") from exc
            if result.returncode or not output_tmp.is_file() or output_tmp.stat().st_size < 10000:
                stderr = result.stderr.strip()[-1200:] or "no additional FFmpeg diagnostic"
                raise HTTPException(503, f"FFmpeg failed to convert this video: {stderr}")
            # Reject truncated output before moving it to permanent cache.
            ffprobe = shutil.which("ffprobe")
            if ffprobe:
                verified = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0",
                       "-show_entries", "stream=codec_name:format=duration", "-of", "json",
                       str(output_tmp)], capture_output=True, text=True, timeout=60)
                if verified.returncode or '"h264"' not in verified.stdout:
                    raise HTTPException(503, "FFmpeg output failed H.264 verification.")
            output_tmp.replace(ready)
        with state_lock:
            converted_local_paths[record["id"]] = ready
        return {"ok": True, "video_id": record["id"], "bytes_received": nbytes,
                "url": f"/api/fixed-video/{index}?version={digest[:16]}"}
    finally:
        video.file.close()
        if source_tmp:
            source_tmp.unlink(missing_ok=True)
        if output_tmp:
            output_tmp.unlink(missing_ok=True)


@app.get("/api/fixed-video/{index}")
def fixed_local_video(index: int):
    record = ensure_index(index)
    with state_lock:
        path = converted_local_paths.get(record["id"])
    if path is None or not path.is_file():
        raise HTTPException(404, "No repaired video for this selected file. Use Fix playback.")
    return FileResponse(path, media_type="video/mp4", content_disposition_type="inline")


@app.get("/api/annotations/{index}")
def get_annotations(index: int):
    record = ensure_index(index)
    with db_conn() as conn:
        rows = conn.execute("SELECT * FROM annotations WHERE video_id=? ORDER BY start_sec, end_sec",
                            (record["id"],)).fetchall()
    return {"annotations": [dict(row) for row in rows]}


@app.post("/api/annotations/{index}", status_code=201)
def create_annotation(index: int, annotation: AnnotationInput):
    record = ensure_index(index)
    validate_interval(annotation)
    timestamp = datetime.now(timezone.utc).isoformat()
    row = {"id": str(uuid.uuid4()), "repo_id": record["repo_id"], "video_id": record["id"],
           "video_index": index, "video_path": record["path"], **annotation.model_dump(),
           "created_at": timestamp, "updated_at": timestamp}
    with db_conn() as conn:
        columns = ", ".join(row)
        placeholders = ", ".join("?" for _ in row)
        conn.execute(f"INSERT INTO annotations ({columns}) VALUES ({placeholders})", list(row.values()))
    return row


@app.put("/api/annotations/{index}/{annotation_id}")
def update_annotation(index: int, annotation_id: str, annotation: AnnotationInput):
    record = ensure_index(index)
    validate_interval(annotation)
    updated_at = datetime.now(timezone.utc).isoformat()
    fields = annotation.model_dump()
    with db_conn() as conn:
        cursor = conn.execute(
            "UPDATE annotations SET start_sec=?, end_sec=?, label=?, pedestrian_type=?, behavior=?, "
            "notes=?, annotator=?, updated_at=? WHERE id=? AND video_id=?",
            (*fields.values(), updated_at, annotation_id, record["id"]))
        if not cursor.rowcount:
            raise HTTPException(404, "Annotation not found.")
        result = conn.execute("SELECT * FROM annotations WHERE id=?", (annotation_id,)).fetchone()
    return dict(result)


@app.delete("/api/annotations/{index}/{annotation_id}")
def delete_annotation(index: int, annotation_id: str):
    record = ensure_index(index)
    with db_conn() as conn:
        cursor = conn.execute("DELETE FROM annotations WHERE id=? AND video_id=?",
                              (annotation_id, record["id"]))
        if not cursor.rowcount:
            raise HTTPException(404, "Annotation not found.")
    return {"deleted": annotation_id}


EXPORT_FIELDS = ["id", "repo_id", "video_id", "video_index", "video_path", "start_sec",
                 "end_sec", "label", "pedestrian_type", "behavior", "notes", "annotator",
                 "created_at", "updated_at"]


def all_annotations() -> list[dict[str, Any]]:
    with db_conn() as conn:
        if SOURCE_MODE == "browser_upload":
            rows = conn.execute("SELECT * FROM annotations WHERE video_id LIKE 'upload:%' "
                                "ORDER BY video_path COLLATE NOCASE, start_sec, end_sec").fetchall()
        else:
            rows = conn.execute("SELECT * FROM annotations ORDER BY video_index, start_sec, end_sec").fetchall()
    return [dict(row) for row in rows]


@app.get("/api/export.csv")
def export_csv():
    content = io.StringIO()
    writer = csv.DictWriter(content, fieldnames=EXPORT_FIELDS)
    writer.writeheader()
    writer.writerows(all_annotations())
    return Response(content="\ufeff" + content.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="video_annotations.csv"'})


@app.get("/api/export.json")
def export_json():
    return Response(content=json.dumps(all_annotations(), ensure_ascii=False, indent=2),
                    media_type="application/json; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="video_annotations.json"'})


@app.get("/api/export.jsonl")
def export_jsonl():
    body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_annotations())
    return Response(content=body, media_type="application/x-ndjson",
                    headers={"Content-Disposition": 'attachment; filename="video_annotations.jsonl"'})


@app.get("/api/health")
def health():
    return {"ok": True, "mode": SOURCE_MODE, "configured": True}


@app.post("/api/import.json")
def import_json(payload: list[dict[str, Any]]):
    if SOURCE_MODE != "browser_upload":
        raise HTTPException(400, "JSON import is available in browser_upload mode.")
    if len(payload) > 100000:
        raise HTTPException(422, "Too many annotations in one import.")
    imported = 0
    skipped = 0
    with db_conn() as conn:
        for a in payload:
            name = str(a.get("video_path") or a.get("video_name") or "").replace("\\", "/").split("/")[-1].strip()
            if not name or Path(name).suffix.lower() not in EXTS:
                skipped += 1
                continue
            try:
                item = AnnotationInput.model_validate(a)
            except Exception:
                skipped += 1
                continue
            if item.end_sec <= item.start_sec:
                skipped += 1
                continue
            record_id = str(a.get("id") or uuid.uuid4())
            # Explicitly namespace imported backups by local video filename.
            video_id = f"upload:{name.casefold()}"
            now = datetime.now(timezone.utc).isoformat()
            row = {"id": record_id, "repo_id": repository_label(), "video_id": video_id,
                   "video_index": 0, "video_path": name, **item.model_dump(),
                   "created_at": str(a.get("created_at") or now),
                   "updated_at": str(a.get("updated_at") or now)}
            columns = ", ".join(row)
            placeholders = ", ".join("?" for _ in row)
            cursor = conn.execute(f"INSERT OR IGNORE INTO annotations ({columns}) VALUES ({placeholders})", list(row.values()))
            if cursor.rowcount:
                imported += 1
            else:
                skipped += 1
    return {"imported": imported, "skipped": skipped}
