import json
import os
import platform
import shutil
import threading
import uuid
import zipfile
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel

from job_manager import Job, JobStore, utc_now
from pipeline import process_selected_job, segment_job
from utils import get_device, load_config


APP_VERSION = "0.2.0"
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_IMAGE_PIXELS = 100_000_000

app = FastAPI(title="Subject Extractor", version=APP_VERSION)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)

STORE = JobStore(OUTPUTS_DIR, max_workers=1)
ACTIVE_JOB_ID = None


class Selection(BaseModel):
    selected_ids: list[str]
    api_key: str = None
    job_id: str = None


class BoxPrompt(BaseModel):
    bbox: list[int]


class PolygonPrompt(BaseModel):
    points: list[list[int]]


def _get_job(job_id: str):
    job = STORE.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _mask_payload(job: Job, mask: dict):
    payload = dict(mask)
    thumb_file = Path(mask["thumb_file"]).name
    payload["thumb_file"] = thumb_file
    payload["thumb_url"] = f"/api/jobs/{job.id}/files/{quote(thumb_file)}"
    if mask.get("mask_file"):
        mask_file = Path(mask["mask_file"]).name
        payload["mask_file"] = mask_file
        payload["mask_url"] = f"/api/jobs/{job.id}/files/{quote(mask_file)}"
    return payload


def _job_payload(job: Job, include_masks=True):
    payload = {
        "job_id": job.id,
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "message": job.message,
        "mode": job.mode,
        "label_mode": job.label_mode,
        "padding_ratio": job.padding_ratio,
        "feather_kernel": job.feather_kernel,
        "error": job.error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "image_url": f"/api/jobs/{job.id}/source",
        "contact_sheet": (
            f"/api/jobs/{job.id}/files/{quote(Path(job.contact_sheet).name)}"
            if job.contact_sheet else None
        ),
        "download_url": f"/api/jobs/{job.id}/download",
        "selected_ids": list(job.selected_ids),
    }
    if include_masks:
        payload["masks"] = [_mask_payload(job, mask) for mask in job.masks_data]
    return payload


def _safe_file(job: Job, filename: str):
    job_dir = job.output_dir.resolve()
    requested = (job_dir / filename).resolve()
    try:
        requested.relative_to(job_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid file path") from exc
    if not requested.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return requested


def _create_export(job: Job):
    export_path = job.output_dir / f"subject-extractor-{job.id}.zip"
    with zipfile.ZipFile(export_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(job.output_dir.glob("*.png")):
            archive.write(path, path.name)
        for filename in ("manifest.json",):
            path = job.output_dir / filename
            if path.is_file():
                archive.write(path, path.name)
    return export_path


@app.get("/api/jobs")
def list_jobs():
    return {"jobs": [_job_payload(job, include_masks=False) for job in STORE.list()]}


@app.get("/api/health")
def health():
    config = load_config()
    model_path = Path(config.get("model_paths", {}).get("mobilesam", "models/mobile_sam.pt"))
    if not model_path.is_absolute():
        model_path = BASE_DIR / model_path
    disk = shutil.disk_usage(OUTPUTS_DIR)
    return {
        "status": "ok",
        "app_version": APP_VERSION,
        "platform": platform.platform(),
        "device": get_device(config),
        "tesseract": bool(shutil.which("tesseract")),
        "mobilesam_checkpoint": model_path.is_file(),
        "disk_free_gb": round(disk.free / (1024 ** 3), 2),
    }


@app.get("/api/models")
def models():
    config = load_config()
    checkpoint = Path(config.get("model_paths", {}).get("mobilesam", "models/mobile_sam.pt"))
    if not checkpoint.is_absolute():
        checkpoint = BASE_DIR / checkpoint
    u2net_cache = BASE_DIR / "models" / "u2net"
    return {
        "models": [
            {"name": "u2netp", "kind": "fast", "cached": (u2net_cache / "u2netp.onnx").is_file()},
            {"name": "mobilesam", "kind": "quality", "cached": checkpoint.is_file()},
            {"name": "moondream2", "kind": "labeling", "cached": False},
        ]
    }


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return _job_payload(_get_job(job_id))


@app.get("/api/jobs/{job_id}/source")
def get_source_image(job_id: str):
    job = _get_job(job_id)
    if not job.source_path.is_file():
        raise HTTPException(status_code=404, detail="Source image not found")
    return FileResponse(job.source_path)


@app.get("/api/jobs/{job_id}/files/{filename:path}")
def get_job_file(job_id: str, filename: str):
    job = _get_job(job_id)
    return FileResponse(_safe_file(job, filename))


@app.get("/api/jobs/{job_id}/download")
def download_job(job_id: str):
    job = _get_job(job_id)
    if job.status != "completed":
        raise HTTPException(status_code=409, detail="Job is not complete")
    return FileResponse(_create_export(job), filename=f"subject-extractor-{job.id}.zip")


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = _get_job(job_id)
    if not job.request_cancel():
        raise HTTPException(status_code=409, detail="Job cannot be cancelled in its current state")
    return _job_payload(job, include_masks=False)


@app.post("/api/jobs/{job_id}/retry")
def retry_job(job_id: str):
    job = _get_job(job_id)
    if job.status not in {"failed", "cancelled"}:
        raise HTTPException(status_code=409, detail="Only failed or cancelled jobs can be retried")
    job.update(status="queued", stage="queued", progress=0, message="Queued for retry",
               error=None, cancel_requested=False, masks_data=[], selected_ids=[], final_data=[])
    STORE.submit(job.id, segment_job)
    return _job_payload(job, include_masks=False)


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    if not STORE.delete(job_id):
        raise HTTPException(status_code=409, detail="Only completed, failed, or cancelled jobs can be deleted")
    return {"status": "ok", "job_id": job_id}


def _add_box_mask(job: Job, box_prompt: BoxPrompt):
    if len(box_prompt.bbox) != 4:
        raise HTTPException(status_code=400, detail="bbox must contain x, y, width, and height")
    image = cv2.imread(str(job.source_path))
    if image is None:
        raise HTTPException(status_code=400, detail="Could not load the source image")

    x, y, width, height = [int(value) for value in box_prompt.bbox]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(image.shape[1], x + width), min(image.shape[0], y + height)
    if x2 <= x1 or y2 <= y1:
        raise HTTPException(status_code=400, detail="bbox must overlap the source image")

    mask_id = f"custom_box_{uuid.uuid4().hex[:8]}"
    mask_filename = f"{mask_id}_mask.png"
    thumb_filename = f"{mask_id}_thumb.png"
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    cv2.imwrite(str(job.output_dir / mask_filename), mask)
    cropped_image = image[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]
    cv2.imwrite(str(job.output_dir / thumb_filename), cv2.merge((*cv2.split(cropped_image), cropped_mask)))

    new_mask = {
        "id": mask_id,
        "mask_file": mask_filename,
        "thumb_file": thumb_filename,
        "area": int((x2 - x1) * (y2 - y1)),
        "bbox": [x1, y1, x2 - x1, y2 - y1],
        "segmentation_mode": "manual_box",
        "skip_refine": False,
    }
    job.masks_data.append(new_mask)
    return new_mask


def _add_polygon_mask(job: Job, polygon: PolygonPrompt):
    if len(polygon.points) < 3 or any(len(point) != 2 for point in polygon.points):
        raise HTTPException(status_code=400, detail="A polygon needs at least three x,y points")
    image = cv2.imread(str(job.source_path))
    if image is None:
        raise HTTPException(status_code=400, detail="Could not load the source image")
    points = np.array([[int(point[0]), int(point[1])] for point in polygon.points], dtype=np.int32)
    points[:, 0] = np.clip(points[:, 0], 0, image.shape[1] - 1)
    points[:, 1] = np.clip(points[:, 1], 0, image.shape[0] - 1)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise HTTPException(status_code=400, detail="Polygon does not cover the image")

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    mask_id = f"custom_polygon_{uuid.uuid4().hex[:8]}"
    mask_filename = f"{mask_id}_mask.png"
    thumb_filename = f"{mask_id}_thumb.png"
    cv2.imwrite(str(job.output_dir / mask_filename), mask)
    crop = image[y1:y2, x1:x2]
    cv2.imwrite(str(job.output_dir / thumb_filename), cv2.merge((*cv2.split(crop), mask[y1:y2, x1:x2])))
    new_mask = {
        "id": mask_id, "mask_file": mask_filename, "thumb_file": thumb_filename,
        "area": int((mask > 0).sum()), "bbox": [x1, y1, x2 - x1, y2 - y1],
        "segmentation_mode": "manual_polygon", "skip_refine": False,
    }
    job.masks_data.append(new_mask)
    return new_mask


@app.post("/api/jobs/{job_id}/prompt_box")
def prompt_box(job_id: str, box_prompt: BoxPrompt):
    job = _get_job(job_id)
    if job.status != "ready":
        raise HTTPException(status_code=409, detail="Job is not ready for curation")
    return {"status": "ok", "mask": _mask_payload(job, _add_box_mask(job, box_prompt))}


@app.post("/api/jobs/{job_id}/prompt_polygon")
def prompt_polygon(job_id: str, polygon: PolygonPrompt):
    job = _get_job(job_id)
    if job.status != "ready":
        raise HTTPException(status_code=409, detail="Job is not ready for curation")
    return {"status": "ok", "mask": _mask_payload(job, _add_polygon_mask(job, polygon))}


@app.delete("/api/jobs/{job_id}/masks/{mask_id}")
def delete_mask(job_id: str, mask_id: str):
    """Remove one manually added mask while a job is being curated."""
    job = _get_job(job_id)
    if job.status != "ready":
        raise HTTPException(status_code=409, detail="Masks can only be removed during curation")

    with job.lock:
        mask = next((item for item in job.masks_data if item.get("id") == mask_id), None)
        if mask is None:
            raise HTTPException(status_code=404, detail="Mask not found")
        job.masks_data = [item for item in job.masks_data if item.get("id") != mask_id]
        job.selected_ids = [selected_id for selected_id in job.selected_ids if selected_id != mask_id]
        job.updated_at = utc_now()

    job_dir = job.output_dir.resolve()
    for key in ("mask_file", "thumb_file"):
        filename = mask.get(key)
        if not filename:
            continue
        candidate = (job_dir / Path(filename).name).resolve()
        if candidate.parent == job_dir and candidate.is_file():
            candidate.unlink()

    return {"status": "ok", "job_id": job.id, "mask_id": mask_id}


@app.post("/api/jobs/{job_id}/select")
def select_job(job_id: str, selection: Selection):
    job = _get_job(job_id)
    if job.status != "ready":
        raise HTTPException(status_code=409, detail="Job is not ready for selection")
    if selection.api_key and len(selection.api_key.strip()) < 5:
        raise HTTPException(status_code=400, detail="Invalid API key format")

    known_ids = {mask["id"] for mask in job.masks_data}
    if set(selection.selected_ids) - known_ids:
        raise HTTPException(status_code=400, detail="Selection contains unknown subjects")
    if not selection.selected_ids:
        raise HTTPException(status_code=400, detail="Select at least one subject")

    job.update(selected_ids=list(selection.selected_ids), api_key=selection.api_key)
    job.selection_event.set()
    if job.pipeline_on_select:
        job.update(status="queued", stage="pipeline", progress=35, message="Queued for extraction")
        STORE.submit(job.id, process_selected_job)
    return _job_payload(job, include_masks=False)


@app.post("/api/upload")
async def upload_image(
    file: UploadFile = File(...),
    mode: str = Form("fast"),
    label_mode: str = Form("ai"),
    padding_ratio: float = Form(0.1),
    feather_kernel: int = Form(5),
):
    mode = mode.lower().strip()
    if mode not in {"fast", "quality"}:
        raise HTTPException(status_code=400, detail="Mode must be fast or quality")
    label_mode = label_mode.lower().strip()
    if label_mode not in {"ai", "ocr", "basic"}:
        raise HTTPException(status_code=400, detail="Label mode must be ai, ocr, or basic")
    if not 0 <= padding_ratio <= 1:
        raise HTTPException(status_code=400, detail="Padding must be between 0 and 1")
    if not 1 <= feather_kernel <= 31:
        raise HTTPException(status_code=400, detail="Feather must be between 1 and 31")

    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell()
    file.file.seek(0)
    if file_size <= 0 or file_size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be between 1 byte and 50 MB")

    job_id = uuid.uuid4().hex[:12]
    output_dir = OUTPUTS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / "source.jpg"
    try:
        with Image.open(file.file) as image:
            image = ImageOps.exif_transpose(image)
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Image dimensions are too large")
            image.convert("RGB").save(source_path, format="JPEG", quality=95)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc

    job = STORE.create(
        source_path, mode=mode, output_dir=output_dir, job_id=job_id,
        label_mode=label_mode, padding_ratio=padding_ratio,
        feather_kernel=feather_kernel,
    )
    STORE.submit(job.id, segment_job)
    return _job_payload(job, include_masks=False)


@app.post("/api/batch")
async def upload_batch(
    files: list[UploadFile] = File(...),
    mode: str = Form("fast"),
    label_mode: str = Form("ai"),
):
    if not files or len(files) > 20:
        raise HTTPException(status_code=400, detail="Batch size must be between 1 and 20 images")
    jobs = []
    for file in files:
        jobs.append(await upload_image(file, mode, label_mode, 0.1, 5))
    return {"jobs": jobs}


@app.get("/api/masks")
def legacy_masks():
    if ACTIVE_JOB_ID is None:
        return {"job_id": None, "masks": [], "status": "idle"}
    job = _get_job(ACTIVE_JOB_ID)
    return _job_payload(job)


@app.post("/api/prompt_box")
def legacy_prompt_box(box_prompt: BoxPrompt):
    if ACTIVE_JOB_ID is None:
        raise HTTPException(status_code=400, detail="No active job")
    return prompt_box(ACTIVE_JOB_ID, box_prompt)


@app.post("/api/submit")
def legacy_submit(selection: Selection):
    job_id = selection.job_id or ACTIVE_JOB_ID
    if job_id is None:
        raise HTTPException(status_code=400, detail="No active job")
    return select_job(job_id, selection)


@app.get("/api/source")
def legacy_source():
    if ACTIVE_JOB_ID is None:
        raise HTTPException(status_code=404, detail="No active job")
    return get_source_image(ACTIVE_JOB_ID)


@app.get("/api/files/{filename:path}")
def legacy_file(filename: str):
    if ACTIVE_JOB_ID is None:
        raise HTTPException(status_code=404, detail="No active job")
    return get_job_file(ACTIVE_JOB_ID, filename)


@app.get("/", response_class=HTMLResponse)
def index():
    with open(BASE_DIR / "static" / "curate.html", "r") as html_file:
        return html_file.read()


def start_server(output_dir, masks_data, port=8000, source_path=None, mode="quality"):
    global ACTIVE_JOB_ID
    job = STORE.create(
        source_path=Path(source_path).resolve() if source_path else Path(output_dir) / "source.jpg",
        mode=mode,
        output_dir=Path(output_dir).resolve(),
        masks_data=masks_data,
        status="ready",
        pipeline_on_select=False,
    )
    ACTIVE_JOB_ID = job.id
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)

    def serve():
        server.run()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    print(f"Curation UI available at: http://127.0.0.1:{port}")
    print("Waiting for your selection in the browser...")
    job.selection_event.wait()
    server.should_exit = True
    thread.join(timeout=5)
    return job.selected_ids, job.api_key, job.masks_data
