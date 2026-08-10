from __future__ import annotations

import shutil
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional


class JobCancelled(Exception):
    """Raised when a job is cancelled between pipeline stages."""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    id: str
    output_dir: Path
    source_path: Path
    mode: str = "fast"
    label_mode: str = "ai"
    padding_ratio: float = 0.1
    feather_kernel: int = 5
    status: str = "queued"
    stage: str = "queued"
    progress: int = 0
    message: str = "Waiting to start"
    masks_data: list = field(default_factory=list)
    selected_ids: list = field(default_factory=list)
    api_key: Optional[str] = None
    final_data: list = field(default_factory=list)
    contact_sheet: Optional[str] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    cancel_requested: bool = False
    pipeline_on_select: bool = True
    selection_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def update(self, **changes):
        with self.lock:
            for key, value in changes.items():
                setattr(self, key, value)
            self.updated_at = utc_now()

    def check_cancelled(self):
        with self.lock:
            if self.cancel_requested:
                self.status = "cancelled"
                self.stage = "cancelled"
                self.message = "Cancelled by user"
                self.progress = min(self.progress, 99)
                self.updated_at = utc_now()
                raise JobCancelled()

    def request_cancel(self):
        with self.lock:
            if self.status in {"completed", "failed", "cancelled"}:
                return False
            self.cancel_requested = True
            self.message = "Cancellation requested"
            self.updated_at = utc_now()
            return True


class JobStore:
    """Thread-safe in-memory job registry with one worker for ML memory safety."""

    def __init__(self, output_root: Path, max_workers: int = 1):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.jobs = {}
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="extractor")
        self._load_existing()

    def _load_existing(self):
        for output_dir in self.output_root.iterdir():
            if not output_dir.is_dir():
                continue
            manifest_path = output_dir / "manifest.json"
            if not manifest_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text())
                source = output_dir / "source.jpg"
                if not source.is_file():
                    source = Path(manifest.get("source_image", source))
                job = Job(
                    id=output_dir.name,
                    output_dir=output_dir.resolve(),
                    source_path=source.resolve(),
                    mode=manifest.get("mode", "fast"),
                    label_mode=manifest.get("label_mode", "ai"),
                    padding_ratio=manifest.get("padding_ratio", 0.1),
                    feather_kernel=manifest.get("feather_kernel", 5),
                    status="completed",
                    stage="complete",
                    progress=100,
                    message="Recovered completed job",
                    final_data=manifest.get("subjects", []),
                    contact_sheet=str(output_dir / "contact_sheet.png")
                    if (output_dir / "contact_sheet.png").is_file() else None,
                )
                self.jobs[job.id] = job
            except (OSError, ValueError, json.JSONDecodeError):
                continue

    def create(self, source_path: Path, mode: str = "fast", output_dir: Optional[Path] = None,
               masks_data=None, status: str = "queued", pipeline_on_select: bool = True,
               job_id: Optional[str] = None, label_mode: str = "ai",
               padding_ratio: float = 0.1, feather_kernel: int = 5):
        job_id = job_id or uuid.uuid4().hex[:12]
        output_dir = Path(output_dir) if output_dir else self.output_root / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        job = Job(
            id=job_id,
            output_dir=output_dir.resolve(),
            source_path=Path(source_path).resolve(),
            mode=mode,
            label_mode=label_mode,
            padding_ratio=padding_ratio,
            feather_kernel=feather_kernel,
            status=status,
            stage="ready" if status == "ready" else "queued",
            progress=35 if status == "ready" else 0,
            message="Ready for curation" if status == "ready" else "Queued",
            masks_data=list(masks_data or []),
            pipeline_on_select=pipeline_on_select,
        )
        with self.lock:
            self.jobs[job_id] = job
        return job

    def get(self, job_id: str):
        with self.lock:
            return self.jobs.get(job_id)

    def submit(self, job_id: str, task: Callable[[Job], None]):
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return self.executor.submit(task, job)

    def list(self):
        with self.lock:
            return sorted(self.jobs.values(), key=lambda job: job.created_at, reverse=True)

    def delete(self, job_id: str):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return False
            if job.status in {"queued", "running"}:
                return False
            self.jobs.pop(job_id, None)
        if job.output_dir.exists():
            shutil.rmtree(job.output_dir)
        return True

    def shutdown(self):
        self.executor.shutdown(wait=False, cancel_futures=True)
