import json
import os
from pathlib import Path

from contact_sheet import create_contact_sheet
from extract import extract_subjects
from label import label_subjects
from refine import refine_edges
from segment import run_segmentation

from job_manager import JobCancelled


def _progress(job, stage, progress, message):
    job.check_cancelled()
    job.update(status="running", stage=stage, progress=progress, message=message, error=None)


def segment_job(job):
    try:
        _progress(job, "segment", 5, f"Analyzing image in {job.mode} mode...")
        masks = run_segmentation(str(job.source_path), str(job.output_dir), mode=job.mode)
        job.check_cancelled()
        if not masks:
            job.update(status="failed", stage="segment", progress=5, message="No subjects found",
                        error="No subjects were detected in this image")
            return
        job.masks_data = masks
        job.update(status="ready", stage="curate", progress=35,
                   message=f"Found {len(masks)} candidate subject(s)")
    except JobCancelled:
        pass
    except Exception as exc:
        job.update(status="failed", stage="segment", message="Analysis failed", error=str(exc))


def process_selected_job(job):
    try:
        _progress(job, "extract", 40, "Extracting selected subjects...")
        extracted = extract_subjects(
            str(job.source_path), job.selected_ids, job.masks_data, str(job.output_dir),
            padding_ratio=job.padding_ratio, feather_kernel=job.feather_kernel,
        )
        _progress(job, "refine", 60, "Refining edges...")
        refined = refine_edges(extracted, str(job.output_dir))
        _progress(job, "label", 75, "Generating labels and filenames...")
        final = label_subjects(
            refined, str(job.output_dir), api_key=job.api_key, label_mode=job.label_mode
        )
        job.check_cancelled()

        manifest = {
            "app_version": "0.2.0",
            "source_image": os.path.abspath(str(job.source_path)),
            "mode": job.mode,
            "label_mode": job.label_mode,
            "padding_ratio": job.padding_ratio,
            "feather_kernel": job.feather_kernel,
            "subjects": final,
        }
        with open(job.output_dir / "manifest.json", "w") as manifest_file:
            json.dump(manifest, manifest_file, indent=2)

        _progress(job, "export", 95, "Generating contact sheet...")
        contact_sheet = create_contact_sheet(final, str(job.output_dir))
        job.update(
            status="completed",
            stage="complete",
            progress=100,
            message=f"Extracted {len(final)} subject(s)",
            final_data=final,
            contact_sheet=contact_sheet,
        )
    except JobCancelled:
        pass
    except Exception as exc:
        job.update(status="failed", stage=job.stage, message="Extraction failed", error=str(exc))
