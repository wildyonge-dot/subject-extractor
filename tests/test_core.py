import tempfile
import unittest
from pathlib import Path

from PIL import Image

from contact_sheet import create_contact_sheet
from job_manager import JobStore
from segment import compute_iou


class CoreTests(unittest.TestCase):
    def test_iou_for_disjoint_boxes(self):
        self.assertEqual(compute_iou([0, 0, 10, 10], [20, 20, 5, 5]), 0.0)

    def test_contact_sheet_is_written(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            image_path = output_dir / "subject.png"
            Image.new("RGBA", (20, 20), (255, 0, 0, 255)).save(image_path)
            result = create_contact_sheet(
                [{"id": "subject_000", "filename": image_path.name}], str(output_dir)
            )
            self.assertTrue(Path(result).is_file())

    def test_job_store_creates_unique_jobs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = JobStore(Path(temp_dir), max_workers=1)
            source = Path(temp_dir) / "source.jpg"
            source.write_bytes(b"source")
            first = store.create(source)
            second = store.create(source)
            self.assertNotEqual(first.id, second.id)
            store.executor.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    unittest.main()
