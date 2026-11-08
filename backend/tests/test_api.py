"""Small unit tests for the safe, non-diagnostic backend operations."""

from __future__ import annotations

import io
import unittest

from PIL import Image
from fastapi.testclient import TestClient

from backend.app.main import app, inspect_image_quality


def sample_png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (96, 80), (95, 120, 105)).save(buffer, format="PNG")
    return buffer.getvalue()


class ImageQualityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_quality_report_has_dimensions_and_no_file_output(self) -> None:
        report = inspect_image_quality(sample_png())
        self.assertEqual((report["width"], report["height"]), (96, 80))
        self.assertEqual(report["channels"], "RGB (3 channels)")
        self.assertTrue(report["quality_check_only"])

    def test_corrupt_image_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "could not be decoded"):
            inspect_image_quality(b"not an image")

    def test_oversize_image_is_rejected_before_decode(self) -> None:
        from backend.app.main import MAX_UPLOAD_BYTES

        with self.assertRaisesRegex(ValueError, "10 MB"):
            inspect_image_quality(b"x" * (MAX_UPLOAD_BYTES + 1))

    def test_upload_endpoint_reads_in_memory_and_returns_quality(self) -> None:
        response = self.client.post("/api/image/quality", files={"image": ("sample.png", sample_png(), "image/png")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["format"], "PNG")
        self.assertEqual(response.json()["width"], 96)

    def test_upload_endpoint_rejects_invalid_image(self) -> None:
        response = self.client.post("/api/image/quality", files={"image": ("bad.png", b"bad", "image/png")})
        self.assertEqual(response.status_code, 400)


class AvailabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health_is_ok(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_unusable_checkpoint_is_never_enabled(self) -> None:
        response = self.client.get("/api/model/status")
        self.assertEqual(response.status_code, 200)
        status = response.json()
        self.assertFalse(status["available"])
        self.assertFalse(status["prediction_enabled"])
        self.assertFalse(status["clinical_validation"])

    def test_prediction_endpoint_fails_closed(self) -> None:
        response = self.client.post("/api/predict")
        self.assertEqual(response.status_code, 503)

    def test_gradcam_endpoint_fails_closed(self) -> None:
        response = self.client.post("/api/gradcam")
        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main(verbosity=2)
