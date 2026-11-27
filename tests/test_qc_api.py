import io
import unittest

from fastapi.testclient import TestClient
from PIL import Image

from defectclip.alerts import decision_for
from qcapi.errors import ServiceError
from qcapi.main import create_app


class FakeService:
    def health(self):
        return {
            "status": "ok",
            "model": "fake",
            "device": "cpu",
            "gallery_indexed": False,
            "alerts_armed": False,
            "default_threshold": None,
            "categories": ["bottle"],
        }

    def predict(self, image, category, threshold=None, image_name="upload"):
        score = 0.2
        decision = decision_for(score, threshold)
        return {
            "category": category,
            "score": score,
            "decision": decision,
            "threshold": threshold,
            "alerted": decision == "defect",
            "alert_channels": ["log"] if decision == "defect" else [],
            "alert_errors": [],
            "model": "fake",
            "inspection_id": 1,
            "image_name": image_name,
            "note": "test double",
        }

    def search(self, category, query, k):
        if category == "missing":
            raise ServiceError(409, "no cache")
        return {
            "category": category,
            "query": query,
            "hits": [{"rank": 1, "defect": "scratch", "score": 0.4, "path": "metal_nut/scratch.png"}],
        }


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


class ApiTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app(FakeService()))

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertEqual(response.json()["model"], "fake")

    def test_predict_without_threshold_is_unscored(self):
        response = self.client.post(
            "/predict",
            data={"category": "bottle"},
            files={"file": ("part.png", _png_bytes(), "image/png")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["decision"], "unscored")
        self.assertFalse(body["alerted"])
        self.assertEqual(body["image_name"], "part.png")

    def test_predict_over_threshold_is_a_defect(self):
        response = self.client.post(
            "/predict",
            data={"category": "bottle", "threshold": "0.1"},
            files={"file": ("part.png", _png_bytes(), "image/png")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["decision"], "defect")
        self.assertTrue(response.json()["alerted"])

    def test_predict_rejects_a_non_image(self):
        response = self.client.post(
            "/predict",
            data={"category": "bottle"},
            files={"file": ("notes.txt", b"not an image", "text/plain")},
        )
        self.assertEqual(response.status_code, 400)

    def test_search(self):
        response = self.client.post("/search", json={"category": "metal_nut", "query": "a scratch", "k": 3})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["hits"][0]["defect"], "scratch")

    def test_search_without_an_index(self):
        response = self.client.post("/search", json={"category": "missing", "query": "a scratch"})
        self.assertEqual(response.status_code, 409)


if __name__ == "__main__":
    unittest.main()
