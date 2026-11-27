"""CLIP-backed inspection service used by the API process.

Unit tests do not import this module. They pass a fake service into create_app.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from defectclip.alerts import AlertRouter, decision_for
from defectclip.cache import _load_table, model_dir
from defectclip.data import MVTEC_CATEGORIES
from defectclip.encoder import encode_texts, load_clip
from defectclip.inspection_log import InspectionLog
from defectclip.zeroshot import category_prototypes, score_images
from qcapi.errors import ServiceError


class ClipInspectionService:
    def __init__(
        self,
        model_name: str,
        pretrained: str,
        device: torch.device,
        log: InspectionLog,
        alerts: AlertRouter,
        gallery: dict | None,
        default_threshold: float | None,
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self.log = log
        self.alerts = alerts
        self.default_threshold = default_threshold
        self.model, self.preprocess, self.tokenizer = load_clip(model_name, pretrained, device)
        self._prototypes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.gallery = gallery
        self._by_category: dict[str, np.ndarray] = {}
        if gallery is not None:
            for category in sorted(set(gallery["category"].tolist())):
                self._by_category[category] = np.flatnonzero(gallery["category"] == category)

    @classmethod
    def from_env(cls) -> "ClipInspectionService":
        data_dir = Path(os.environ.get("QC_DATA_DIR", "var/qc"))
        model_name = os.environ.get("QC_MODEL", "ViT-B-16")
        pretrained = os.environ.get("QC_PRETRAINED", "openai")
        device_name = os.environ.get("QC_DEVICE", "cpu")
        if device_name == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_name)
        log = InspectionLog(data_dir / "inspections.db")
        alerts = AlertRouter(log, data_dir / "alerts.jsonl")
        cache_root = os.environ.get("QC_CACHE", "cache")
        gallery_file = model_dir(cache_root, model_name, pretrained) / "test_images.npz"
        gallery = _load_table(gallery_file) if gallery_file.is_file() else None
        raw_threshold = os.environ.get("ALERT_SCORE_THRESHOLD", "").strip()
        threshold = float(raw_threshold) if raw_threshold else None
        return cls(model_name, pretrained, device, log, alerts, gallery, threshold)

    def health(self) -> dict:
        return {
            "status": "ok",
            "model": self.model_name,
            "pretrained": self.pretrained,
            "device": str(self.device),
            "gallery_indexed": self.gallery is not None,
            "categories": MVTEC_CATEGORIES,
            "alerts_armed": self.default_threshold is not None,
            "default_threshold": self.default_threshold,
        }

    def _protos(self, category: str) -> tuple[np.ndarray, np.ndarray]:
        if category not in self._prototypes:
            normal, anomaly, _count = category_prototypes(
                self.model,
                self.tokenizer,
                category,
                self.device,
                variant="ensemble",
            )
            self._prototypes[category] = (normal, anomaly)
        return self._prototypes[category]

    def predict(
        self,
        image: Image.Image,
        category: str,
        threshold: float | None = None,
        image_name: str = "upload",
    ) -> dict:
        category = category.strip()
        if not category:
            raise ServiceError(400, "category is required")
        tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            features = self.model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        normal, anomaly = self._protos(category)
        score = float(score_images(features.float().cpu().numpy(), normal, anomaly)[0])
        used_threshold = self.default_threshold if threshold is None else threshold
        decision = decision_for(score, used_threshold)
        inspection_id = self.log.add_inspection(
            category=category,
            score=score,
            decision=decision,
            threshold=used_threshold,
            image_name=image_name,
            source="api",
            alerted=False,
        )
        alert = {"alerted": False, "channels": [], "errors": []}
        if decision == "defect":
            alert = self.alerts.send(
                {
                    "event": "defect",
                    "category": category,
                    "score": score,
                    "threshold": used_threshold,
                    "image_name": image_name,
                    "source": "api",
                    "inspection_id": inspection_id,
                },
                inspection_id=inspection_id,
            )
        return {
            "category": category,
            "score": score,
            "decision": decision,
            "threshold": used_threshold,
            "alerted": alert["alerted"],
            "alert_channels": alert["channels"],
            "alert_errors": alert["errors"],
            "model": self.model_name,
            "inspection_id": inspection_id,
            "note": (
                "Score is the cosine gap between the anomaly prompt prototype and the "
                "normal one. It is not a calibrated probability. No default threshold "
                "is shipped; set ALERT_SCORE_THRESHOLD or pass threshold to arm alerts."
            ),
        }

    def search(self, category: str, query: str, k: int) -> dict:
        if self.gallery is None:
            raise ServiceError(
                409,
                "No feature cache for this model. Run scripts/run_retrieval.py and point QC_CACHE at that cache.",
            )
        category = category.strip()
        query = query.strip()
        if not category or not query:
            raise ServiceError(400, "category and query are required")
        if category not in self._by_category:
            raise ServiceError(404, f"No indexed test images for {category}")
        k = max(1, min(int(k), 50))
        proto = encode_texts(self.model, self.tokenizer, [query], self.device).numpy()
        index = self._by_category[category]
        scores = self.gallery["features"][index] @ proto
        order = np.argsort(-scores, kind="mergesort")[:k]
        hits = []
        for rank, position in enumerate(order, start=1):
            row = int(index[int(position)])
            hits.append(
                {
                    "rank": rank,
                    "defect": str(self.gallery["defect"][row]),
                    "score": float(scores[int(position)]),
                    "path": str(self.gallery["path"][row]),
                }
            )
        return {"category": category, "query": query, "hits": hits}
