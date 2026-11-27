"""FastAPI entrypoint.

`create_app` can take a fake service so tests never download CLIP. The
process entrypoint (`app`) loads the real model on startup.
"""

from __future__ import annotations

import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from PIL import Image
from pydantic import BaseModel, Field

from qcapi.errors import ServiceError


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "service", None) is None:
        from qcapi.clip_service import ClipInspectionService

        app.state.service = ClipInspectionService.from_env()
    yield


def create_app(service=None) -> FastAPI:
    app = FastAPI(title="CLIP defect QC", version="0.1.0", lifespan=lifespan)
    app.state.service = service

    @app.get("/health")
    def health(request: Request):
        return request.app.state.service.health()

    @app.post("/predict")
    async def predict(
        request: Request,
        file: UploadFile = File(...),
        category: str = Form(...),
        threshold: float | None = Form(None),
    ):
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="empty upload")
        try:
            image = Image.open(io.BytesIO(raw))
            image.load()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail="could not read that file as an image") from exc
        try:
            return request.app.state.service.predict(
                image,
                category,
                threshold=threshold,
                image_name=file.filename or "upload",
            )
        except ServiceError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    @app.post("/search")
    def search(body: SearchRequest, request: Request):
        try:
            return request.app.state.service.search(body.category, body.query, body.k)
        except ServiceError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    return app


class SearchRequest(BaseModel):
    category: str
    query: str
    k: int = Field(default=5, ge=1, le=50)


app = create_app()
