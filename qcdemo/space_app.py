"""Upload demo that does not need the MVTec image files.

Hugging Face Spaces imports the root app.py, which builds this UI. Category
names are the MVTec list; they only fill the prompt ("a bottle", "a metal nut").
The first click loads CLIP.
"""

from __future__ import annotations

import os

from defectclip.data import MVTEC_CATEGORIES


def _crop_uint8(preprocess, image):
    import numpy as np
    import torchvision.transforms as T

    ops = [step for step in preprocess.transforms if step.__class__.__name__ in {"Resize", "CenterCrop"}]
    array = T.Compose([*ops, T.ToTensor()])(image.convert("RGB")).numpy()
    return np.clip(np.transpose(array, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)


def build_upload_demo():
    import gradio as gr
    import torch
    from PIL import Image

    from defectclip.encoder import encode_image_tokens, encode_texts, load_clip
    from defectclip.localize import similarity_grid
    from defectclip.plots import _overlay
    from defectclip.prompts import object_phrase, retrieval_query_text
    from defectclip.zeroshot import category_prototypes, score_images

    state: dict = {}

    def bundle():
        if "model" not in state:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, preprocess, tokenizer = load_clip(
                os.environ.get("QC_MODEL", "ViT-B-16"),
                os.environ.get("QC_PRETRAINED", "openai"),
                device,
            )
            state.update(model=model, preprocess=preprocess, tokenizer=tokenizer, device=device, protos={})
        return state

    def score(image, category: str, query: str):
        if image is None:
            return None, "Upload an image first."
        held = bundle()
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        tensor = held["preprocess"](image.convert("RGB")).unsqueeze(0).to(held["device"])
        with torch.inference_mode():
            pooled, tokens = encode_image_tokens(held["model"], tensor)
            pooled = pooled / pooled.norm(dim=-1, keepdim=True)
            tokens = tokens / tokens.norm(dim=-1, keepdim=True)
        if category not in held["protos"]:
            held["protos"][category] = category_prototypes(
                held["model"],
                held["tokenizer"],
                category,
                held["device"],
                variant="ensemble",
            )
        normal, anomaly, _count = held["protos"][category]
        gap = float(score_images(pooled.float().cpu().numpy(), normal, anomaly)[0])
        text = (query or "").strip() or retrieval_query_text(category, "scratch")
        proto = encode_texts(held["model"], held["tokenizer"], [text], held["device"]).numpy()
        heat = similarity_grid(tokens[0].float().cpu().numpy(), proto, windows=(1,))
        overlay = Image.fromarray(_overlay(_crop_uint8(held["preprocess"], image), heat))
        note = (
            f"{category} ({object_phrase(category)})\n"
            f"zero-shot anomaly gap: {gap:.4f}\n"
            f"query: {text}\n"
            "The gap is not a calibrated probability, and this page does not send alerts. "
            "POST /predict does that when a threshold is set."
        )
        return overlay, note

    with gr.Blocks(title="CLIP defect check") as demo:
        gr.Markdown(
            """
# CLIP defect check

Upload one photo. The score is a frozen CLIP prompt ensemble (normal vs damaged)
for the product name you pick. The picture on the right is patch similarity to
the sentence in the query box. This page does not ship the MVTec gallery; the
retrieval numbers are in the repository README.
            """.strip()
        )
        with gr.Row():
            image = gr.Image(type="pil", label="Photo")
            heat = gr.Image(type="pil", label="Patch similarity")
        category = gr.Dropdown(MVTEC_CATEGORIES, value="metal_nut", label="Product")
        query = gr.Textbox(value="a scratch on the metal nut", label="Sentence for the map")
        button = gr.Button("Score")
        note = gr.Textbox(label="Score", lines=6)
        button.click(score, inputs=[image, category, query], outputs=[heat, note])
    return demo
