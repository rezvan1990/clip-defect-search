#!/usr/bin/env python3
"""Small Gradio demo for text-to-image defect search and a similarity map.

The gallery is the MVTec test split. Image embeddings are cached the first
time you run an experiment (or the first time this app starts). The heatmap
is computed for the one image you pick, so it does not need the patch cache.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from defectclip.cli import build_parser
from defectclip.data import MVTEC_CATEGORIES
from defectclip.encoder import encode_image_tokens, encode_texts
from defectclip.localize import similarity_grid
from defectclip.plots import _crop_for_display, _overlay
from defectclip.prompts import defect_phrase, object_phrase
from defectclip.session import start
from defectclip.zeroshot import category_prototypes, score_images


def _label(defect: str, path: str) -> str:
    return f"{defect} | {Path(path).name}"


class DemoIndex:
    def __init__(self, args):
        session = start(args, split="test", with_patches=False)
        self.device = session["device"]
        self.model = session["model"]
        self.tokenizer = session["tokenizer"]
        self.preprocess = session["preprocess"]
        self.records = session["records"]
        self.features = session["table"]["features"]
        self.by_cat = {}
        for category in MVTEC_CATEGORIES:
            index = [i for i, r in enumerate(self.records) if r.category == category]
            if not index:
                continue
            rows = [self.records[i] for i in index]
            labels = []
            # All anomalies, plus the first three good images so the dropdown stays short.
            good_kept = 0
            for row in rows:
                if row.is_anomaly or good_kept < 3:
                    labels.append(_label(row.defect, row.path))
                    if not row.is_anomaly:
                        good_kept += 1
            self.by_cat[category] = {
                "index": np.array(index),
                "features": self.features[index],
                "labels": labels,
                "lookup": {_label(r.defect, r.path): r for r in rows},
            }
        self.categories = [c for c in MVTEC_CATEGORIES if c in self.by_cat]
        self._prototypes = {}

    def prototypes(self, category: str):
        if category not in self._prototypes:
            names = sorted({r.defect for r in self.records if r.category == category and r.is_anomaly})
            self._prototypes[category] = category_prototypes(
                self.model, self.tokenizer, category, self.device, variant="ensemble", defect_names=names
            )
        return self._prototypes[category]

    def search(self, category: str, query: str, k: int):
        query = (query or "").strip()
        if not query:
            return []
        k = int(k)
        proto = encode_texts(self.model, self.tokenizer, [query], self.device).numpy()
        pack = self.by_cat[category]
        scores = pack["features"] @ proto
        order = np.argsort(-scores, kind="mergesort")[:k]
        gallery = []
        for j in order:
            record = self.records[pack["index"][j]]
            with Image.open(record.path) as img:
                photo = img.convert("RGB").copy()
            caption = f"{record.defect.replace('_', ' ')}   {scores[j]:.3f}"
            gallery.append((photo, caption))
        return gallery

    def choices(self, category: str):
        return self.by_cat[category]["labels"]

    def localize(self, category: str, image_label: str, query: str):
        query = (query or "").strip()
        record = self.by_cat[category]["lookup"][image_label]
        with Image.open(record.path) as img:
            tensor = self.preprocess(img.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            pooled, tokens = encode_image_tokens(self.model, tensor)
            pooled = pooled / pooled.norm(dim=-1, keepdim=True)
            tokens = tokens / tokens.norm(dim=-1, keepdim=True)
        proto = encode_texts(self.model, self.tokenizer, [query], self.device).numpy()
        heat = similarity_grid(tokens[0].float().cpu().numpy(), proto, windows=(1,))
        crop = _crop_for_display(record.path, self.preprocess)
        overlay = Image.fromarray(_overlay(crop, heat))
        normal, anomaly, _n = self.prototypes(category)
        # The cached row is the whole-category feature, already normalized.
        local_index = self.by_cat[category]["lookup"]
        # Recompute the gap from this forward so the number matches the pixels above.
        gap = float(score_images(pooled.float().cpu().numpy(), normal, anomaly)[0])
        patch_scores = tokens[0].float().cpu().numpy() @ proto
        cosine = float(patch_scores.max())
        text = (
            f"{category} / {record.defect.replace('_', ' ')}\n"
            f"query: {query}\n"
            f"zero-shot anomaly gap (ensemble): {gap:.4f}\n"
            f"peak patch cosine to the query: {cosine:.4f}\n"
            "A higher gap means the image embedding is closer to the anomaly "
            "prompts than to the normal prompts. It is not a calibrated probability."
        )
        del local_index
        return Image.fromarray(crop), overlay, text


def default_query(category: str) -> str:
    obj = object_phrase(category)
    return f"a scratch on the {obj}" if category == "metal_nut" else f"a defect on the {obj}"


def build_demo(index: DemoIndex):
    import gradio as gr

    first = "metal_nut" if "metal_nut" in index.categories else index.categories[0]
    example_rows = [
        row
        for row in [
            ["metal_nut", "a scratch on the metal nut", 8],
            ["carpet", "a hole in the carpet", 8],
            ["leather", "a scratch on the leather surface", 8],
            ["bottle", "a broken bottle", 8],
        ]
        if row[0] in index.categories
    ]
    with gr.Blocks(title="CLIP defect search") as demo:
        gr.Markdown(
            """
# CLIP defect search

Text-to-image retrieval and a patch similarity map on the MVTec AD test split.
The encoder is a frozen OpenAI CLIP model. Retrieval ranks that category's test
images by cosine similarity to whatever you type. The map is the same similarity
on patch tokens, drawn on the crop CLIP actually saw.
            """.strip()
        )
        with gr.Tab("Retrieve"):
            with gr.Row():
                category = gr.Dropdown(index.categories, value=first, label="Category")
                query = gr.Textbox(value="a scratch on the metal nut" if first == "metal_nut" else default_query(first), label="Query")
                k = gr.Slider(1, 12, value=8, step=1, label="Top k")
            search = gr.Button("Search")
            gallery = gr.Gallery(label="Most similar test images", columns=4, height=480, type="pil")
            search.click(index.search, inputs=[category, query, k], outputs=gallery)
            if example_rows:
                gr.Examples(
                    examples=example_rows,
                    inputs=[category, query, k],
                    fn=index.search,
                    outputs=gallery,
                    cache_examples=False,
                )
        with gr.Tab("Localize"):
            start_choices = index.choices(first)
            with gr.Row():
                loc_category = gr.Dropdown(index.categories, value=first, label="Category")
                loc_image = gr.Dropdown(start_choices, value=start_choices[0], label="Test image")
            loc_query = gr.Textbox(value=default_query(first), label="Defect description")
            loc_button = gr.Button("Map the description")
            with gr.Row():
                original = gr.Image(label="CLIP crop", type="pil")
                heat = gr.Image(label="Patch similarity", type="pil")
            caption = gr.Textbox(label="Scores", lines=5)

            def _refresh(cat):
                choices = index.choices(cat)
                # Prefer a real defect over the good images at the top of the list.
                value = next((c for c in choices if not c.startswith("good ")), choices[0])
                phrase = defect_phrase(value.split("|")[0].strip()) if "|" in value else "defect"
                return (
                    gr.Dropdown(choices=choices, value=value),
                    f"a {phrase} on the {object_phrase(cat)}",
                )

            loc_category.change(_refresh, inputs=loc_category, outputs=[loc_image, loc_query])
            loc_button.click(index.localize, inputs=[loc_category, loc_image, loc_query], outputs=[original, heat, caption])
    return demo


def main() -> None:
    parser = build_parser("Launch the defect-search demo.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    index = DemoIndex(args)
    demo = build_demo(index)
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
