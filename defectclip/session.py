"""Load the dataset, the feature cache, and a CLIP model for one experiment."""

from __future__ import annotations

import torch

from defectclip.cache import ensure_features, load_model_bundle
from defectclip.data import find_mvtec_root, load_mvtec
from defectclip.util import pick_device, set_seed


def start(args, split: str, with_patches: bool):
    set_seed(args.seed)
    torch.set_num_threads(max(torch.get_num_threads(), 1))
    device = pick_device(args.device)
    root = find_mvtec_root(args.data).resolve()
    records = load_mvtec(root, split=split, categories=args.categories)
    print(f"{split}: {len(records)} images on {device} ({args.model}/{args.pretrained})")
    table = ensure_features(
        records,
        split=split,
        model_name=args.model,
        pretrained=args.pretrained,
        device=device,
        batch_size=args.batch_size,
        cache_root=args.cache,
        with_patches=with_patches,
    )
    model, preprocess, tokenizer, mask_transform = load_model_bundle(args.model, args.pretrained, device)
    return {
        "device": device,
        "root": root,
        "records": records,
        "table": table,
        "model": model,
        "preprocess": preprocess,
        "tokenizer": tokenizer,
        "mask_transform": mask_transform,
    }
