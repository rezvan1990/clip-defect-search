"""CLIP image, patch, and text embeddings.

ViT patch tokens are the transformer outputs (class token removed), passed
through the same linear projection as the pooled image embedding. That puts
every patch in the text space, which is what the similarity maps use.

For ResNet, the pooled image embedding still comes from CLIP's attention pool.
Patch vectors are a per-location linear readout (value projection, then the
output projection, plus the spatial positional embedding) so the map is not
washed out by the global attention pool.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


_CHECKED_MODELS: set[int] = set()


def load_clip(model_name: str, pretrained: str, device: torch.device):
    import open_clip

    # OpenAI CLIP was trained with QuickGELU. open_clip 4.x builds ViT-B with
    # plain GELU unless this flag is set, and then warns that the checkpoint
    # does not match the architecture.
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        force_quick_gelu=True,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    model.to(device)
    return model, preprocess, tokenizer


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _vit_forward(visual, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    previous = visual.output_tokens
    visual.output_tokens = True
    try:
        pooled, tokens = visual(images)
    finally:
        visual.output_tokens = previous
    if visual.proj is not None:
        tokens = tokens @ visual.proj
    if tokens.shape[-1] != pooled.shape[-1]:
        raise RuntimeError(f"token dim {tokens.shape[-1]} != image dim {pooled.shape[-1]}")
    return pooled, tokens


def _resnet_forward(visual, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}

    def _hook(_module, _inputs, output):
        captured["layer4"] = output

    handle = visual.layer4.register_forward_hook(_hook)
    try:
        pooled = visual(images)
    finally:
        handle.remove()
    spatial = captured["layer4"]
    n, channels, height, width = spatial.shape
    tokens = spatial.reshape(n, channels, height * width).permute(0, 2, 1)
    pool = visual.attnpool
    pos = pool.positional_embedding[1 : 1 + tokens.shape[1], :].to(dtype=tokens.dtype, device=tokens.device)
    if pos.shape[0] != tokens.shape[1]:
        raise RuntimeError(
            f"ResNet positional embedding length {pos.shape[0]} does not match the {tokens.shape[1]} spatial cells"
        )
    tokens = tokens + pos
    tokens = pool.c_proj(pool.v_proj(tokens))
    if tokens.shape[-1] != pooled.shape[-1]:
        raise RuntimeError(f"token dim {tokens.shape[-1]} != image dim {pooled.shape[-1]}")
    return pooled, tokens


def encode_image_tokens(model, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pooled image features and patch tokens, both unnormalized."""
    visual = model.visual
    if hasattr(visual, "transformer") and hasattr(visual, "conv1"):
        return _vit_forward(visual, images)
    if hasattr(visual, "layer4") and hasattr(visual, "attnpool"):
        return _resnet_forward(visual, images)
    raise TypeError(f"Unsupported visual tower: {type(visual)}")


def _check_matches_encode_image(model, images: torch.Tensor, pooled: torch.Tensor) -> None:
    key = id(model)
    if key in _CHECKED_MODELS:
        return
    with torch.inference_mode():
        reference = model.encode_image(images[:1])
    if not torch.allclose(reference, pooled[:1], atol=1e-4, rtol=1e-4):
        raise RuntimeError("Patch encoder pooled features do not match model.encode_image")
    _CHECKED_MODELS.add(key)


def encode_images(model, preprocess, paths: list[str], device: torch.device, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a list of image paths.

    Returns float32 CPU tensors: image features [N, D] and patch tokens [N, P, D],
    both L2-normalized.
    """
    from PIL import Image
    from tqdm import tqdm

    image_feats = []
    patch_feats = []
    for start in tqdm(range(0, len(paths), batch_size), desc="encode images", leave=False):
        batch_paths = paths[start : start + batch_size]
        tensors = []
        for path in batch_paths:
            with Image.open(path) as img:
                tensors.append(preprocess(img.convert("RGB")))
        images = torch.stack(tensors, dim=0).to(device)
        with torch.inference_mode():
            pooled, tokens = encode_image_tokens(model, images)
            _check_matches_encode_image(model, images, pooled)
            pooled = _normalize(pooled).float().cpu()
            tokens = _normalize(tokens).float().cpu()
        image_feats.append(pooled)
        patch_feats.append(tokens)
    if not image_feats:
        raise RuntimeError("No images to encode")
    return torch.cat(image_feats, dim=0), torch.cat(patch_feats, dim=0)


def encode_texts(model, tokenizer, texts: list[str], device: torch.device, batch_size: int = 64) -> torch.Tensor:
    """Average L2-normalized text embeddings, then L2-normalize the average."""
    if not texts:
        raise ValueError("encode_texts received an empty prompt list")
    chunks = []
    for start in range(0, len(texts), batch_size):
        tokens = tokenizer(texts[start : start + batch_size]).to(device)
        with torch.inference_mode():
            feats = model.encode_text(tokens)
            feats = _normalize(feats).float().cpu()
        chunks.append(feats)
    proto = torch.cat(chunks, dim=0).mean(dim=0)
    proto = proto / proto.norm().clamp_min(1e-8)
    return proto


def mask_preprocess(image_preprocess):
    """Geometric half of the CLIP preprocess, nearest-neighbor, for masks."""
    import torchvision.transforms as T

    ops = []
    for transform in image_preprocess.transforms:
        name = transform.__class__.__name__
        if name == "Resize":
            ops.append(T.Resize(transform.size, interpolation=T.InterpolationMode.NEAREST))
        elif name == "CenterCrop":
            ops.append(T.CenterCrop(transform.size))
    if not ops:
        raise RuntimeError("CLIP preprocess has no Resize/CenterCrop; mask alignment is undefined")
    ops.append(T.ToTensor())
    return T.Compose(ops)


def load_aligned_mask(mask_path: str | None, mask_transform, size: int = 224) -> torch.Tensor:
    """Return a float mask in {0, 1} shaped [size, size], aligned with the CLIP crop."""
    import numpy as np
    from PIL import Image

    if mask_path is None:
        return torch.zeros(size, size, dtype=torch.float32)
    with Image.open(mask_path) as img:
        mask = img.convert("L")
    tensor = mask_transform(mask)
    if tensor.ndim == 3:
        tensor = tensor[0]
    # Some masks are 0/255, ToTensor scales those to 0/1. Threshold either way.
    binary = (tensor > 0).float()
    if binary.shape[-1] != size or binary.shape[-2] != size:
        binary = F.interpolate(binary[None, None], size=(size, size), mode="nearest")[0, 0]
    return binary


def upsample_map(score_grid, size: int = 224):
    import numpy as np

    grid = torch.tensor(np.asarray(score_grid), dtype=torch.float32)[None, None]
    up = F.interpolate(grid, size=(size, size), mode="bilinear", align_corners=False)
    return up[0, 0].numpy()
