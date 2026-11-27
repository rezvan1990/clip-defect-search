"""Figures written to assets/."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Fixed before looking at results. A mix of objects, a thin structure, and textures.
FIGURE_CATEGORIES = ["bottle", "cable", "capsule", "carpet", "metal_nut", "tile"]

NAVY = "#1f4e79"
RED = "#a33b32"
TEAL = "#1d6a6a"
GOLD = "#c48a1a"


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "regular",
        }
    )


def save(fig, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def barh(
    labels: list[str],
    values: list[float],
    xlabel: str,
    title: str,
    path: Path | str,
    xlim=(0.0, 1.0),
    show_mean: bool = True,
):
    _style()
    fig, ax = plt.subplots(figsize=(8.2, 0.38 * len(labels) + 1.4))
    y = np.arange(len(labels))
    ax.barh(y, values, color=NAVY, height=0.72)
    finite = [v for v in values if v is not None]
    if show_mean and finite:
        mean = float(np.mean(finite))
        ax.axvline(mean, color=RED, linestyle="--", linewidth=1.2)
        title = f"{title}   mean {mean:.3f}"
    ax.set_yticks(y, labels)
    ax.set_xlabel(xlabel)
    ax.set_xlim(*xlim)
    ax.set_title(title)
    ax.invert_yaxis()
    for i, value in enumerate(values):
        if value is None:
            continue
        ax.text(min(value + 0.01, xlim[1] - 0.02), i, f"{value:.3f}", va="center", fontsize=8, color="#222")
    save(fig, path)


def grouped_bars(group_labels: list[str], series: dict[str, list[float]], ylabel: str, title: str, path: Path | str):
    _style()
    names = list(series.keys())
    n_groups = len(group_labels)
    n_series = len(names)
    x = np.arange(n_groups)
    width = min(0.8 / max(n_series, 1), 0.22)
    fig, ax = plt.subplots(figsize=(max(8, 1.3 * n_groups + 2), 4.6))
    colors = [NAVY, TEAL, GOLD, RED, "#5b4b8a", "#3d6b4f"]
    for i, name in enumerate(names):
        offset = (i - (n_series - 1) / 2) * width
        vals = series[name]
        ax.bar(x + offset, vals, width=width, label=name, color=colors[i % len(colors)])
    ax.set_xticks(x, group_labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.legend(frameon=False, ncol=min(n_series, 3))
    save(fig, path)


def _thumbnail(path: str, side: int = 180) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((side, side))
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
        return np.asarray(canvas)


def _wrap(text: str, width: int = 28) -> str:
    words = text.split()
    lines, cur = [], ""
    for word in words:
        trial = word if not cur else cur + " " + word
        if len(trial) > width and cur:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def retrieval_grid(queries: list[dict], path: Path | str, k: int = 5) -> None:
    """queries: rows with query text and a `top` list of {path, relevant, defect, score}."""
    _style()
    n_rows = len(queries)
    fig = plt.figure(figsize=(k * 1.75 + 3.4, n_rows * 2.25))
    grid = fig.add_gridspec(n_rows, k + 1, width_ratios=[1.35] + [1] * k, wspace=0.12, hspace=0.45)
    for r, query in enumerate(queries):
        label_ax = fig.add_subplot(grid[r, 0])
        label_ax.axis("off")
        label_ax.text(
            0.0,
            0.5,
            _wrap(query["query"]),
            ha="left",
            va="center",
            fontsize=8,
            transform=label_ax.transAxes,
        )
        for c in range(k):
            ax = fig.add_subplot(grid[r, c + 1])
            item = query["top"][c]
            ax.imshow(_thumbnail(item["path"]))
            color = "#2e7d32" if item["relevant"] else "#b3261e"
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(3)
            ax.set_xticks([])
            ax.set_yticks([])
            label = item["defect"].replace("_", " ")
            ax.set_xlabel(f"{label}\n{item['score']:.2f}", fontsize=7.5)
    fig.suptitle("Top-5 retrieval. Green border: same defect type. Red border: something else.", fontsize=11)
    save(fig, path)


def _crop_for_display(image_path: str, preprocess) -> np.ndarray:
    """The 224 crop CLIP actually encoded, as an RGB uint8 array."""
    import torchvision.transforms as T

    with Image.open(image_path) as img:
        img = img.convert("RGB")
    ops = []
    for transform in preprocess.transforms:
        name = transform.__class__.__name__
        if name in {"Resize", "CenterCrop"}:
            ops.append(transform)
    ops.append(T.ToTensor())
    tensor = T.Compose(ops)(img)
    array = (tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    return array


def _overlay(image: np.ndarray, heat: np.ndarray) -> np.ndarray:
    heat = np.asarray(heat, dtype=np.float32)
    span = float(heat.max() - heat.min())
    norm = (heat - heat.min()) / span if span > 1e-8 else np.zeros_like(heat)
    color = plt.cm.magma(norm)[..., :3]
    mixed = 0.58 * (image.astype(np.float32) / 255.0) + 0.42 * color
    return (mixed * 255).clip(0, 255).astype(np.uint8)


def localization_grid(rows: list[dict], path: Path | str, suptitle: str | None = None) -> None:
    """Each row needs image, heat, mask (224 arrays) and a title."""
    _style()
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(7.4, n * 2.15))
    if n == 1:
        axes = np.expand_dims(axes, 0)
    headers = ["image (CLIP crop)", "text similarity", "ground-truth mask"]
    for c, header in enumerate(headers):
        axes[0, c].set_title(header, fontsize=10)
    for r, row in enumerate(rows):
        axes[r, 0].imshow(row["image"])
        axes[r, 1].imshow(_overlay(row["image"], row["heat"]))
        axes[r, 2].imshow(row["mask"], cmap="gray", vmin=0, vmax=1)
        for c in range(3):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(row["title"], fontsize=7.5)
    fig.suptitle(
        suptitle or "Patch similarity to the defect description, next to the mask.",
        fontsize=11,
    )
    save(fig, path)


def median_query_per_category(queries: list[dict], categories: list[str]) -> list[dict]:
    chosen = []
    for category in categories:
        rows = [q for q in queries if q["category"] == category and q.get("ap") is not None]
        if not rows:
            continue
        rows = sorted(rows, key=lambda q: q["ap"])
        chosen.append(rows[len(rows) // 2])
    return chosen


def median_image_per_category(image_rows_by_cat: dict, categories: list[str], key: str) -> list[dict]:
    chosen = []
    for category in categories:
        rows = [r for r in image_rows_by_cat.get(category, []) if r.get(key) is not None]
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: r[key])
        chosen.append(rows[len(rows) // 2])
    return chosen
