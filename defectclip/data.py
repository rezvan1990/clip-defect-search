"""MVTec AD reader.

The dataset is the standard industrial anomaly benchmark: each category has
defect-free training images and a test set of good images plus several named
defect types, with pixel masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


@dataclass(frozen=True)
class Record:
    path: str
    category: str
    split: str
    defect: str
    is_anomaly: bool
    mask_path: str | None

    @property
    def rel_path(self) -> str:
        return self.path


def find_mvtec_root(path: Path | str) -> Path:
    path = Path(path)
    candidates = [path, path / "mvtec", path / "mvtec_anomaly_detection"]
    if path.is_dir():
        candidates.extend(p for p in path.iterdir() if p.is_dir())
    for cand in candidates:
        if (cand / "bottle" / "train" / "good").is_dir():
            return cand
    raise FileNotFoundError(
        f"Could not find an MVTec AD root under {path}. "
        "Run `python scripts/download_mvtec.py` first."
    )


def _iter_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    files = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    return sorted(files, key=lambda p: p.name)


def _mask_for(root: Path, category: str, defect: str, image_path: Path) -> Path | None:
    mask = root / category / "ground_truth" / defect / f"{image_path.stem}_mask{image_path.suffix}"
    if mask.is_file():
        return mask
    png = root / category / "ground_truth" / defect / f"{image_path.stem}_mask.png"
    if png.is_file():
        return png
    return None


def load_mvtec(
    root: Path | str,
    split: str | None = None,
    categories: list[str] | None = None,
) -> list[Record]:
    """Load records. `split` is 'train', 'test', or None for both."""
    root = find_mvtec_root(root)
    cats = categories or MVTEC_CATEGORIES
    unknown = [c for c in cats if c not in MVTEC_CATEGORIES]
    if unknown:
        raise ValueError(f"Unknown MVTec categories: {unknown}")
    records: list[Record] = []
    for category in cats:
        cat_dir = root / category
        if not cat_dir.is_dir():
            raise FileNotFoundError(f"Missing category directory: {cat_dir}")
        splits = ("train", "test") if split is None else (split,)
        for which in splits:
            split_dir = cat_dir / which
            if not split_dir.is_dir():
                continue
            for defect_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                defect = defect_dir.name
                is_anomaly = defect != "good"
                for image_path in _iter_images(defect_dir):
                    mask = None
                    if is_anomaly and which == "test":
                        found = _mask_for(root, category, defect, image_path)
                        mask = str(found) if found is not None else None
                    records.append(
                        Record(
                            path=str(image_path.resolve()),
                            category=category,
                            split=which,
                            defect=defect,
                            is_anomaly=is_anomaly,
                            mask_path=mask,
                        )
                    )
    if not records:
        raise RuntimeError(f"No images found under {root}")
    return records


def defect_types(records: list[Record], category: str) -> list[str]:
    names = {r.defect for r in records if r.category == category and r.is_anomaly}
    return sorted(names)


def category_slice(records: list[Record], category: str) -> list[Record]:
    return [r for r in records if r.category == category]
