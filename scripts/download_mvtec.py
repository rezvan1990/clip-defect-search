#!/usr/bin/env python3
"""Download and extract MVTec AD.

The official MVTec page currently asks for a form, and the old direct links
return HTTP 404 (checked November 2026). This script downloads a public copy
of the same `mvtec_anomaly_detection.tar.xz` archive (5,264,982,680 bytes)
and extracts it under data/mvtec. The archive is not committed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from defectclip.data import MVTEC_CATEGORIES, find_mvtec_root

ARCHIVE_URL = (
    "https://huggingface.co/datasets/ProgrammerGnome/MVTecAD/resolve/main/mvtec_anomaly_detection.tar.xz"
)
EXPECTED_BYTES = 5264982680


def _has_dataset(path: Path) -> bool:
    try:
        find_mvtec_root(path)
        return True
    except FileNotFoundError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MVTec AD into data/mvtec.")
    parser.add_argument("--dest", default="data/mvtec")
    parser.add_argument("--url", default=ARCHIVE_URL)
    args = parser.parse_args()

    dest = Path(args.dest)
    if not dest.is_absolute():
        dest = ROOT / dest
    if _has_dataset(dest):
        root = find_mvtec_root(dest)
        missing = [c for c in MVTEC_CATEGORIES if not (root / c).is_dir()]
        if missing:
            raise SystemExit(f"MVTec root {root} is missing categories: {missing}")
        print(f"MVTec AD already present at {root}")
        return

    archive = dest.parent / "mvtec_anomaly_detection.tar.xz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists() or archive.stat().st_size != EXPECTED_BYTES:
        print(f"Downloading {args.url}")
        subprocess.run(
            ["wget", "-c", "-O", str(archive), args.url],
            check=True,
        )
    if archive.stat().st_size != EXPECTED_BYTES:
        raise SystemExit(
            f"Expected {EXPECTED_BYTES} bytes, got {archive.stat().st_size}. "
            "The download looks truncated; rerun to continue it."
        )

    extract_into = dest.parent / "_mvtec_extract"
    extract_into.mkdir(parents=True, exist_ok=True)
    print(f"Extracting into {extract_into}")
    subprocess.run(["tar", "-xJf", str(archive), "-C", str(extract_into)], check=True)
    found = find_mvtec_root(extract_into)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        if dest.is_symlink() or dest.is_file():
            dest.unlink()
        elif dest.is_dir() and not any(dest.iterdir()):
            dest.rmdir()
        elif _has_dataset(dest):
            print(f"MVTec AD already present at {find_mvtec_root(dest)}")
            return
        else:
            raise SystemExit(f"Refusing to replace {dest}")
    if found.resolve() == dest.resolve():
        pass
    else:
        dest.symlink_to(found.resolve(), target_is_directory=True)
    root = find_mvtec_root(dest)
    missing = [c for c in MVTEC_CATEGORIES if not (root / c).is_dir()]
    if missing:
        raise SystemExit(f"Extracted archive is missing categories: {missing}")
    print(f"MVTec AD ready at {root}")


if __name__ == "__main__":
    main()
