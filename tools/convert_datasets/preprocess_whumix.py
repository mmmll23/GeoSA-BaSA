import argparse
import os
from pathlib import Path
from zipfile import ZipFile
from shutil import copy2, rmtree

import numpy as np
from PIL import Image

IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
LBL_EXTS = {".tif", ".tiff", ".png"}

CITIES = {"dunedin", "kitsap", "wuxi", "khartoum", "potsdam"}


def ensure_empty_dir(d: Path):
    if d.exists():
        rmtree(d)
    d.mkdir(parents=True, exist_ok=True)


def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)


def unzip(zip_path: Path, out_dir: Path):
    assert zip_path.exists(), f"Missing: {zip_path}"
    print(f"[INFO] Extracting: {zip_path.name}")
    with ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def find_subdir(root: Path, name: str) -> Path:
    cands = [p for p in root.rglob("*") if p.is_dir() and p.name.lower() == name.lower()]
    if not cands:
        raise FileNotFoundError(f"Cannot find subdir '{name}' under {root}")
    cands.sort(key=lambda p: len(p.parts))  # shallowest
    return cands[0]


def copy_all(src_dir: Path, dst_dir: Path, exts):
    ensure_dir(dst_dir)
    n = 0
    for p in src_dir.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            copy2(str(p), str(dst_dir / p.name))
            n += 1
    return n


def binarize_folder(src_dir: Path, dst_dir: Path):
    ensure_empty_dir(dst_dir)
    n = 0
    for p in src_dir.iterdir():
        if not p.is_file() or p.suffix.lower() not in LBL_EXTS:
            continue
        arr = np.array(Image.open(p).convert("L"), dtype=np.uint8)
        out = (arr > 0).astype(np.uint8)  # 0/255 -> 0/1
        Image.fromarray(out, mode="L").save(dst_dir / (p.stem + ".png"))
        n += 1
    return n


def split_city_from_name(fname: str) -> str | None:
    stem = Path(fname).stem
    city = stem.split("_", 1)[0].lower()
    return city if city in CITIES else None


def process_train(train_root: Path, out_base: Path):
    img_src = find_subdir(train_root, "image")
    lbl_src = find_subdir(train_root, "label")

    out_train = out_base / "train"
    out_img = out_train / "image"
    out_lbl = out_train / "label"
    out_lbl01 = out_train / "label01"

    ensure_empty_dir(out_img)
    ensure_empty_dir(out_lbl)
    ensure_dir(out_lbl01)

    n_img = copy_all(img_src, out_img, IMG_EXTS)
    n_lbl = copy_all(lbl_src, out_lbl, LBL_EXTS)
    print(f"[INFO] train copied: image={n_img}, label={n_lbl}")

    n_bin = binarize_folder(out_lbl, out_lbl01)
    print(f"[INFO] train binarized label -> label01: {n_bin}")

    if out_lbl.exists():
        rmtree(out_lbl)
        print(f"[CLEAN] removed: {out_lbl}")


def process_test(test_root: Path, out_base: Path):
    img_src = find_subdir(test_root, "image")
    lbl_src = find_subdir(test_root, "label")

    per_city_counters = {c: {"img": 0, "lbl": 0} for c in CITIES}
    unknown_images, unknown_labels = 0, 0

    for p in img_src.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            city = split_city_from_name(p.name)
            if city is None:
                unknown_images += 1
                continue
            dst = out_base / city / "image"
            ensure_dir(dst)
            copy2(str(p), str(dst / p.name))
            per_city_counters[city]["img"] += 1

    for p in lbl_src.iterdir():
        if p.is_file() and p.suffix.lower() in LBL_EXTS:
            city = split_city_from_name(p.name)
            if city is None:
                unknown_labels += 1
                continue
            dst = out_base / city / "label"
            ensure_dir(dst)
            copy2(str(p), str(dst / p.name))
            per_city_counters[city]["lbl"] += 1

    for city, cnt in per_city_counters.items():
        print(f"[INFO] test copied {city}: image={cnt['img']}, label={cnt['lbl']}")
    if unknown_images or unknown_labels:
        print(f"[WARN] skipped unknown-city files: image={unknown_images}, label={unknown_labels}")

    # binarize per-city labels -> label01, then remove label
    for city in CITIES:
        lbl_dir = out_base / city / "label"
        if not lbl_dir.exists():
            continue
        lbl01_dir = out_base / city / "label01"
        n_bin = binarize_folder(lbl_dir, lbl01_dir)
        print(f"[INFO] test binarized {city}: {n_bin}")
        rmtree(lbl_dir)
        print(f"[CLEAN] removed: {lbl_dir}")


def main():
    ap = argparse.ArgumentParser(description="Prepare whu-mix: unzip, copy, split test by city, binarize labels 0/255->0/1.")
    ap.add_argument("--input-dir", type=Path, required=True, help="Directory containing train.zip and test.zip")
    ap.add_argument("--out-base", type=Path, default=Path("./data/whumix"), help="Output base (default: ./data/whumix)")
    args = ap.parse_args()

    in_dir: Path = args.input_dir
    out_base: Path = args.out_base

    train_zip = in_dir / "train.zip"
    test_zip = in_dir / "test.zip"
    assert train_zip.exists() and test_zip.exists(), f"Missing train.zip or test.zip in {in_dir}"

    # extract under out_base to avoid /tmp or input disk space issues
    extract_dir = out_base / "_extract"
    ensure_empty_dir(extract_dir)

    unzip(train_zip, extract_dir)
    unzip(test_zip, extract_dir)

    train_root = find_subdir(extract_dir, "train")
    test_root = find_subdir(extract_dir, "test")

    ensure_dir(out_base)
    process_train(train_root, out_base)
    process_test(test_root, out_base)

    # cleanup extraction directory
    if extract_dir.exists():
        rmtree(extract_dir)
        print(f"[CLEAN] removed temp extraction: {extract_dir}")

    print("[DONE]")
    print(f"Train out: {out_base / 'train' / 'image'}  |  {out_base / 'train' / 'label01'}")
    for c in sorted(CITIES):
        print(f"Test out[{c}]: {out_base / c / 'image'}  |  {out_base / c / 'label01'}")


if __name__ == "__main__":
    main()
