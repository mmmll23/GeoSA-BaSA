import argparse
import shutil
from pathlib import Path
from zipfile import ZipFile
import os
import numpy as np
import rasterio
from PIL import Image

EXCLUDE_TILE_IDS = {"D004_2021", "D014_2020", "D029_2021", "D031_2019", "D058_2020", "D066_2021", "D067_2021", "D077_2021"}

def ensure_empty_dir(d: Path):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

def unzip_all(zip_dir: Path, workdir: Path):
    workdir.mkdir(parents=True, exist_ok=True)
    zips = list(zip_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No .zip files found in {zip_dir}")
    for z in zips:
        print(f"[INFO] Extracting: {z.name}")
        with ZipFile(z, "r") as f:
            f.extractall(workdir)

def strip_prefix(name: str, prefixes=("IMG_", "MSK_")) -> str:
    for p in prefixes:
        if name.startswith(p):
            return name[len(p):]
    return name

def ensure_unique(dst_dir: Path, base_name: str) -> Path:
    stem = Path(base_name).stem
    suffix = Path(base_name).suffix
    out = dst_dir / (stem + suffix)
    i = 1
    while out.exists():
        out = dst_dir / f"{stem}_{i}{suffix}"
        i += 1
    return out

def read_rio(path: Path):
    with rasterio.open(path) as src:
        arr = src.read()
    return arr

def select_rgb_from_5band(img_arr: np.ndarray) -> np.ndarray:
    if img_arr.ndim != 3:
        raise ValueError("Image array must be (C,H,W)")
    rgb = img_arr[0:3, ...]
    return np.transpose(rgb, (1, 2, 0))

def clamp_labels_gt12_to0(lbl: np.ndarray) -> np.ndarray:
    if lbl.ndim == 2:
        a = lbl
    elif lbl.ndim == 3 and lbl.shape[0] == 1:
        a = lbl[0]
    else:
        raise ValueError("Label must be single-band (1,H,W) or (H,W).")
    out = a.copy()
    out[out > 12] = 0
    return out.astype(np.uint8)

def find_root_dir(tmp_dir: Path, key: str) -> Path:
    candidates = []
    for p in tmp_dir.rglob("*"):
        if p.is_dir() and key.lower() in p.name.lower():
            candidates.append(p)
    if not candidates:
        for p in tmp_dir.iterdir():
            if p.is_dir() and key.lower() in str(p).lower():
                candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f"Cannot locate directory containing: {key}")
    candidates.sort(key=lambda x: len(x.parts))
    return candidates[0]

def collect_files(root: Path, exts=(".tif", ".tiff")):
    files = []
    for ext in exts:
        files.extend(root.rglob(f"*{ext}"))
    return files

def prune_excluded_dirs(root: Path):
    removed = 0
    for d in list(root.rglob("*")):
        if d.is_dir() and d.name in EXCLUDE_TILE_IDS:
            print(f"[INFO] Removing excluded dir: {d}")
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
    print(f"[INFO] Pruned {removed} excluded directories under: {root}")

def process_split_images(src_root: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = collect_files(src_root)
    print(f"[INFO] Found {len(files)} image files under: {src_root}")
    for fp in files:
        try:
            arr = read_rio(fp)
            rgb = select_rgb_from_5band(arr)
            out_name = strip_prefix(fp.name, prefixes=("IMG_",))
            out_name = Path(out_name).with_suffix(".png").name
            out_path = ensure_unique(dst_dir, out_name)
            Image.fromarray(rgb.astype(np.uint8)).save(out_path)
        except Exception as e:
            print(f"[WARN] Skip image {fp}: {e}")

def process_split_labels(src_root: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = collect_files(src_root)
    print(f"[INFO] Found {len(files)} label files under: {src_root}")
    for fp in files:
        try:
            arr = read_rio(fp)
            if arr.ndim == 3 and arr.shape[0] > 1:
                arr = arr[0:1, ...]
            lbl = clamp_labels_gt12_to0(arr)
            out_name = strip_prefix(fp.name, prefixes=("MSK_",))
            out_name = Path(out_name).with_suffix(".png").name
            out_path = ensure_unique(dst_dir, out_name)
            Image.fromarray(lbl).save(out_path)
        except Exception as e:
            print(f"[WARN] Skip label {fp}: {e}")

def main():
    parser = argparse.ArgumentParser(description="FLAIR dataset preprocessor (PNG output)")
    parser.add_argument("--zip-dir", type=Path, required=True, help="Directory containing the 4 FLAIR zip files")
    parser.add_argument("--out-root", type=Path, default=Path("./data"), help="Output root (will create data/flair/...)")
    args = parser.parse_args()

    out_root = args.out_root
    flair_root = out_root / "flair"
    train_img_out = flair_root / "train" / "image"
    train_lbl_out = flair_root / "train" / "label"
    test_img_out = flair_root / "test" / "image"
    test_lbl_out = flair_root / "test" / "label"

    # Make sure output base exists and is writable
    flair_root.mkdir(parents=True, exist_ok=True)
    assert os.access(str(flair_root), os.W_OK), f"{flair_root} is not writable"

    # Use output folder as extraction workspace
    extract_dir = flair_root / "_extract"
    ensure_empty_dir(extract_dir)

    # Unzip into out_root/flair/_extract
    unzip_all(args.zip_dir, extract_dir)

    # Locate actual folders inside extraction area
    aerial_train_dir = find_root_dir(extract_dir, "flair_aerial_train")
    aerial_test_dir  = find_root_dir(extract_dir, "flair_1_aerial_test")
    labels_train_dir = find_root_dir(extract_dir, "flair_labels_train")
    labels_test_dir  = find_root_dir(extract_dir, "flair_1_labels_test")

    print(f"[INFO] Aerial train dir: {aerial_train_dir}")
    print(f"[INFO] Aerial test dir : {aerial_test_dir}")
    print(f"[INFO] Labels train dir: {labels_train_dir}")
    print(f"[INFO] Labels test dir : {labels_test_dir}")

    prune_excluded_dirs(aerial_train_dir)
    prune_excluded_dirs(labels_train_dir)

    process_split_images(aerial_train_dir, train_img_out)
    process_split_images(aerial_test_dir,  test_img_out)
    process_split_labels(labels_train_dir, train_lbl_out)
    process_split_labels(labels_test_dir,  test_lbl_out)

    # Clean extraction workspace under output root
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
        print(f"[CLEAN] removed temp extraction: {extract_dir}")

    print("\n[OK] Done. Final structure:")
    print(f"  {flair_root}/train/image")
    print(f"  {flair_root}/train/label")
    print(f"  {flair_root}/test/image")
    print(f"  {flair_root}/test/label")

if __name__ == "__main__":
    main()
