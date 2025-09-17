import argparse
import os
from pathlib import Path
from shutil import copy2, rmtree
from typing import Dict, Iterable, List

import numpy as np
from PIL import Image

IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
LBL_EXTS = {".tif", ".tiff", ".png"}


def read_id_list(txt_path: Path) -> List[str]:
    ids: List[str] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            s = os.path.basename(s)
            ids.append(Path(s).stem)
    return ids


def scan_oem(root: Path) -> tuple[Dict[str, Path], Dict[str, Path]]:
    img_map, lbl_map = {}, {}
    for city in sorted([p for p in root.iterdir() if p.is_dir()]):
        img_dir = city / "images"
        lbl_dir = city / "labels"
        if img_dir.is_dir():
            for p in img_dir.iterdir():
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    img_map[p.stem] = p
        if lbl_dir.is_dir():
            for p in lbl_dir.iterdir():
                if p.is_file() and p.suffix.lower() in LBL_EXTS:
                    lbl_map[p.stem] = p
    return img_map, lbl_map


def ensure_empty_dir(d: Path):
    if d.exists():
        rmtree(d)
    d.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    copy2(str(src), str(dst))


def binarize_label_inplace(p: Path):
    arr = np.array(Image.open(p).convert("L"), dtype=np.uint8)
    out = (arr == 8).astype(np.uint8)
    Image.fromarray(out, mode="L").save(p)


def binarize_all_labels(label_dir: Path):
    files = [p for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() in LBL_EXTS]
    for p in files:
        try:
            binarize_label_inplace(p)
        except Exception as e:
            print(f"[WARN] binarize failed: {p} ({e})")


def map_by_stem(folder: Path, exts: Iterable[str]) -> Dict[str, Path]:
    m: Dict[str, Path] = {}
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            m[p.stem] = p
    return m


def crop_pair_512(img_path: Path, lbl_path: Path, out_img_dir: Path, out_lbl_dir: Path):
    img = Image.open(img_path).convert("RGB")
    lbl = Image.open(lbl_path).convert("L")

    def save_four(src_img: Image.Image, src_lbl: Image.Image, stem: str):
        for i in range(2):
            for j in range(2):
                box = (j * 512, i * 512, (j + 1) * 512, (i + 1) * 512)
                ci = src_img.crop(box)
                cl = src_lbl.crop(box)
                ci.save(out_img_dir / f"{stem}_crop_{i}_{j}.png")
                cl.save(out_lbl_dir / f"{stem}_crop_{i}_{j}.png")

    stem = img_path.stem

    if img.size == (1024, 1024):
        save_four(img, lbl, stem)
    elif img.size == (1000, 1000):
        pad_img = Image.new("RGB", (1024, 1024), (0, 0, 0))
        pad_lab = Image.new("L", (1024, 1024), 255)
        pad_img.paste(img, (24, 24))
        pad_lab.paste(lbl, (24, 24))
        save_four(pad_img, pad_lab, stem)
    elif img.size == (900, 900):
        pad_img = Image.new("RGB", (1024, 1024), (0, 0, 0))
        pad_lab = Image.new("L", (1024, 1024), 255)
        pad_img.paste(img, (124, 124))
        pad_lab.paste(lbl, (124, 124))
        save_four(pad_img, pad_lab, stem)
    elif img.size in {(650, 650), (438, 406), (439, 406)}:
        ri = img.resize((512, 512), Image.LANCZOS)
        rl = lbl.resize((512, 512), Image.NEAREST)
        ri.save(out_img_dir / f"{stem}.png")
        rl.save(out_lbl_dir / f"{stem}.png")
    else:
        print(f"[ERROR] unexpected size: {img.size} ({img_path})")


def run_crop_512(image_dir: Path, label_dir: Path, out_img_dir: Path, out_lbl_dir: Path):
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    img_map = map_by_stem(image_dir, IMG_EXTS)
    lbl_map = map_by_stem(label_dir, LBL_EXTS)
    common = sorted(set(img_map.keys()) & set(lbl_map.keys()))
    miss_img = sorted(set(lbl_map.keys()) - set(img_map.keys()))
    miss_lbl = sorted(set(img_map.keys()) - set(lbl_map.keys()))
    if miss_img:
        print(f"[WARN] labels without image: {len(miss_img)} (e.g., {miss_img[:5]})")
    if miss_lbl:
        print(f"[WARN] images without label: {len(miss_lbl)} (e.g., {miss_lbl[:5]})")
    for k in common:
        try:
            crop_pair_512(img_map[k], lbl_map[k], out_img_dir, out_lbl_dir)
        except Exception as e:
            print(f"[WARN] crop failed: {k} ({e})")


def main():
    ap = argparse.ArgumentParser(
        description="Prepare OEM trainval set, binarize labels (id==8->1 else 0), crop to 512 PNG."
    )
    ap.add_argument("--oem-root", type=Path, required=True,
                    help="OEM root containing train.txt/val.txt and city subfolders.")
    ap.add_argument("--out-base", type=Path, default=Path("./data/oem"),
                    help="Output base directory.")
    args = ap.parse_args()

    oem_root: Path = args.oem_root
    out_base: Path = args.out_base

    # locate lists
    train_txt = oem_root / "train.txt"
    val_txt_candidates = [
        oem_root / "val.txt",
        oem_root / "valid.txt",
        oem_root / "validation.txt",
        oem_root / "test.txt",
    ]
    assert train_txt.exists(), f"Missing {train_txt}"
    val_txt = next((p for p in val_txt_candidates if p.exists()), None)
    assert val_txt is not None, "Missing val.txt (or valid.txt/validation.txt/test.txt)"

    # dirs
    inter_img = out_base / "trainval" / "images"
    inter_lbl = out_base / "trainval" / "labels01"
    final_img = out_base / "trainval" / "image"
    final_lbl = out_base / "trainval" / "label01"

    # always clean
    for d in [inter_img, inter_lbl, final_img, final_lbl]:
        ensure_empty_dir(d)

    ids = sorted(set(read_id_list(train_txt)) | set(read_id_list(val_txt)))
    print(f"[INFO] ids (trainval): {len(ids)}")

    img_map, lbl_map = scan_oem(oem_root)
    print(f"[INFO] scanned: images={len(img_map)}, labels={len(lbl_map)}")

    n_img = n_lbl = 0
    miss_img, miss_lbl = [], []
    for s in ids:
        if s in img_map:
            copy_file(img_map[s], inter_img / img_map[s].name)
            n_img += 1
        else:
            miss_img.append(s)
        if s in lbl_map:
            copy_file(lbl_map[s], inter_lbl / lbl_map[s].name)
            n_lbl += 1
        else:
            miss_lbl.append(s)

    print(f"[OK] collected (intermediate) images={n_img}, labels={n_lbl}")
    if miss_img:
        print(f"[WARN] missing images: {len(miss_img)} (e.g., {miss_img[:5]})")
    if miss_lbl:
        print(f"[WARN] missing labels: {len(miss_lbl)} (e.g., {miss_lbl[:5]})")

    print("[INFO] binarizing labels in trainval/labels01 ...")
    binarize_all_labels(inter_lbl)

    print("[INFO] cropping to 512 PNG into trainval/{image,label01} ...")
    run_crop_512(inter_img, inter_lbl, final_img, final_lbl)

    # remove intermediate
    if inter_img.exists():
        rmtree(inter_img)
        print(f"[CLEAN] removed: {inter_img}")
    if inter_lbl.exists():
        rmtree(inter_lbl)
        print(f"[CLEAN] removed: {inter_lbl}")

    print("[DONE]")
    print(f"Final outputs:\n  {final_img}\n  {final_lbl}")


if __name__ == "__main__":
    main()
