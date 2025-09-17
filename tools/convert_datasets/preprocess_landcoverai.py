import os
import argparse
from pathlib import Path
from shutil import copy2, move, rmtree
from PIL import Image

DEFAULT_TRAIN_NAME = [
    'M-33-20-D-c-4-2.tif', 'M-33-20-D-d-3-3.tif', 'M-33-32-B-b-4-4.tif', 'M-33-48-A-c-4-4.tif', 'M-33-7-A-d-2-3.tif',
    'M-33-7-A-d-3-2.tif', 'M-34-32-B-a-4-3.tif', 'M-34-32-B-b-1-3.tif', 'M-34-5-D-d-4-2.tif', 'M-34-51-C-b-2-1.tif',
    'M-34-51-C-d-4-1.tif', 'M-34-55-B-b-4-1.tif', 'M-34-56-A-b-1-4.tif', 'M-34-6-A-d-2-2.tif', 'M-34-65-D-a-4-4.tif',
    'M-34-65-D-c-4-2.tif', 'M-34-65-D-d-4-1.tif', 'M-34-68-B-a-1-3.tif', 'M-34-77-B-c-2-3.tif', 'N-33-104-A-c-1-1.tif',
    'N-33-130-A-d-3-3.tif', 'N-33-130-A-d-4-4.tif', 'N-33-139-C-d-2-2.tif', 'N-33-139-C-d-2-4.tif', 'N-33-139-D-c-1-3.tif',
    'N-34-106-A-b-3-4.tif', 'N-34-106-A-c-1-3.tif', 'N-34-140-A-b-3-2.tif', 'N-34-140-A-b-4-2.tif', 'N-34-140-A-d-3-4.tif',
    'N-34-140-A-d-4-2.tif', 'N-34-77-A-b-1-4.tif', 'N-34-94-A-b-2-4.tif'
]

def ensure_empty_dir(p: Path):
    if p.exists():
        rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def list_files_by_stem(folder: Path, exts={".png", ".jpg", ".jpeg", ".tif", ".tiff"}):
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return {p.stem: p for p in files}

def copy_or_move(src: Path, dst: Path, do_move: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if do_move:
        move(str(src), str(dst))
    else:
        copy2(str(src), str(dst))

def crop_and_save_pair(image_path: Path, mask_path: Path, out_img_dir: Path, out_lbl_dir: Path, crop_size=512):
    image = Image.open(image_path)
    mask = Image.open(mask_path)
    if mask.mode not in ("L", "P"):
        mask = mask.convert("L")
    assert image.size == mask.size, f"Size mismatch: {image_path} vs {mask_path}"
    width, height = image.size
    crop_id = 0
    for top in range(0, height, crop_size):
        for left in range(0, width, crop_size):
            right = min(left + crop_size, width)
            bottom = min(top + crop_size, height)
            cropped_image = image.crop((left, top, right, bottom))
            cropped_mask = mask.crop((left, top, right, bottom))
            if cropped_image.size != (crop_size, crop_size):
                pad_img = Image.new(image.mode, (crop_size, crop_size), 0)
                pad_img.paste(cropped_image, (0, 0))
                cropped_image = pad_img
                pad_msk = Image.new(mask.mode, (crop_size, crop_size), 255)
                pad_msk.paste(cropped_mask, (0, 0))
                cropped_mask = pad_msk
            stem = image_path.stem
            img_name = f"{stem}_crop_{crop_id}.png"
            mask_name = f"{mask_path.stem}_crop_{crop_id}.png"
            cropped_image.save(out_img_dir / img_name)
            cropped_mask.save(out_lbl_dir / mask_name)
            crop_id += 1

def process_dir_pairs(images_dir: Path, masks_dir: Path, out_img_dir: Path, out_lbl_dir: Path, crop_size=512):
    img_map = list_files_by_stem(images_dir)
    msk_map = list_files_by_stem(masks_dir)
    common = sorted(set(img_map.keys()) & set(msk_map.keys()))
    ensure_empty_dir(out_img_dir)
    ensure_empty_dir(out_lbl_dir)
    total_pairs = 0
    for stem in common:
        crop_and_save_pair(img_map[stem], msk_map[stem], out_img_dir, out_lbl_dir, crop_size=crop_size)
        total_pairs += 1
    print(f"[OK] {images_dir} cropped: {total_pairs} pairs.")

def main():
    parser = argparse.ArgumentParser(
        description="Split by a train list (built-in), copy/move to out_base/landcoverai025 and out_base/landcoverai05, crop to PNG."
    )
    parser.add_argument("--in-base", required=True, type=Path, help="Input base dir containing images/ and masks/")
    parser.add_argument("--out-base", type=Path, default=Path("./data/landcoverai"), help="Output base dir")
    parser.add_argument("--crop-size", type=int, default=512, help="Crop size")
    parser.add_argument("--move", action="store_true", help="Move instead of copy")
    args = parser.parse_args()

    in_base = args.in_base
    out_base = args.out_base
    images_dir = in_base / "images"
    masks_dir = in_base / "masks"
    train_dir = out_base / "landcoverai025"
    test_dir = out_base / "landcoverai05"
    do_move = args.move
    crop = args.crop_size

    assert images_dir.exists() and masks_dir.exists(), f"Missing {images_dir} or {masks_dir}"

    all_images = list_files_by_stem(images_dir)
    all_masks = list_files_by_stem(masks_dir)
    all_common = sorted(set(all_images.keys()) & set(all_masks.keys()))
    print(f"[INFO] Paired samples: {len(all_common)}")

    names = DEFAULT_TRAIN_NAME
    print(f"[INFO] Using built-in train list: {len(names)} items")

    train_stems = [Path(n).stem for n in names]
    train_set = sorted([s for s in train_stems if s in all_common])
    test_set = sorted([s for s in all_common if s not in train_set])

    missing = sorted(set(train_stems) - set(train_set))
    if missing:
        print(f"[WARN] Missing in dataset (first 10): {missing[:10]}")

    print(f"[INFO] Train: {len(train_set)}, Test: {len(test_set)}")

    for sub in ["images", "labels"]:
        (train_dir / sub).mkdir(parents=True, exist_ok=True)
        (test_dir / sub).mkdir(parents=True, exist_ok=True)

    n_train = 0
    for stem in train_set:
        copy_or_move(all_images[stem], train_dir / "images" / all_images[stem].name, do_move)
        copy_or_move(all_masks[stem], train_dir / "labels" / all_masks[stem].name, do_move)
        n_train += 1

    n_test = 0
    for stem in test_set:
        copy_or_move(all_images[stem], test_dir / "images" / all_images[stem].name, do_move)
        copy_or_move(all_masks[stem], test_dir / "labels" / all_masks[stem].name, do_move)
        n_test += 1

    print(f"[OK] Copied/moved: landcoverai025={n_train}, landcoverai05={n_test} (move={do_move})")

    out_train_img = train_dir / "image"
    out_train_lbl = train_dir / "label"
    out_test_img = test_dir / "image"
    out_test_lbl = test_dir / "label"

    process_dir_pairs(train_dir / "images", train_dir / "labels", out_train_img, out_train_lbl, crop_size=crop)
    process_dir_pairs(test_dir / "images", test_dir / "labels", out_test_img, out_test_lbl, crop_size=crop)

    for p in [train_dir / "images", train_dir / "labels", test_dir / "images", test_dir / "labels"]:
        if p.exists():
            rmtree(p)
            print(f"[CLEAN] Removed: {p}")

    print("[DONE]")
    print(f"Outputs:\n  - {out_train_img}\n  - {out_train_lbl}\n  - {out_test_img}\n  - {out_test_lbl}")

if __name__ == "__main__":
    main()

