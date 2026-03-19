"""
=============================================================
  Philippine ID Classifier - Dataset Preparation v2
  FIX: Split FIRST → Augment Training Set ONLY
  Classes: Passport vs National ID (PhilID)
=============================================================

CORRECT ORDER:
  1. Load original images
  2. Split into train / val / test (no augmentation yet)
  3. Augment ONLY the training split
  4. Val and Test sets stay as clean original images

This prevents data leakage where augmented copies of the
same image end up in both train and val/test sets.
=============================================================
"""

import os
import random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
SOURCE_PASSPORT        = r"C:\Users\Renzo\Documents\MindVision\passport"
SOURCE_PHILID          = r"C:\Users\Renzo\Documents\MindVision\nID"
SOURCE_DRIVERS_LICENSE = r"C:\Users\Renzo\Documents\MindVision\dL"
SOURCE_SSS             = r"C:\Users\Renzo\Documents\MindVision\sss"
SOURCE_PHILHEALTH      = r"C:\Users\Renzo\Documents\MindVision\philhealth"
SOURCE_SENIOR          = r"C:\Users\Renzo\Documents\MindVision\sID"
OUTPUT_DIR      = r"C:\Users\Renzo\Documents\MindVision\dataset"

IMG_SIZE        = (640, 640)
TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.15
TEST_RATIO      = 0.15
AUG_MULTIPLIER  = 5       # Augmented copies per TRAINING image only
RANDOM_SEED     = 42

CLASSES = {
    "passport":        SOURCE_PASSPORT,
    "philid":          SOURCE_PHILID,
    "drivers_license": SOURCE_DRIVERS_LICENSE,
    "sss":             SOURCE_SSS,
    "philhealth":      SOURCE_PHILHEALTH,
    "senior":          SOURCE_SENIOR,
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def load_image_paths(folder):
    paths = []
    for f in Path(folder).iterdir():
        if f.suffix.lower() in VALID_EXTENSIONS:
            paths.append(f)
    return paths


def preprocess(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    return img


def augment(img: Image.Image) -> Image.Image:
    img = img.copy()

    # Random rotation ±15 degrees
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, expand=False, fillcolor=(200, 200, 200))

    # Random brightness
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))

    # Random contrast
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))

    # Random blur (50% chance)
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

    # Random horizontal flip (50% chance)
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Random zoom/crop (50% chance)
    if random.random() < 0.5:
        crop_pct = random.uniform(0.05, 0.15)
        w, h = img.size
        left   = int(w * crop_pct)
        top    = int(h * crop_pct)
        right  = int(w * (1 - crop_pct))
        bottom = int(h * (1 - crop_pct))
        img = img.crop((left, top, right, bottom))
        img = img.resize(IMG_SIZE, Image.LANCZOS)

    return img


def split_paths(paths):
    """Split BEFORE any augmentation."""
    random.shuffle(paths)
    n         = len(paths)
    train_end = int(n * TRAIN_RATIO)
    val_end   = train_end + int(n * VAL_RATIO)
    train     = paths[:train_end]
    val       = paths[train_end:val_end]
    test      = paths[val_end:]
    return train, val, test


def save_image(img: Image.Image, folder: str, filename: str):
    os.makedirs(folder, exist_ok=True)
    img.save(os.path.join(folder, filename), "JPEG", quality=95)


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Philippine ID Classifier - Dataset Preparation v2")
    print("  ORDER: Split First → Augment Train Only")
    print("=" * 60)

    total_summary = {}

    for class_name, source_folder in CLASSES.items():
        print(f"\n📁 Processing class: [{class_name.upper()}]")
        print(f"   Source: {source_folder}")

        paths = load_image_paths(source_folder)
        if not paths:
            print(f"   ⚠️  No images found. Skipping.")
            continue
        print(f"   Found: {len(paths)} original images")

        # ── STEP 1: Split original images first ──
        train_paths, val_paths, test_paths = split_paths(paths)
        print(f"   Split (originals only):")
        print(f"     Train : {len(train_paths)} originals")
        print(f"     Val   : {len(val_paths)} originals (NO augmentation)")
        print(f"     Test  : {len(test_paths)} originals (NO augmentation)")

        counts = {"train": 0, "val": 0, "test": 0}

        # ── STEP 2: Save Val and Test as clean originals only ──
        for split_name, split_paths_list in [("val", val_paths), ("test", test_paths)]:
            out_folder = os.path.join(OUTPUT_DIR, split_name, class_name)
            for idx, img_path in enumerate(split_paths_list):
                try:
                    img = preprocess(Image.open(img_path))
                    save_image(img, out_folder, f"{class_name}_{split_name}_{idx:04d}.jpg")
                    counts[split_name] += 1
                except Exception as e:
                    print(f"   ⚠️  Skipping {img_path.name}: {e}")

        # ── STEP 3: Save Train originals + augmented copies ──
        out_folder = os.path.join(OUTPUT_DIR, "train", class_name)
        for idx, img_path in enumerate(train_paths):
            try:
                img = preprocess(Image.open(img_path))

                # Save original
                save_image(img, out_folder, f"{class_name}_train_{idx:04d}_orig.jpg")
                counts["train"] += 1

                # Save augmented copies (training only)
                for aug_i in range(AUG_MULTIPLIER):
                    aug_img = augment(img)
                    save_image(aug_img, out_folder, f"{class_name}_train_{idx:04d}_aug{aug_i}.jpg")
                    counts["train"] += 1

            except Exception as e:
                print(f"   ⚠️  Skipping {img_path.name}: {e}")

        total_summary[class_name] = counts
        print(f"\n   ✅ Done:")
        print(f"     Train : {counts['train']} images (originals + augmented)")
        print(f"     Val   : {counts['val']} images (clean originals only)")
        print(f"     Test  : {counts['test']} images (clean originals only)")

    # ── Final Summary ──
    print("\n" + "=" * 60)
    print("  DATASET PREPARATION v2 COMPLETE")
    print("=" * 60)
    for class_name, counts in total_summary.items():
        total = sum(counts.values())
        print(f"\n  [{class_name.upper()}]")
        print(f"    Train : {counts['train']} (augmented)")
        print(f"    Val   : {counts['val']}  (clean)")
        print(f"    Test  : {counts['test']}  (clean)")
        print(f"    Total : {total}")

    print("\n  ✅ No data leakage — safe to train!")
    print("=" * 60)


if __name__ == "__main__":
    main()