"""
train_yolo.py
─────────────────────────────────────────────────────────────
YOLOv8n Object Detection Training — ID Card Detector
Single class: id_card

Usage:
    python train_yolo.py

Make sure your dataset folder structure is:
    dataset/
    ├── data.yaml
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
─────────────────────────────────────────────────────────────
"""

import torch
from ultralytics import YOLO

# ─────────────────────────────────────────────
#  CONFIG — edit these paths to match your setup
# ─────────────────────────────────────────────
DATA_YAML  = "C:/Users/Renzo/Documents/MindVision/ID_detection.v2i.yolov8/data.yaml"  # path to your exported data.yaml
MODEL      = "yolov8s.pt"          # YOLOv8 nano — lightest and fastest
PROJECT    = "runs/id_card"        # folder where results will be saved
RUN_NAME   = "train_v1"            # name of this training run

# ─────────────────────────────────────────────
#  TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────
EPOCHS     = 100
IMG_SIZE   = 640
BATCH      = 16     # reduce to 8 if you get out of memory errors
WORKERS    = 4      # number of dataloader workers
PATIENCE   = 30     # early stopping — stops if no improvement after 20 epochs

# ─────────────────────────────────────────────
#  CHECK GPU
# ─────────────────────────────────────────────
print("=" * 55)
print(f"CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
    print(f"VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    DEVICE = 0  # use GPU
else:
    print("GPU not found — training on CPU (will be slow)")
    DEVICE = "cpu"
print("=" * 55)

# ─────────────────────────────────────────────
#  TRAIN
# ─────────────────────────────────────────────
def main():
    print(f"\n[Train] Loading model: {MODEL}")
    model = YOLO(MODEL)

    print(f"[Train] Starting training...")
    print(f"[Train] Dataset  : {DATA_YAML}")
    print(f"[Train] Epochs   : {EPOCHS}")
    print(f"[Train] Image sz : {IMG_SIZE}")
    print(f"[Train] Batch    : {BATCH}")
    print(f"[Train] Device   : {DEVICE}\n")

    results = model.train(
        data      = DATA_YAML,
        epochs    = EPOCHS,
        imgsz     = IMG_SIZE,
        batch     = BATCH,
        device    = DEVICE,
        workers   = WORKERS,
        patience  = PATIENCE,
        project   = PROJECT,
        name      = RUN_NAME,
        exist_ok  = True,

        # optimizer settings
        optimizer = "AdamW",
        lr0       = 0.0001,      # initial learning rate
        lrf       = 0.001,       # final learning rate fraction
        momentum  = 0.937,
        weight_decay = 0.0005,

        # augmentation — these supplement what roboflow already did
        hsv_h     = 0.015,      # hue augmentation
        hsv_s     = 0.7,        # saturation augmentation
        hsv_v     = 0.4,        # value/brightness augmentation
        degrees   = 10.0,       # rotation
        translate = 0.1,        # translation
        scale     = 0.5,        # scale
        fliplr    = 0.5,        # horizontal flip probability
        flipud    = 0.5,        # vertical flip probability
        mosaic    = 0.3,        # mosaic augmentation probability
        mixup     = 0.0,        # mixup augmentation probability

        # logging
        verbose   = True,
        save      = True,       # save best and last checkpoint
        save_period = 10,       # save checkpoint every 10 epochs
        plots     = True,       # save training plots
    )

    print("\n" + "=" * 55)
    print(f"[Train] Training complete!")
    print(f"[Train] Results saved to: {PROJECT}/{RUN_NAME}")
    print(f"[Train] Best model: {PROJECT}/{RUN_NAME}/weights/best.pt")
    print("=" * 55)

    # ─────────────────────────────────────────
    #  VALIDATE ON TEST SET
    # ─────────────────────────────────────────
    print("\n[Validate] Running validation on test set...")
    best_model = YOLO(f"{PROJECT}/{RUN_NAME}/weights/best.pt")
    metrics    = best_model.val(
        data   = DATA_YAML,
        imgsz  = IMG_SIZE,
        device = DEVICE,
        split  = "test",        # evaluate on test split
        plots  = True,
    )

    print("\n[Validate] Results:")
    print(f"  mAP50       : {metrics.box.map50:.4f}")
    print(f"  mAP50-95    : {metrics.box.map:.4f}")
    print(f"  Precision   : {metrics.box.mp:.4f}")
    print(f"  Recall      : {metrics.box.mr:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()