# train_row8_2.py
# ============================================================================
# Row 8: SAGA + RLE-Box + σ-Rank on VisDrone — trained FROM SCRATCH
# (no pretrained weights; all weights randomly initialised)
#
# VisDrone notes vs AI-TOD:
#  - 10 classes, images vary in resolution (typically ~1920×1080 or 2000×1500),
#    trained at 1024 input (EXPERIMENTS.md §2).
#  - gate_tau scales with imgsz: AI-TOD used 24 @ 800 → 30.7 @ 1024
#    (linear: gate_tau_new = 24 * 1024/800 ≈ 30.7, rounded to 30).
#  - maxDets must be raised; VisDrone toolkit tradition: [10, 100, 500].
#  - From-scratch training typically needs longer schedules and lower lr warmup;
#    300 epochs recommended.  Adjust DOWN if you're GPU-constrained.
#
# PHASE 1 (no arch change, run first to validate SAGA):
#   USE_RLE = False, MODEL = "yolo26n.yaml"
#
# PHASE 2 (full row 8):
#   USE_RLE = True,  MODEL = "yolo26n-sigma.yaml"
# ============================================================================

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from ultralytics import YOLO
from tiny_loss import Row8E2ELoss

# ----------------------------- experiment config -----------------------------
PHASE = 2                              # 1 = SAGA only; 2 = SAGA + RLE + σ-Rank

if PHASE == 1:
    MODEL    = "yolo26n.yaml"          # yaml only — no .pt → random init
    USE_RLE  = False
    RUN_NAME = "visdrone_row8_p1_saga_scratch"
else:
    MODEL    = "yolo26n-sigma.yaml"    # DetectSigma yaml — random init
    USE_RLE  = True
    RUN_NAME = "visdrone_row8_p2_full_scratch"

DATA   = "VisDrone.yaml"
IMGSZ  = 640                          # VisDrone standard eval size
EPOCHS = 300                           # from-scratch needs more epochs than fine-tune
BATCH  = 8                             # lower than AI-TOD: 1024² images are larger

# gate_tau scaled linearly from AI-TOD baseline: 24 * (1024/800) ≈ 30
SAGA_KW = dict(
    gate_tau  = 30.0,
    gate_temp = 4.0,
    erf_k     = 1.0,                   # sweep {0.75, 1.0, 1.5, 2.0}
    cand_topk = 12,
    metric    = "kld",                 # "kld" | "nwd"
    nwd_C     = 12.8,                  # update with estimate_dataset_C() on VisDrone labels
    guarantee = True,
)
LAMBDA_RLE       = 1.5
SIGMA_RANK_GAMMA = 0.5                 # sweep {0.25, 0.5, 1.0}; 0 = baseline exact


# ------------------------------- callbacks -----------------------------------
def on_train_start(trainer):
    """Swap in Row8E2ELoss BEFORE the first batch (no pretrained weights to load)."""
    crit = Row8E2ELoss(
        trainer.model,
        saga_kw=SAGA_KW,
        use_rle=USE_RLE,
        lambda_rle=LAMBDA_RLE,
    )
    trainer.model.criterion = crit
    print(f"[visdrone-row8] Row8E2ELoss attached  rle={USE_RLE}  saga={SAGA_KW}")

    if USE_RLE:
        head = trainer.model.model[-1]
        if hasattr(head, "set_sigma_rank_gamma"):
            head.set_sigma_rank_gamma(SIGMA_RANK_GAMMA)


# ----------------------------- evaluation ------------------------------------
def run_val(model):
    """Both e2e and NMS-mode val per EXPERIMENTS.md §2 protocol."""
    print("\n[visdrone-row8] === e2e val ===")
    model.val(data=DATA, imgsz=IMGSZ)
    print("\n[visdrone-row8] === NMS-mode val (end2end=False) ===")
    model.val(data=DATA, imgsz=IMGSZ, end2end=False)


def audit_gt_density(coco_gt_json: str, cap: int = 300):
    """% of images whose GT count exceeds the one-to-one head's fixed 300-det cap.

    VisDrone is dense — many frames exceed 300 instances. Run this once and
    include the table in the paper (EXPERIMENTS.md §2 requirement).
    """
    import json
    from collections import Counter
    anns = json.load(open(coco_gt_json))
    per_img = Counter(a["image_id"] for a in anns["annotations"])
    n_over  = sum(v > cap for v in per_img.values())
    print(f"{n_over}/{len(per_img)} images ({100*n_over/max(len(per_img),1):.1f}%) "
          f"exceed {cap} GT instances → e2e recall ceiling; include this in the paper.")
    # histogram
    for threshold in [50, 100, 200, 300, 500]:
        n = sum(v > threshold for v in per_img.values())
        print(f"  > {threshold:4d} GT: {n} images ({100*n/max(len(per_img),1):.1f}%)")


# -------------------------------- baseline -----------------------------------
def run_baseline():
    """Row 0 from scratch on VisDrone — stock YOLO26n, no modifications."""
    model = YOLO("yolo26n.yaml")       # yaml → random init, no weights
    model.train(
        data=DATA, imgsz=IMGSZ, epochs=EPOCHS, batch=BATCH,
        project="tiny_yolo26_visdrone", name="row0_baseline_scratch", seed=0,
    )
    run_val(model)


# --------------------------------- main -------------------------------------
if __name__ == "__main__":
    model = YOLO(MODEL)
    model.add_callback("on_train_start", on_train_start)

    model.train(
        data=DATA, imgsz=IMGSZ, epochs=EPOCHS, batch=BATCH,
        project="tiny_yolo26_visdrone", name=RUN_NAME,
        seed=0,                        # paper: seeds {0, 1, 2} for mean±std

        # From-scratch hyperparameter tweaks vs fine-tuning defaults:
        # Warmup is more important with random init; keep lr defaults unless
        # you see a spike in the first 3 epochs, then lower lr0.
        warmup_epochs=5,               # ultralytics default is 3
        cos_lr=True,                   # cosine LR decay for long schedules

        # VisDrone-specific: raise maxDets for the dense scenes
        # (passed to the validator; toolkit tradition is 500)
        max_det=500,
    )

    run_val(model)
