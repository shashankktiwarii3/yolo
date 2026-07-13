# train_row8.py
# ============================================================================
# Row 8: SAGA + RLE-Box + σ-Rank  (main system, no SD-ProgLoss)
#
# PHASE 1 (run first — stock yolo26n.pt weights, no arch change):
#   USE_RLE = False, MODEL = "yolo26n.pt"
#   Tests SAGA alone (equivalent to row 3 in the ablation matrix).
#   Use this to verify SAGA is working before adding the sigma head.
#
# PHASE 2 (row 8 proper — needs DetectSigma head):
#   USE_RLE = True,  MODEL = "yolo26n-sigma.yaml" (then load yolo26n.pt)
#   The on_train_start callback does the weight surgery automatically.
#
# Comparison / baseline (row 0):
#   Run with stock YOLO("yolo26n.pt").train(...) without any callbacks.
#   Or call run_baseline() below.
# ============================================================================

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))  # make repo root importable

from ultralytics import YOLO
from tiny_loss import Row8E2ELoss

# ----------------------------- experiment config -----------------------------
PHASE = 1                         # 1 = SAGA only; 2 = SAGA + RLE + σ-Rank

if PHASE == 1:
    MODEL = "yolo26n.pt"          # pretrained weights load cleanly
    USE_RLE = False
    RUN_NAME = "row8_phase1_saga"
else:
    MODEL = "yolo26n-sigma.yaml"  # DetectSigma yaml; weights loaded in callback
    PRETRAINED = "yolo26n.pt"     # surgery source
    USE_RLE = True
    RUN_NAME = "row8_phase2_full"

DATA     = "AI-TOD-v2.yaml"       # swap to "VisDrone.yaml" for secondary dataset
IMGSZ    = 800                    # AI-TOD native; VisDrone: use 1024
EPOCHS   = 150
BATCH    = 16

SAGA_KW = dict(
    gate_tau  = 24.0,             # input-pixel gate midpoint (scale with imgsz)
    gate_temp = 4.0,
    erf_k     = 1.0,              # sweep {0.75, 1.0, 1.5, 2.0}
    cand_topk = 12,
    metric    = "kld",            # "kld" (default) | "nwd"
    nwd_C     = 12.8,             # only used if metric="nwd"
    guarantee = True,
)
LAMBDA_RLE = 1.5                  # occupies the dead DFL gain slot
SIGMA_RANK_GAMMA = 0.5            # sweep {0.25, 0.5, 1.0}; 0 = baseline exact


# ------------------------------- callbacks -----------------------------------
def on_train_start(trainer):
    """Swap in Row8E2ELoss BEFORE the first batch.

    ultralytics builds criterion lazily (DetectionModel.loss() checks
    `if self.criterion is None`), so presetting it here is safe.
    """
    crit = Row8E2ELoss(
        trainer.model,
        saga_kw=SAGA_KW,
        use_rle=USE_RLE,
        lambda_rle=LAMBDA_RLE,
    )
    trainer.model.criterion = crit
    print(f"[row8] Row8E2ELoss attached  rle={USE_RLE}  saga={SAGA_KW}")

    if PHASE == 2:
        # Weight surgery: load pretrained box_head/cls_head, skip sig_head
        import torch
        state = torch.load(PRETRAINED, map_location="cpu")
        sd = state.get("model", state)
        if hasattr(sd, "state_dict"):
            sd = sd.state_dict()
        missing, unexpected = trainer.model.load_state_dict(sd, strict=False)
        print(f"[row8] weight surgery: {len(missing)} missing, {len(unexpected)} unexpected")
        # Set sigma-Rank gamma on the head
        head = trainer.model.model[-1]
        if hasattr(head, "set_sigma_rank_gamma"):
            head.set_sigma_rank_gamma(SIGMA_RANK_GAMMA)


def on_train_epoch_start(trainer):
    """No-op for row 8 (SD-ProgLoss epoch hook not needed here)."""
    pass


# ----------------------------- evaluation ------------------------------------
def run_val(model, data=DATA, imgsz=IMGSZ):
    """Run both e2e and NMS-mode val per EXPERIMENTS.md protocol."""
    print("\n[row8] === e2e val ===")
    model.val(data=data, imgsz=imgsz)
    print("\n[row8] === NMS-mode val (end2end=False) ===")
    model.val(data=data, imgsz=imgsz, end2end=False)


def audit_gt_density(coco_gt_json: str, cap: int = 300):
    """% of images whose GT count exceeds the one-to-one head's fixed cap."""
    import json
    from collections import Counter
    anns = json.load(open(coco_gt_json))
    per_img = Counter(a["image_id"] for a in anns["annotations"])
    n_over = sum(v > cap for v in per_img.values())
    print(f"{n_over}/{len(per_img)} images ({100*n_over/max(len(per_img),1):.1f}%) "
          f"exceed {cap} GT instances -> e2e recall ceiling; report this table.")


# -------------------------------- baseline -----------------------------------
def run_baseline():
    """Row 0: stock YOLO26n with no modifications (for comparison)."""
    model = YOLO("yolo26n.pt")
    model.train(data=DATA, imgsz=IMGSZ, epochs=EPOCHS, batch=BATCH,
                project="tiny_yolo26", name="row0_baseline", seed=0)
    run_val(model)


# --------------------------------- main -------------------------------------
if __name__ == "__main__":
    model = YOLO(MODEL)
    model.add_callback("on_train_start",       on_train_start)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)

    model.train(
        data=DATA, imgsz=IMGSZ, epochs=EPOCHS, batch=BATCH,
        project="tiny_yolo26", name=RUN_NAME,
        seed=0,   # paper uses seeds {0, 1, 2}; re-run with seed=1,2 for mean±std
    )

    run_val(model)
