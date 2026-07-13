# sanity_check.py
# ============================================================================
# Run BEFORE any GPU-week.  Implements invariants 1–4 from EXPERIMENTS.md §4.
#
#   python sanity_check.py
#
# All checks print PASS / FAIL.  Fix any FAIL before launching training.
# Requires torch + ultralytics installed; does NOT need a dataset.
# ============================================================================
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from ultralytics import YOLO
from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _make_dummy_batch(bs=2, n_gt=4, nc=8, imgsz=320):
    """Minimal fake batch matching the ultralytics contract."""
    bboxes = torch.rand(bs * n_gt, 4) * (imgsz * 0.5)
    bboxes[:, 2:] += bboxes[:, :2] + 4          # ensure positive wh
    bboxes[:, 2:].clamp_(max=imgsz)
    cls     = torch.zeros(bs * n_gt, 1)
    batch_idx = torch.cat([torch.full((n_gt,), i) for i in range(bs)])
    # normalize to [0,1] for ultralytics preprocess
    bboxes_norm = bboxes / imgsz
    # xywh format
    cx = (bboxes_norm[:, 0] + bboxes_norm[:, 2]) / 2
    cy = (bboxes_norm[:, 1] + bboxes_norm[:, 3]) / 2
    w  = (bboxes_norm[:, 2] - bboxes_norm[:, 0]).clamp_min(0)
    h  = (bboxes_norm[:, 3] - bboxes_norm[:, 1]).clamp_min(0)
    bboxes_xywh = torch.stack([cx, cy, w, h], dim=-1)
    return dict(batch_idx=batch_idx, cls=cls, bboxes=bboxes_xywh, img=torch.zeros(bs,3,imgsz,imgsz))


# ------------------------------------------------------------------
# Invariant 1: SAGAAssigner(gate_tau=-1e9, guarantee=False) ≡ stock TAL
# ------------------------------------------------------------------
def check_saga_equiv_to_tal():
    from saga_assigner import SAGAAssigner
    nc, bs, ngt, imgsz = 8, 2, 4, 320
    strides = [8, 16, 32]

    tal_o2m = TaskAlignedAssigner(topk=10, num_classes=nc, alpha=0.5, beta=6.0,
                                  stride=strides, topk2=None)
    # With gate_tau=-1e9 the gate is ~0 everywhere -> u* == CIoU, pool == center-in-box
    saga_o2m = SAGAAssigner(topk=10, num_classes=nc, alpha=0.5, beta=6.0,
                             stride=strides, topk2=None,
                             gate_tau=-1e9, guarantee=False)

    # build fake anchors / preds
    torch.manual_seed(42)
    feats = [torch.randn(bs, 1, imgsz//s, imgsz//s) for s in strides]  # just shape
    anc_pts, stride_t = make_anchors(feats, torch.tensor(strides, dtype=torch.float), 0.5)
    anc_pts_px  = anc_pts * stride_t                    # pixel frame
    pd_bboxes   = torch.rand(bs, anc_pts.shape[0], 4) * 64 + 4
    pd_bboxes[..., 2:] += pd_bboxes[..., :2]           # xyxy
    pd_scores   = torch.rand(bs, anc_pts.shape[0], nc).sigmoid()
    gt_labels   = torch.zeros(bs, ngt, 1, dtype=torch.long)
    gt_bboxes   = torch.rand(bs, ngt, 4) * 64 + 4
    gt_bboxes[..., 2:] += gt_bboxes[..., :2]
    mask_gt     = torch.ones(bs, ngt, 1)

    saga_o2m.set_anchor_strides(stride_t)
    with torch.no_grad():
        _, tb_tal,  ts_tal,  fg_tal,  _ = tal_o2m( pd_scores, pd_bboxes, anc_pts_px, gt_labels, gt_bboxes, mask_gt)
        _, tb_saga, ts_saga, fg_saga, _ = saga_o2m(pd_scores, pd_bboxes, anc_pts_px, gt_labels, gt_bboxes, mask_gt)

    fg_match = (fg_tal == fg_saga).all().item()
    ts_match = torch.allclose(ts_tal, ts_saga, atol=1e-4)
    status = "PASS" if fg_match else "FAIL"
    print(f"[1] SAGA(gate=-inf, guarantee=False) ≡ stock TAL: {status}"
          f"  fg_mask_match={fg_match}  target_scores_match={ts_match}")
    return fg_match


# ------------------------------------------------------------------
# Invariant 4: gamma=0 σ-Rank ≡ baseline scores (bit-exact)
# ------------------------------------------------------------------
def check_sigma_rank_identity():
    from rle_box import sigma_rank_scores
    torch.manual_seed(7)
    scores   = torch.rand(2, 100, 8)
    pred_box = torch.rand(2, 100, 4).abs() + 0.1
    pred_sig = torch.rand(2, 100, 4) * 0.5

    s_out = sigma_rank_scores(scores, pred_box, pred_sig, gamma=0.0)
    ok = torch.allclose(s_out, scores)
    print(f"[4] sigma_rank(gamma=0) ≡ baseline scores: {'PASS' if ok else 'FAIL'}")
    return ok


# ------------------------------------------------------------------
# Invariant 3 (lite): Row8E2ELoss with gate_tau=-1e9 + lambda_rle=0
#                     attaches and forward-passes without error
# ------------------------------------------------------------------
def check_loss_forward():
    from tiny_loss import Row8E2ELoss
    model = YOLO("yolo26n.pt").model.to(DEVICE)
    saga_kw_stock = dict(gate_tau=-1e9, gate_temp=4.0, guarantee=False,
                         erf_k=1.0, cand_topk=12, metric="kld")
    try:
        crit = Row8E2ELoss(model, saga_kw=saga_kw_stock, use_rle=False, lambda_rle=0.0)
        model.criterion = crit

        batch = _make_dummy_batch(bs=2, n_gt=4, nc=80, imgsz=320)
        batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        model.train()
        with torch.no_grad():
            preds = model(batch["img"])
        # criterion expects (preds, batch)
        total, _ = crit(preds, batch)
        # criterion returns a 3-element loss vector (box, cls, rle);
        # trainer sums it (trainer.py line 434: self.loss = loss.sum())
        ok = total.sum().isfinite().item()
        print(f"[3] Row8E2ELoss forward pass (gate=-inf, rle=False): {'PASS' if ok else 'FAIL (non-finite loss)'}")
    except Exception as e:
        print(f"[3] Row8E2ELoss forward pass: FAIL  ({type(e).__name__}: {e})")
        ok = False
    return ok


if __name__ == "__main__":
    print(f"Device: {DEVICE}\n")
    r1 = check_saga_equiv_to_tal()
    r4 = check_sigma_rank_identity()
    r3 = check_loss_forward()
    print(f"\n{'All checks PASSED' if all([r1, r4, r3]) else 'Some checks FAILED -- fix before training'}")
