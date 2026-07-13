# tiny_loss.py  (adapted for ultralytics 8.4.x installed in this repo)
# ============================================================================
# TinyDetectionLoss + Row8E2ELoss for YOLO26 row-8 experiment
# (SAGA + RLE-Box + σ-Rank; no SD-ProgLoss -- that is row 9)
#
# Key API changes vs paper draft:
#  - preds per branch is a dict {"boxes":(b,4,a), "scores":(b,nc,a), "feats":[...]}
#    NOT a list of per-level feature maps.
#  - Optional "sigma":(b,4,a) key added by DetectSigma head (Phase 2 / RLE).
#  - E2ELoss anneal is step-based: update() called once per batch by the trainer.
#    Row8E2ELoss mirrors this exactly so baseline and row-8 are comparable.
#  - SAGAAssigner.set_anchor_strides() called inside get_assigned_targets_and_loss.
# ============================================================================

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist, dist2bbox, make_anchors

from saga_assigner import SAGAAssigner
from rle_box import RLEBoxLoss


class TinyDetectionLoss(v8DetectionLoss):
    """Single-branch detection loss with SAGA assigner + optional RLE-Box.

    Instantiate twice (o2m / o2o) via `branch=`.
    Preds contract per branch: {"boxes":(b,4,a), "scores":(b,nc,a),
                                "feats":[...], "sigma":(b,4,a) if use_rle}
    """

    def __init__(self, model, branch: str = "o2m",
                 saga_kw: dict | None = None,
                 use_rle: bool = False,
                 lambda_rle: float = 1.5):
        super().__init__(model)  # sets hyp, stride, nc, no, bce, reg_max, bbox_loss, ...

        saga_kw = dict(saga_kw or {})
        saga_kw.setdefault("stride", self.stride.tolist())  # pass model strides to parent TAL
        if branch == "o2o":
            saga_kw.setdefault("topk", 7)
            saga_kw.setdefault("topk2", 1)
        else:
            saga_kw.setdefault("topk", 10)
            saga_kw.setdefault("topk2", None)

        self.assigner = SAGAAssigner(num_classes=self.nc, alpha=0.5, beta=6.0, **saga_kw)
        self.use_rle = use_rle
        self.lambda_rle = lambda_rle
        self.rle = RLEBoxLoss() if use_rle else None

        # Cache gain values at init with YOLO26 defaults as fallback.
        # Inside a trainer model.args is a full namespace; outside (sanity checks,
        # val-only) it is a sparse dict with only a few keys.
        _h = self.hyp
        _get = (lambda k, d: _h.get(k, d)) if isinstance(_h, dict) else (lambda k, d: getattr(_h, k, d))
        self._gain_box = _get("box", 7.5)
        self._gain_cls = _get("cls", 0.5)

    def get_assigned_targets_and_loss(self, preds, batch):
        """Override: wire SAGA + optional RLE into the standard loss flow.

        preds is the per-branch dict (not the full model output dict).
        Returns the same 3-tuple as the parent so loss() works unchanged.
        """
        loss = torch.zeros(3, device=self.device)  # box, cls, rle (replaces dfl)

        # unpack the branch preds dict
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()   # (b, a, 4)
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()  # (b, a, nc)
        pred_sigma = None
        if self.use_rle and "sigma" in preds:
            pred_sigma = preds["sigma"].permute(0, 2, 1).contiguous().sigmoid()  # (b, a, 4)

        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = (torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype)
                 * self.stride[0])

        # preprocess targets (standard v8 flow)
        targets = torch.cat((batch["batch_idx"].view(-1, 1),
                             batch["cls"].view(-1, 1),
                             batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size,
                                  scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # decode to xyxy in grid units (reg_max=1, so dist2bbox is a passthrough)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # (b, a, 4) xyxy grid

        # SAGA assignment
        self.assigner.set_anchor_strides(stride_tensor)              # TINY-MOD [S]
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels, gt_bboxes, mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss (BCE, all anchors)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # box + RLE loss (foreground only)
        if fg_mask.sum():
            tbox_grid = target_bboxes / stride_tensor
            weight = target_scores.sum(-1, keepdim=True)             # (b, a, 1)

            iou = bbox_iou(pred_bboxes[fg_mask], tbox_grid[fg_mask], xywh=False, CIoU=True)
            loss[0] = ((1.0 - iou) * weight[fg_mask]).sum() / target_scores_sum

            if self.use_rle and pred_sigma is not None:              # TINY-MOD [R]
                # tgt_ltrb in stride units (unclipped, reg_max=1e9)
                tgt_ltrb = bbox2dist(anchor_points, tbox_grid, reg_max=1e9)
                flow_touch = sum(p.sum() for p in self.rle.flow.parameters()) * 0.0
                if fg_mask.any():
                    loss[2] = self.rle(pred_distri[fg_mask], pred_sigma[fg_mask],
                                       tgt_ltrb[fg_mask], weight[fg_mask], target_scores_sum)
                else:
                    loss[2] = pred_distri.sum() * 0.0 + flow_touch + pred_sigma.sum() * 0.0

        loss[0] *= self._gain_box      # 7.5 default
        loss[1] *= self._gain_cls      # 0.5 default
        loss[2] *= self.lambda_rle     # 1.5 (occupies the dead dfl slot)

        return (fg_mask, None, target_bboxes, anchor_points, stride_tensor), loss, loss.detach()


# ---------------------------------------------------------------------------
# Row-8 dual-head loss  (SAGA + RLE-Box + σ-Rank; stock ProgLoss schedule)
# ---------------------------------------------------------------------------
class Row8E2ELoss(E2ELoss):
    """E2ELoss subclass that replaces both v8DetectionLoss branches with TinyDetectionLoss.

    Inherits the step-based ProgLoss anneal (update() / decay()) unchanged,
    so the training curve is directly comparable to the stock baseline.
    Use_rle=False for row 3 (SAGA only); True for row 8 (needs DetectSigma head).
    """

    def __init__(self, model, saga_kw=None, use_rle=False, lambda_rle=1.5):
        # Init parent to set up o2m/o2m_copy/final_o2m etc., then replace branches.
        super().__init__(model)
        self.one2many = TinyDetectionLoss(model, "o2m", saga_kw, use_rle, lambda_rle)
        self.one2one  = TinyDetectionLoss(model, "o2o", saga_kw, use_rle, lambda_rle)

    # __call__, update(), decay() are all inherited from E2ELoss unchanged.
    # E2ELoss.__call__ does:
    #   preds = self.one2many.parse_output(preds)   <- handles dict/tuple
    #   loss_o2m = self.one2many.loss(preds["one2many"], batch)
    #   loss_o2o = self.one2one.loss(preds["one2one"], batch)
    #   return loss_o2m[0]*self.o2m + loss_o2o[0]*self.o2o, loss_o2o[1]
