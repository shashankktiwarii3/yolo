# saga_assigner.py  (adapted for ultralytics 8.4.x installed in this repo)
# ============================================================================
# SAGA: Scale-Adaptive Gaussian Alignment for YOLO26's dual-head TAL
#
# API changes vs the paper draft:
#  - TaskAlignedAssigner.__init__ now takes `stride` (list), passed through.
#  - select_candidates_in_gts now requires mask_gt as third positional arg.
#  - set_anchor_strides() still called each step from TinyDetectionLoss.
# ============================================================================

from __future__ import annotations

import glob
import math
import os

import torch
import torch.nn as nn

from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import TaskAlignedAssigner

EPS = 1e-9


# ----------------------------------------------------------------------------
# Gaussian utilities
# ----------------------------------------------------------------------------
def boxes_to_gauss(boxes_xyxy: torch.Tensor):
    """xyxy (..., 4) [pixels] -> (mu (..., 2), sigma (..., 2)); sigma = (w/2, h/2)."""
    x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
    mu = torch.stack(((x1 + x2) * 0.5, (y1 + y2) * 0.5), dim=-1)
    sigma = torch.stack(((x2 - x1).clamp_min(EPS) * 0.5, (y2 - y1).clamp_min(EPS) * 0.5), dim=-1)
    return mu, sigma


def kld_diag_gauss(mu1, s1, mu2, s2):
    """KL( N(mu1, diag(s1^2)) || N(mu2, diag(s2^2)) ), broadcastable, last dim = 2."""
    v1, v2 = s1.pow(2).clamp_min(EPS), s2.pow(2).clamp_min(EPS)
    trace = (v1 / v2).sum(-1)
    maha = ((mu2 - mu1).pow(2) / v2).sum(-1)
    logdet = (v2.log() - v1.log()).sum(-1)
    return 0.5 * (trace + maha - 2.0 + logdet)


def gauss_sim_kld(mu_g, s_g, mu_a, s_a, direction: str = "gt2anchor"):
    """RFLA-style similarity 1/(1+KL). `direction` is an ablation knob."""
    if direction == "gt2anchor":
        kld = kld_diag_gauss(mu_g, s_g, mu_a, s_a)
    else:
        kld = kld_diag_gauss(mu_a, s_a, mu_g, s_g)
    return 1.0 / (1.0 + kld)


def gauss_sim_nwd(mu_g, s_g, mu_a, s_a, C: float = 12.8):
    """Normalized Wasserstein similarity exp(-W2/C). Set C = mean object size (px)."""
    w2 = (mu_g - mu_a).pow(2).sum(-1) + (s_g - s_a).pow(2).sum(-1)
    return torch.exp(-w2.clamp_min(0).sqrt() / max(C, EPS))


def estimate_dataset_C(label_dir: str, imgsz: int = 800) -> float:
    """Mean sqrt(w*h) in pixels over a YOLO-format label dir (AI-TOD-v2 ~ 12.8 px)."""
    sizes = []
    for f in glob.glob(os.path.join(label_dir, "*.txt")):
        with open(f) as fh:
            for line in fh:
                p = line.split()
                if len(p) >= 5:
                    w, h = float(p[3]) * imgsz, float(p[4]) * imgsz
                    if w > 0 and h > 0:
                        sizes.append(math.sqrt(w * h))
    if not sizes:
        raise ValueError(f"No labels found under {label_dir}")
    return float(sum(sizes) / len(sizes))


# ----------------------------------------------------------------------------
# SAGA assigner
# ----------------------------------------------------------------------------
class SAGAAssigner(TaskAlignedAssigner):
    """Drop-in TAL replacement for BOTH YOLO26 heads.

    o2m branch: SAGAAssigner(topk=10, topk2=None, stride=[8,16,32])
    o2o branch: SAGAAssigner(topk=7,  topk2=1,    stride=[8,16,32])
    """

    def __init__(self, topk=10, num_classes=80, alpha=0.5, beta=6.0, eps=1e-9,
                 stride=None, gate_tau=24.0, gate_temp=4.0, erf_k=1.0, cand_topk=12,
                 metric="kld", nwd_C=12.8, kld_dir="gt2anchor",
                 guarantee=True, topk2=None):
        stride = stride or [8, 16, 32]
        # installed TAL takes (topk, num_classes, alpha, beta, stride, eps, topk2)
        super().__init__(topk=topk, num_classes=num_classes, alpha=alpha, beta=beta,
                         stride=stride, eps=eps, topk2=topk2)
        self.gate_tau, self.gate_temp = gate_tau, gate_temp
        self.erf_k, self.cand_topk = erf_k, cand_topk
        self.metric, self.nwd_C, self.kld_dir = metric, nwd_C, kld_dir
        self.guarantee = guarantee
        self._anc_strides = None  # (num_anchors, 1), set each step by the loss

    # -- wiring ---------------------------------------------------------------
    def set_anchor_strides(self, stride_tensor: torch.Tensor):
        """Call from the loss each step with make_anchors()' stride_tensor."""
        self._anc_strides = stride_tensor

    # -- pieces ---------------------------------------------------------------
    def _tiny_gate(self, gt_bboxes):
        """(b, n, 4) xyxy px -> soft gate g in [0,1], (b, n)."""
        w = (gt_bboxes[..., 2] - gt_bboxes[..., 0]).clamp_min(0)
        h = (gt_bboxes[..., 3] - gt_bboxes[..., 1]).clamp_min(0)
        size = (w * h).clamp_min(EPS).sqrt()
        return torch.sigmoid((self.gate_tau - size) / max(self.gate_temp, EPS))

    def _gauss_sim(self, gt_bboxes, anc_points):
        """(b, n, 4), (a, 2) -> similarity (b, n, a) in (0, 1]. Runs in fp32."""
        assert self._anc_strides is not None, (
            "SAGAAssigner: call set_anchor_strides(stride_tensor) before forward."
        )
        with torch.autocast(device_type=gt_bboxes.device.type, enabled=False):
            mu_g, s_g = boxes_to_gauss(gt_bboxes.float())          # (b, n, 2)
            mu_a = anc_points.float().view(1, 1, -1, 2)            # (1, 1, a, 2)
            r = (self.erf_k * self._anc_strides.float()).view(1, 1, -1, 1)
            s_a = r.expand(-1, -1, -1, 2)                          # isotropic ERF
            mu_g, s_g = mu_g.unsqueeze(2), s_g.unsqueeze(2)        # (b, n, 1, 2)
            if self.metric == "nwd":
                sim = gauss_sim_nwd(mu_g, s_g, mu_a, s_a, C=self.nwd_C)
            else:
                sim = gauss_sim_kld(mu_g, s_g, mu_a, s_a, direction=self.kld_dir)
        return sim.to(gt_bboxes.dtype)                             # (b, n, a)

    @staticmethod
    def _topk_bool(metrics, k, valid_mask):
        """Boolean top-k along the anchor dim, invalid entries excluded."""
        m = metrics.masked_fill(~valid_mask, -float("inf"))
        k = min(k, metrics.shape[-1])
        topk_idx = m.topk(k, dim=-1, largest=True).indices
        out = torch.zeros_like(metrics, dtype=torch.long)
        ones = torch.ones_like(topk_idx, dtype=torch.long)
        out.scatter_add_(-1, topk_idx, ones)
        out.masked_fill_(out > 1, 0)
        return out.bool() & valid_mask

    def _fused_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_pool, gsim, gate):
        """align = s^alpha * (u*)^beta with u* = g*gsim + (1-g)*CIoU."""
        b, n, a = mask_pool.shape
        u_iou = torch.zeros_like(gsim)
        scores = torch.zeros_like(gsim)

        if mask_pool.any():
            ib = torch.arange(b, device=pd_scores.device).view(-1, 1).expand(-1, n)
            lbl = gt_labels.long().squeeze(-1)                       # (b, n)
            scores_full = pd_scores[ib, :, lbl]                      # (b, n, a)
            scores[mask_pool] = scores_full[mask_pool]

            gt_e = gt_bboxes.unsqueeze(2).expand(-1, -1, a, -1)[mask_pool]
            pd_e = pd_bboxes.unsqueeze(1).expand(-1, n, -1, -1)[mask_pool]
            iou_fn = getattr(self, "iou_calculation",
                             lambda g_, p_: bbox_iou(g_, p_, xywh=False, CIoU=True).squeeze(-1))
            u_iou[mask_pool] = iou_fn(gt_e, pd_e).clamp_(0)

        g = gate.unsqueeze(-1)
        u_star = (g * gsim + (1.0 - g) * u_iou).clamp_(0, 1) * mask_pool
        align = scores.clamp_min(EPS).pow(self.alpha) * u_star.clamp_min(EPS).pow(self.beta)
        return align * mask_pool, u_star

    # -- main override --------------------------------------------------------
    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """Returns (mask_pos, align_metric, overlaps) like the parent.

        mask_pos is FLOAT (0/1) so that select_highest_overlaps can call
        argmax(-2) on it without error (argmax does not accept bool in PyTorch).
        The topk2 secondary filter (o2o argmax) is intentionally LEFT to the
        parent's select_highest_overlaps — removing the duplicate application
        also makes the gate=-inf path exactly reproduce stock TAL (invariant 1).
        """
        mask_gt_b = mask_gt.bool().squeeze(-1)                       # (b, n)

        # 1) candidate pools (float tensors throughout to stay compatible)
        mask_in = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)  # float
        gsim = self._gauss_sim(gt_bboxes, anc_points)                # (b, n, a) float
        gate = self._tiny_gate(gt_bboxes)                            # (b, n) float
        is_tiny = (gate > 0.5) & mask_gt_b

        # Gaussian candidate pool for tiny GTs (stay float via _topk_bool→float())
        pool_tiny = self._topk_bool(gsim, self.cand_topk,
                                    mask_gt_b.unsqueeze(-1).expand_as(gsim)).float()
        mask_pool = torch.where(is_tiny.unsqueeze(-1),
                                (pool_tiny + mask_in).clamp_max(1.0),
                                mask_in)                             # float (0/1)

        if self.guarantee:
            best = gsim.argmax(dim=-1, keepdim=True)
            mask_pool_bool = mask_pool.bool()
            mask_pool_bool.scatter_(-1, best, True)
            mask_pool = mask_pool_bool.float()

        mask_pool = mask_pool * mask_gt_b.unsqueeze(-1).float()      # zero out padding GTs

        # 2) fused alignment metric on the pool (pass bool view for index logic)
        align, u_star = self._fused_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes,
                                            mask_pool.bool(), gsim, gate)

        # 3) top-k selection — use the INSTALLED select_topk_candidates so the
        #    candidate-counting logic matches stock TAL exactly (fixes invariant 1).
        #    Pre-zero align outside pool so topk naturally stays within it.
        align_in_pool = align * mask_pool
        mask_pos = self.select_topk_candidates(
            align_in_pool,
            topk_mask=mask_gt.expand(-1, -1, self.topk).bool(),
        ) * mask_pool                                                # float, pool-constrained

        # NOTE: topk2 secondary filter (o2o argmax) is handled by the parent's
        # select_highest_overlaps (called from _forward). Do NOT apply it here.

        return mask_pos, align_in_pool, u_star
