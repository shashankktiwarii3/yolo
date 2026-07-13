# saga_assigner.py
# ============================================================================
# SAGA: Scale-Adaptive Gaussian Alignment for YOLO26's dual-head TAL
# (Proposal 1 of the tiny-object YOLO26 paper)
#
# Core idea
# ---------
# YOLO26 supervision is gated by Task-Aligned Learning (TAL):
#       t = s^alpha * u^beta          (alpha=0.5, beta=6.0 defaults)
# where u = CIoU(pred, gt). For tiny objects (AI-TOD: 2-16 px), u is
# near-zero and noise-dominated for EVERY candidate anchor, and beta=6
# annihilates it -> soft targets ~ 0 -> the one-to-many head barely learns
# the object, and the one-to-one head's topk=7 -> topk2=1 pick is a coin
# flip among equally-bad candidates.
#
# SAGA replaces, for tiny objects only:
#   (1) the binary center-in-box candidate mask  -> top-k by Gaussian
#       receptive-field similarity (RFLA-style prior), with a per-GT
#       argmax guarantee that SUBSUMES STAL's zero-positive fix;
#   (2) the IoU term u inside the alignment metric -> a scale-gated fusion
#           u* = g * u_gauss + (1 - g) * u_iou,
#           g  = sigmoid((tau - sqrt(w*h)) / T)
#       so behaviour on normal-scale objects (COCO regime) is untouched.
#
# The SAME u* drives the one-to-one branch's topk=7 -> argmax(=topk2=1)
# selection, making this (to our knowledge) the first tiny-object-aware
# one-to-one matcher inside an NMS-free CNN detector.
#
# Gaussian modelling (comments give the exact math used):
#   GT box (cx, cy, w, h)      -> N(mu_g, diag(sx_g^2, sy_g^2)),
#                                  sx_g = w/2, sy_g = h/2          [NWD conv.]
#   anchor point p at stride s -> N(p,   diag(r^2, r^2)),
#                                  r = erf_k * s                    [ERF proxy]
#
#   KL(N1 || N2)  (diagonal 2D):
#     0.5 * [ s1x^2/s2x^2 + s1y^2/s2y^2
#             + (m2x-m1x)^2/s2x^2 + (m2y-m1y)^2/s2y^2
#             - 2 + ln(s2x^2 s2y^2 / (s1x^2 s1y^2)) ]
#   similarity (RFLA-style):  u_kld = 1 / (1 + KL)
#
#   Wasserstein-2 (axis-aligned diagonal):
#     W2^2 = ||mu1-mu2||^2 + (s1x-s2x)^2 + (s1y-s2y)^2
#   similarity (NWD):         u_nwd = exp(-sqrt(W2^2) / C)
#     C should be the dataset mean absolute object size (AI-TOD ~ 12.8 px);
#     use estimate_dataset_C() below. A COCO-ish C saturates the exponential
#     and flattens gradients -- one reason plain NWD-loss runs fail.
#
# Compatibility
# -------------
# Developed against the stable ultralytics 8.x tal.py API
# (TaskAlignedAssigner with forward / get_pos_mask / get_box_metrics /
# select_highest_overlaps). YOLO26 (8.4.x) still follows this structure
# (see Hidayatullah & Tubagus, arXiv:2602.14582, source-level analysis).
# Only get_pos_mask() is overridden; if your installed version renamed
# internals, re-align that single method.
#
# The assigner needs per-anchor strides for the ERF radius. The loss call
# site must call `assigner.set_anchor_strides(stride_tensor)` each step
# (one line -- see tiny_loss.py, tagged  # <<< TINY-MOD [S]).
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
    else:  # "anchor2gt"
        kld = kld_diag_gauss(mu_a, s_a, mu_g, s_g)
    return 1.0 / (1.0 + kld)


def gauss_sim_nwd(mu_g, s_g, mu_a, s_a, C: float = 12.8):
    """Normalized Wasserstein similarity exp(-W2/C). Set C = mean object size (px)."""
    w2 = (mu_g - mu_a).pow(2).sum(-1) + (s_g - s_a).pow(2).sum(-1)
    return torch.exp(-w2.clamp_min(0).sqrt() / max(C, EPS))


def estimate_dataset_C(label_dir: str, imgsz: int = 800) -> float:
    """Mean absolute object size sqrt(w*h) in pixels over a YOLO-format label dir.

    Use the returned value as `nwd_C` (and to sanity-check gate_tau).
    AI-TOD-v2 should land near ~12.8 px at native 800x800.
    """
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

    o2m branch: SAGAAssigner(topk=10, o2o=False)
    o2o branch: SAGAAssigner(topk=7,  o2o=True)   # topk=7 pool -> argmax = topk2=1

    Args (beyond parent):
        gate_tau:   gate midpoint in *input pixels* (default 24 = between the
                    16/32 px tiny/small boundaries). Scale with input res.
        gate_temp:  gate softness T (default 4 px).
        erf_k:      anchor ERF radius = erf_k * stride (sweep {0.75,1,1.5,2}).
        cand_topk:  Gaussian candidate-pool size for tiny GTs (default 12).
        metric:     'kld' (default, RFLA-style) or 'nwd'.
        nwd_C:      NWD constant; MUST be set per dataset if metric='nwd'.
        kld_dir:    'gt2anchor' (default) or 'anchor2gt' (ablation).
        guarantee:  per-GT argmax-similarity anchor always in the pool
                    (soft superset of STAL's zero-positive guarantee).
        o2o:        collapse final positives to 1 best anchor per GT.
    """

    def __init__(self, topk=10, num_classes=80, alpha=0.5, beta=6.0, eps=1e-9,
                 gate_tau=24.0, gate_temp=4.0, erf_k=1.0, cand_topk=12,
                 metric="kld", nwd_C=12.8, kld_dir="gt2anchor",
                 guarantee=True, o2o=False):
        super().__init__(topk=topk, num_classes=num_classes, alpha=alpha, beta=beta, eps=eps)
        self.gate_tau, self.gate_temp = gate_tau, gate_temp
        self.erf_k, self.cand_topk = erf_k, cand_topk
        self.metric, self.nwd_C, self.kld_dir = metric, nwd_C, kld_dir
        self.guarantee, self.o2o = guarantee, o2o
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
            "SAGAAssigner: call set_anchor_strides(stride_tensor) before forward "
            "(see tiny_loss.py, tag # <<< TINY-MOD [S])."
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
        """Boolean top-k along the anchor dim, invalid entries excluded,
        duplicate-safe (same guard pattern as ultralytics' select_topk_candidates)."""
        m = metrics.masked_fill(~valid_mask, -float("inf"))
        k = min(k, metrics.shape[-1])
        topk_idx = m.topk(k, dim=-1, largest=True).indices          # (b, n, k)
        out = torch.zeros_like(metrics, dtype=torch.long)
        ones = torch.ones_like(topk_idx, dtype=torch.long)
        out.scatter_add_(-1, topk_idx, ones)
        out.masked_fill_(out > 1, 0)                                # kill dup collisions
        return out.bool() & valid_mask

    def _fused_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_pool, gsim, gate):
        """align = s^alpha * (u*)^beta with u* = g*gsim + (1-g)*CIoU. Returns (align, u*)."""
        b, n, a = mask_pool.shape
        u_iou = torch.zeros_like(gsim)
        scores = torch.zeros_like(gsim)

        if mask_pool.any():
            # class score of each anchor at its candidate GT's label
            ib = torch.arange(b, device=pd_scores.device).view(-1, 1).expand(-1, n)
            lbl = gt_labels.long().squeeze(-1)                       # (b, n)
            scores_full = pd_scores[ib, :, lbl]                      # (b, n, a)
            scores[mask_pool] = scores_full[mask_pool]

            gt_e = gt_bboxes.unsqueeze(2).expand(-1, -1, a, -1)[mask_pool]
            pd_e = pd_bboxes.unsqueeze(1).expand(-1, n, -1, -1)[mask_pool]
            iou_fn = getattr(self, "iou_calculation",
                             lambda g_, p_: bbox_iou(g_, p_, xywh=False, CIoU=True).squeeze(-1))
            u_iou[mask_pool] = iou_fn(gt_e, pd_e).clamp_(0)

        g = gate.unsqueeze(-1)                                       # (b, n, 1)
        u_star = (g * gsim + (1.0 - g) * u_iou).clamp_(0, 1) * mask_pool
        align = scores.clamp_min(EPS).pow(self.alpha) * u_star.clamp_min(EPS).pow(self.beta)
        return align * mask_pool, u_star

    # -- main override ----------------------------------------------------------
    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """Returns (mask_pos, align_metric, overlaps) like the parent.

        `overlaps` is our fused u*: the parent forward() then reuses it for
        select_highest_overlaps() conflict resolution and target-score
        normalization, keeping the whole pipeline consistent under one metric.
        """
        mask_gt_b = mask_gt.bool().squeeze(-1)                       # (b, n)

        # 1) candidate pools
        mask_in = self.select_candidates_in_gts(anc_points, gt_bboxes).bool()  # center-in-box
        gsim = self._gauss_sim(gt_bboxes, anc_points)                # (b, n, a)
        gate = self._tiny_gate(gt_bboxes)                            # (b, n)
        is_tiny = (gate > 0.5) & mask_gt_b                           # hard gate for pooling

        pool_tiny = self._topk_bool(gsim, self.cand_topk, mask_gt_b.unsqueeze(-1).expand_as(gsim))
        mask_pool = torch.where(is_tiny.unsqueeze(-1), pool_tiny | mask_in, mask_in)

        if self.guarantee:  # soft superset of STAL: best-similarity anchor always eligible
            best = gsim.argmax(dim=-1, keepdim=True)                 # (b, n, 1)
            mask_pool.scatter_(-1, best, True)

        mask_pool &= mask_gt_b.unsqueeze(-1)

        # 2) fused alignment metric on the pool
        align, u_star = self._fused_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes,
                                            mask_pool, gsim, gate)

        # 3) top-k by alignment (o2m: k=10 | o2o: k=7 pool ...)
        mask_pos = self._topk_bool(align, self.topk, mask_pool)

        # 4) ... then o2o secondary filter = argmax per GT ( == paper's topk2=1 )
        if self.o2o:
            a_m = align.masked_fill(~mask_pos, -float("inf"))
            best = a_m.argmax(dim=-1, keepdim=True)
            one = torch.zeros_like(mask_pos)
            one.scatter_(-1, best, True)
            mask_pos = one & mask_pos

        return mask_pos, align, u_star


# Sanity invariant (unit test): with gate_tau -> -inf (gate==0 everywhere),
# guarantee=False and o2o=False, SAGAAssigner must reproduce the parent
# TaskAlignedAssigner assignment exactly (u* == CIoU, pool == center-in-box).
