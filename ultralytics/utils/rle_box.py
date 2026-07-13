# rle_box.py
# ============================================================================
# RLE-Box: distributional localization without a distributional head
# (Proposal 2 of the tiny-object YOLO26 paper)
#
# Motivation (the DFL-removal tension)
# ------------------------------------
# DFL (YOLOv8..v13) predicted each box side as a K=16-bin distribution:
#     d_hat = sum_i i * softmax(z)_i,   64 regression channels total,
#     range-capped at (K-1)*stride per side.
# Its expectation decoding gave sub-bin (sub-pixel) refinement and implicit
# aleatoric edge-uncertainty modelling -- exactly what tiny boxes need most:
# for an 8x8 GT, a 1 px shift in both axes -> IoU 0.62; 2 px -> IoU 0.39.
# YOLO26 removed DFL (4 channels, unconstrained range, cleaner export) and
# compensated with ProgLoss+STAL -- which fix ASSIGNMENT, not LOCALIZATION
# PRECISION. Validated on COCO where tiny objects are a minority; on AI-TOD
# they are the whole dataset.
#
# RLE-Box restores the distribution at 8 channels instead of 64:
#   * a sigma branch predicts per-side uncertainty  sigma in (0,1)^4
#   * training uses Residual Log-Likelihood Estimation with a shared
#     RealNVP normalizing flow (YOLO26 already ships this machinery in its
#     POSE head; we port it to box sides):
#
#         eps   = (d_hat - d*) / sigma                       (per side)
#         L_RLE = log(sigma) - log phi(eps) + [ log(2 sigma) + |eps| ]
#                                              \____ Laplace anchor ____/
#
#     (exactly the printed YOLO26 pose objective, arXiv:2606.03748 Eq. 11,
#      applied per box side; phi = flow density of the normalized residual.)
#   * L_RLE occupies the loss slot DFL vacated (v8 weights box/cls/dfl =
#     7.5/0.5/1.5 -> default lambda_rle = 1.5).
#   * the FLOW is train-only and discarded at export (zero inference cost);
#     the SIGMA branch stays (4 extra channels, ~negligible), because it
#     powers the second contribution:
#
# sigma-Rank (fixes the NMS-free ranking pathology)
# -------------------------------------------------
# The one-to-one head emits a FIXED set of <=300 detections ranked purely by
# classification score -- tiny objects get crowded out by large confident
# ones. Calibrate the ranking with predicted localization quality:
#
#     r_i = sigma_i / (d_hat_i + 1)            (relative side uncertainty)
#     q   = exp( -gamma * mean_i r_i )
#     s'  = s * q                              (before the top-300 cut)
#
# Deviation from the pose head, to state in the paper: pose RLE flows 2D
# (x, y) residuals; we flow the joint 4D (l, t, r, b) residual because the
# four sides of one box share object-level ambiguity (occlusion, blur).
# Ablate 4D-joint vs 2x2D-paired flows.
# ============================================================================

from __future__ import annotations

import math

import torch
import torch.nn as nn

EPS = 1e-9


# ----------------------------------------------------------------------------
# RealNVP normalizing flow (train-only; never exported)
# ----------------------------------------------------------------------------
class _Coupling(nn.Module):
    def __init__(self, dim: int, mask: torch.Tensor, hidden: int = 64):
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.scale = nn.Sequential(nn.Linear(dim, hidden), nn.LeakyReLU(0.1),
                                   nn.Linear(hidden, hidden), nn.LeakyReLU(0.1),
                                   nn.Linear(hidden, dim), nn.Tanh())
        self.translate = nn.Sequential(nn.Linear(dim, hidden), nn.LeakyReLU(0.1),
                                       nn.Linear(hidden, hidden), nn.LeakyReLU(0.1),
                                       nn.Linear(hidden, dim))

    def forward(self, x):
        xm = x * self.mask
        s = self.scale(xm) * (1 - self.mask)
        t = self.translate(xm) * (1 - self.mask)
        z = xm + (1 - self.mask) * (x * torch.exp(s) + t)
        return z, s.sum(dim=-1)  # log|det J| contribution


class RealNVP(nn.Module):
    """Small RealNVP density estimator over residual vectors.

    dim=4 -> joint (l, t, r, b) flow (default);  dim=2 -> paired-2D ablation.
    log_prob(x) = log N(z; 0, I) + sum log|det J|.
    """

    def __init__(self, dim: int = 4, n_pairs: int = 3, hidden: int = 64):
        super().__init__()
        half = dim // 2
        m = torch.zeros(dim); m[:half] = 1.0
        masks = [m if i % 2 == 0 else 1 - m for i in range(2 * n_pairs)]
        self.layers = nn.ModuleList(_Coupling(dim, mk, hidden) for mk in masks)
        self.dim = dim

    def log_prob(self, x):
        z, logdet = x, torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
        for layer in self.layers:
            z, ld = layer(z)
            logdet = logdet + ld
        log_base = -0.5 * (z.pow(2).sum(-1) + self.dim * math.log(2 * math.pi))
        return log_base + logdet


# ----------------------------------------------------------------------------
# RLE-Box loss (slots into the vacated DFL position -- see tiny_loss.py)
# ----------------------------------------------------------------------------
class RLEBoxLoss(nn.Module):
    """L_RLE = sum_sides[ log(sig) - log phi(eps) + log(2 sig) + |eps| ].

    Inputs are foreground-gathered tensors in *stride units* (same frame the
    direct-regression head predicts in):
        pred_ltrb   (F, 4)   predicted side distances
        pred_sigma  (F, 4)   sigma in (0,1) (sigmoid already applied)
        target_ltrb (F, 4)   from bbox2dist(anchor_points, target_bboxes/stride)
        weight      (F, 1)   TAL weight = target_scores.sum(-1)[fg]
    Normalized by target_scores_sum, mirroring the existing box-loss reduction.
    Flow runs in fp32 under autocast for numerical safety.
    """

    def __init__(self, flow_dim: int = 4, n_pairs: int = 3, hidden: int = 64):
        super().__init__()
        self.flow = RealNVP(dim=flow_dim, n_pairs=n_pairs, hidden=hidden)

    def forward(self, pred_ltrb, pred_sigma, target_ltrb, weight, target_scores_sum):
        if pred_ltrb.numel() == 0:
            return pred_ltrb.sum() * 0.0
        with torch.autocast(device_type=pred_ltrb.device.type, enabled=False):
            p = pred_ltrb.float()
            s = pred_sigma.float().clamp(1e-4, 1.0)
            t = target_ltrb.float()
            eps = (p - t) / s                                   # (F, 4)

            if self.flow.dim == 4:
                log_phi = self.flow.log_prob(eps)               # (F,)
            else:  # paired-2D ablation: (l,r) and (t,b) through a shared 2D flow
                lr = torch.stack((eps[:, 0], eps[:, 2]), dim=-1)
                tb = torch.stack((eps[:, 1], eps[:, 3]), dim=-1)
                log_phi = self.flow.log_prob(lr) + self.flow.log_prob(tb)

            nll = (s.log().sum(-1)                # change-of-variables:  +log sigma
                   - log_phi                      # learned residual density
                   + ((2 * s).log() + eps.abs()).sum(-1))       # Laplace anchor
            loss = (nll * weight.float().squeeze(-1)).sum() / target_scores_sum.float().clamp_min(1)
        return loss.to(pred_ltrb.dtype)


# ----------------------------------------------------------------------------
# sigma-Rank: uncertainty-calibrated end-to-end ranking (inference-time)
# ----------------------------------------------------------------------------
def sigma_rank_scores(scores, pred_ltrb, pred_sigma, gamma: float = 0.5):
    """s' = s * exp(-gamma * mean_i[ sigma_i / (d_hat_i + 1) ]).

    Apply to the one-to-one branch's class scores BEFORE the top-300 cut
    (and before any conf threshold). gamma=0 recovers the baseline exactly.
    Sweep gamma in {0.25, 0.5, 1.0} on val; report top-300 tiny-recall lift.
    """
    rel = pred_sigma / (pred_ltrb.clamp_min(0) + 1.0)
    q = torch.exp(-gamma * rel.mean(dim=-1, keepdim=True))
    return scores * q


# ----------------------------------------------------------------------------
# Sigma-branch head patch -- TEMPLATE (align with your installed Detect)
# ----------------------------------------------------------------------------
# YOLO26's Detect head (v10-style e2e) holds per-level branch ModuleLists:
#   cv2 (box, 4ch since reg_max=1), cv3 (cls), plus one2one_cv2/one2one_cv3.
# Add a mirrored sigma branch to BOTH paths and change the per-level
# training-output channel contract to:
#
#       cat([ box(4) | sigma(4) | cls(nc) ], dim=1)        # <-- loss splits on this
#
# Template (copy the hidden-width computation from your installed __init__):
#
#   import copy
#   from ultralytics.nn.modules.conv import Conv
#   from ultralytics.nn.modules.head import Detect
#
#   class DetectSigma(Detect):
#       def __init__(self, nc=80, ch=()):
#           super().__init__(nc, ch)
#           c2 = max(16, ch[0] // 4)   # mirror parent's box-branch width calc
#           self.cv_sig = nn.ModuleList(
#               nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4, 1))
#               for x in ch)
#           if hasattr(self, "one2one_cv2"):
#               self.one2one_cv_sig = copy.deepcopy(self.cv_sig)
#
#       # In forward / forward_end2end (copy parent body, insert one line per
#       # branch):   x[i] = cat((cv2[i](f), cv_sig[i](f), cv3[i](f)), 1)
#       # At decode: sigma = raw_sig.sigmoid()  ->  feed sigma_rank_scores()
#       #            just before the topk(300) selection in postprocess().
#
# Register in ultralytics.nn.tasks parse_model (or monkey-patch the module
# symbol) and duplicate yolo26.yaml -> yolo26-tiny.yaml swapping the head
# class name. Weight surgery: COCO-pretrained cv2/cv3 weights load untouched;
# cv_sig starts fresh (zero-init the final 1x1 bias to ~ -2.0 so initial
# sigma ~ 0.12, i.e. mildly confident -- avoids a cold-start loss spike).
#
# Phase discipline: run Phase-1 (SAGA + SD-ProgLoss, NO head change) first;
# add this head only for Phase-2. See EXPERIMENTS.md.
# ============================================================================
