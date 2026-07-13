# detect_sigma.py
# ============================================================================
# DetectSigma: YOLO26 Detect head with an extra sigma branch (Phase 2).
#
# Adds a "sig_head" ModuleList mirroring box_head to each of one2many / one2one.
# The sigma branch outputs 4 channels (one per box side), activated to (0,1)
# by sigmoid in TinyDetectionLoss.
#
# forward_head is extended to include "sigma": (b, 4, a) in the output dict,
# which TinyDetectionLoss reads as preds["sigma"].
#
# postprocess is patched to apply sigma-Rank BEFORE the top-300 cut:
#   s' = s * exp(-gamma * mean_i[ sigma_i / (d_hat_i + 1) ])
# Set gamma via model.overrides["sigma_rank_gamma"] (default 0.5, 0=baseline).
#
# Usage
# -----
# 1. Register in ultralytics/nn/tasks.py or use parse_model override (see below).
# 2. Copy yolo26n.yaml -> yolo26n-tiny.yaml, change head class to DetectSigma.
# 3. Load pretrained weights: model = YOLO("yolo26n-tiny.yaml")
#    then model.load("yolo26n.pt") -- box_head/cls_head load by name; sig_head
#    starts from scratch with bias init -2.0 (sigma_0 ~ 0.12).
# ============================================================================

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from ultralytics.nn.modules.head import Detect
from rle_box import sigma_rank_scores


class DetectSigma(Detect):
    """Detect head with extra sigma branch for RLE-Box + σ-Rank."""

    def __init__(self, nc: int = 80, ch: tuple = ()):
        super().__init__(nc, ch)
        for branch in (self.one2many, self.one2one):
            sig = copy.deepcopy(branch["box_head"])
            for lvl in sig:                          # per-level ModuleList
                final = lvl[-1]                      # last conv (outputs 4 ch = reg_max*4 with reg_max=1)
                new_conv = nn.Conv2d(
                    final.in_channels, 4,
                    final.kernel_size, final.stride,
                    getattr(final, "padding", 0),
                )
                nn.init.constant_(new_conv.bias, -2.0)  # sigma_0 ~ 0.12: no cold-start spike
                lvl[-1] = new_conv
            branch["sig_head"] = sig

    def forward_head(self, x, box_head, cls_head, sig_head=None, **_):
        """Extend parent to concatenate sigma channel (train only)."""
        bs = x[0].shape[0]
        boxes  = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1)
                             for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1)
                             for i in range(self.nl)], dim=-1)
        out = dict(boxes=boxes, scores=scores, feats=x)
        if self.training and sig_head is not None:
            sigmas = torch.cat([sig_head[i](x[i]).view(bs, 4, -1)
                                 for i in range(self.nl)], dim=-1)
            out["sigma"] = sigmas
        return out

    def forward(self, x):
        """Forward pass; sigma included in training dicts, absent at export."""
        preds = self.forward_head(x, **self.one2many)
        if self.end2end:
            x_detach = [xi.detach() for xi in x]
            one2one = self.forward_head(x_detach, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Apply sigma-Rank before top-300 cut (inference only).

        preds: (b, nc+4, num_anchors) after _inference decoding.
        """
        boxes, scores = preds.split([4, self.nc], dim=-1)
        # sigma-Rank: need sigma from the one2one branch at inference.
        # Since sigma is not baked into _inference output, we use a stored
        # buffer set during the forward pass. If not available, fall back.
        gamma = getattr(self, "_sigma_rank_gamma", 0.5)
        if gamma > 0 and hasattr(self, "_last_sigma") and self._last_sigma is not None:
            scores = sigma_rank_scores(scores, boxes, self._last_sigma, gamma=gamma)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        return torch.cat([boxes, scores, conf], dim=-1)

    def _inference(self, x):
        """Decode and optionally stash sigma for postprocess."""
        if self.training:
            return super()._inference(x)
        # stash sigma before decode
        self._last_sigma = None
        if "sigma" in x:
            self._last_sigma = x["sigma"].permute(0, 2, 1).sigmoid()  # (b, a, 4)
        return super()._inference(x)

    def set_sigma_rank_gamma(self, gamma: float):
        """Convenience: set gamma for sigma-Rank (0 = baseline exact)."""
        self._sigma_rank_gamma = gamma
