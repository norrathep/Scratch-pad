"""Perturbation objectives and the PGD crafter.

Objectives (all under an l_inf budget eps, quantised to 8 bit at the end):

  hide    Invisibility-Cloak style: push every person score towards 0 (BCE, Eq. 3
          of the IC paper). Goal: aimbot sees nothing.
  decoy   Suppress the real player AND synthesise a confident phantom player box
          1.0-1.6 player-sizes away in a random direction. Goal: aimbot locks on
          the wrong place (slows it down, and every lock on a phantom is
          honeypot-style evidence).
  drift   Same mechanism with a small offset (0.45-0.7 player-sizes): the
          detection still overlaps the player but its centre (the aim point) is
          pushed off the hitbox. Goal: aimbot is "almost right", keeps missing.

decoy/drift are re-sampled every frame (new random direction), so a closed-loop
aimbot is dragged in a different direction each frame -> it oscillates instead
of converging. That is the "slow the aimbot down" defence, as opposed to the
"make the aimbot blind" defence.
"""
from __future__ import annotations

import math
import random

import numpy as np
import torch

from .core import ANCH, ANCH_STRIDE, IMG, eot_transform, raw, xywh2xyxy

OFFSETS = {"decoy": (1.0, 1.6), "drift": (0.45, 0.7)}


def sample_targets(persons: np.ndarray, kind: str, rng: random.Random):
    """For decoy/drift: one fake box (xyxy) per real person."""
    if kind not in OFFSETS:
        return np.zeros((0, 4), np.float32)
    lo, hi = OFFSETS[kind]
    out = []
    for x1, y1, x2, y2 in persons:
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        s = max(w, h) if kind == "decoy" else max(w, 0.6 * h)
        for _ in range(20):
            a = rng.uniform(0, 2 * math.pi)
            d = rng.uniform(lo, hi) * s
            nx, ny = cx + d * math.cos(a), cy + d * math.sin(a)
            if w / 2 <= nx <= IMG - w / 2 and h / 2 <= ny <= IMG - h / 2:
                break
        nx, ny = float(np.clip(nx, w / 2, IMG - w / 2)), float(np.clip(ny, h / 2, IMG - h / 2))
        out.append([nx - w / 2, ny - h / 2, nx + w / 2, ny + h / 2])
    return np.asarray(out, np.float32)


def _loss(p: torch.Tensor, fakes: list[torch.Tensor]) -> torch.Tensor:
    """p: [B, 4+nc, N] raw YOLOv8 output; fakes: per-image [F,4] xyxy targets."""
    s = p[:, 4].clamp(1e-6, 1 - 1e-6)           # person score [B, N]
    box = p[:, :4].transpose(1, 2)              # [B, N, 4] cxcywh
    total = 0.0
    for b in range(p.shape[0]):
        attract = torch.zeros(s.shape[1], dtype=torch.bool)
        tgt = torch.zeros(s.shape[1], 4)
        for fb in fakes[b]:
            fcx, fcy = (fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2
            fw, fh = fb[2] - fb[0], fb[3] - fb[1]
            near = ((ANCH[:, 0] - fcx).abs() < 0.3 * fw) & ((ANCH[:, 1] - fcy).abs() < 0.3 * fh)
            near &= ANCH_STRIDE <= max(8.0, 0.5 * float(min(fw, fh)))
            attract |= near
            tgt[near] = torch.tensor([fcx, fcy, fw, fh], dtype=torch.float32)
        hide = -torch.log(1 - s[b][~attract]).sum()
        if attract.any():
            bt = tgt[attract]
            l1 = ((box[b][attract] - bt).abs() / bt[:, 2:].repeat(1, 2)).sum(-1)
            att = (-torch.log(s[b][attract]) + 2.0 * l1).sum()
        else:
            att = 0.0
        total = total + hide + att
    return total / p.shape[0]


def craft(models, x: torch.Tensor, fakes: list[np.ndarray], eps: float = 8 / 255,
          steps: int = 20, eot: bool = False, seed: int = 0) -> torch.Tensor:
    """Returns the perturbed, 8-bit-quantised frames x_adv (same shape as x)."""
    g = torch.Generator().manual_seed(seed)
    fk = [torch.as_tensor(f, dtype=torch.float32) for f in fakes]
    delta = (torch.rand(x.shape, generator=g) * 2 - 1) * eps
    alpha = 2.5 * eps / steps
    for _ in range(steps):
        delta.requires_grad_(True)
        xa = (x + delta).clamp(0, 1)
        loss = 0.0
        for m in models:
            loss = loss + _loss(raw(m, eot_transform(xa) if eot else xa), fk)
        grad, = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            delta = (delta - alpha * grad.sign()).clamp(-eps, eps)
            delta = (x + delta).clamp(0, 1) - x
    xa = (x + delta.detach()).clamp(0, 1)
    return (xa * 255).round() / 255
