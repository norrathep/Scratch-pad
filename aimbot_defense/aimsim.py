"""Closed-loop aimbot simulator.

A typical open-source visual aimbot loop (e.g. the YOLO-based ones IC evaluates):
  capture frame -> detect persons (conf > thr) -> pick the box closest to the
  crosshair -> move the mouse a fraction `gain` of the way to the box centre
  (smoothing to look human) -> fire when locked.

We emulate camera motion by moving the crosshair over a static frame, and model
the defence as re-drawing a fresh perturbation each frame from a bank of K
crafted variants. Metrics:
  hit@15   P(crosshair enters a real hitbox within 15 frames = 250 ms @ 60 FPS,
           roughly human visual reaction time). Main "is the aimbot still
           superhuman" number.
  hit@60   P(hit within 1 s).
  ttk      median frames-to-hit (61 = never within 1 s).
  fake     mean number of "locked on something that is not a player" frames per
           engagement (aimbot would fire at a phantom; honeypot evidence).
  det      per-frame rate at which some detection overlaps a real player
           (IoU > 0.3), i.e. 1 - IC's defence-success-rate.
"""
from __future__ import annotations

import math
import random

import numpy as np


def _iou(a, b):
    x1, y1 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None] - inter + 1e-9)


def hit_region(p):
    cx, cy, w, h = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2, p[2] - p[0], p[3] - p[1]
    return cx - w / 4, cy - h / 4, cx + w / 4, cy + h / 4


def simulate(bank, persons, thr=0.4, starts=12, T=60, gain=0.5, lock_tol=3.0, seed=0, return_ttks=False):
    """bank: list (len K) of [D,5] detections (xyxy, conf) for one scene."""
    rng = random.Random(seed)
    regions = [hit_region(p) for p in persons]
    det_hits = []
    for d in bank:
        d = d[d[:, 4] > thr] if len(d) else d
        det_hits.append(float(len(d) > 0 and _iou(d[:, :4], persons).max() > 0.3))
    ttks, fakes = [], []
    for _ in range(starts):
        p = persons[rng.randrange(len(persons))]
        cx, cy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
        a, r = rng.uniform(0, 2 * math.pi), rng.uniform(50, 110)
        c = np.array([np.clip(cx + r * math.cos(a), 0, 319), np.clip(cy + r * math.sin(a), 0, 319)])
        ttk, fake = T + 1, 0
        for t in range(T):
            d = bank[rng.randrange(len(bank))]
            d = d[d[:, 4] > thr] if len(d) else d
            if len(d) == 0:
                continue
            centres = np.stack([(d[:, 0] + d[:, 2]) / 2, (d[:, 1] + d[:, 3]) / 2], 1)
            j = int(np.argmin(((centres - c) ** 2).sum(1)))
            c = c + gain * (centres[j] - c)
            if any(x1 <= c[0] <= x2 and y1 <= c[1] <= y2 for x1, y1, x2, y2 in regions):
                ttk = t + 1
                break
            if np.linalg.norm(centres[j] - c) < lock_tol:
                fake += 1
        ttks.append(ttk)
        fakes.append(fake)
    ttks = np.asarray(ttks)
    r = dict(hit15=float((ttks <= 15).mean()), hit60=float((ttks <= T).mean()),
             ttk=float(np.median(ttks)), fake=float(np.mean(fakes)), det=float(np.mean(det_hits)))
    if return_ttks:
        r["ttks"] = ttks.tolist()
    return r


def aggregate(rows):
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}
