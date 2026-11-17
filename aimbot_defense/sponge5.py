"""Visible energy (sponge) cloak against the RootKit AI-Aimbot post-processing.

RootKit (main_onnx.py / main_tensorrt.py): YOLOv5s 320x320, ONNX or TensorRT forward, then
`non_max_suppression(pred, conf=0.4, iou=0.4, classes=0, agnostic=False, max_det=10)` on the CPU
(ONNX path always moves outputs to CPU), target = detection closest to the screen centre,
mouse gain 0.4, `visuals = False` by default (no box overlay), `cpsDisplay = True`.

Sponge objective: every stride-8 cell, all 3 anchors, outside the real players gets a confident
small box (half the anchor prior, so neighbours mostly survive NMS); the real player is left alone
(variant "sponge") or suppressed too ("sponge+hide").

Reported per config: candidates above 0.4 (NMS input), boxes out, NMS ms with RootKit arguments
on CPU, whether the real player survives RootKit's own max_det=10 cap, closed-loop hit@10 with
RootKit post-processing, and aimbot loop time = inference + NMS.

  python -m aimbot_defense.sponge5
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

from . import capacity as C
from .aimsim import _iou, simulate
from .core import xywh2xyxy

sys.path.insert(0, C.Y5)
from utils.general import non_max_suppression  # noqa: E402

torch.set_num_threads(4)
HALF_PRIORS = torch.tensor([[5.0, 6.5], [8.0, 15.0], [16.5, 11.5]])  # half of P3 anchors (10,13) (16,30) (33,23)


def sponge_masks(persons):
    n = 40
    ph = torch.zeros(6300, dtype=torch.bool)
    ph[:3 * n * n] = True                                   # stride-8 level, all 3 anchors
    wh = torch.zeros(6300, 2)
    wh[:3 * n * n] = HALF_PRIORS.repeat_interleave(n * n, 0)
    pl = torch.zeros(6300, dtype=torch.bool)
    for x1, y1, x2, y2 in persons:
        near = (C.V5C[:, 0] > x1 - 8) & (C.V5C[:, 0] < x2 + 8) & (C.V5C[:, 1] > y1 - 8) & (C.V5C[:, 1] < y2 + 8)
        ph &= ~near
        pl |= (C.V5C[:, 0] > x1 - 4) & (C.V5C[:, 0] < x2 + 4) & (C.V5C[:, 1] > y1 - 4) & (C.V5C[:, 1] < y2 + 4)
    return ph, pl, wh


def sponge_loss(box, s, masks, hide_player):
    s = s.clamp(1e-6, 1 - 1e-6)
    tot = 0.0
    for b, (ph, pl, wh) in enumerate(masks):
        sp = s[b][ph]
        tot = tot - (torch.log(sp) * (sp < 0.9)).sum() \
            + ((box[b][ph][:, 2:] - wh[ph]).abs() / wh[ph]).sum() + ((box[b][ph][:, :2] - C.V5C[ph]).abs() / 4).sum()
        if hide_player:
            sl = s[b][pl]
            tot = tot - 4 * (torch.log(1 - sl) * (sl > 0.05)).sum()
    return tot / len(s)


def craft(models, x, persons, eps, hide_player, eot, steps=60, bs=16, seed=0):
    out = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs]
        masks = [sponge_masks(p) for p in persons[i:i + bs]]
        g = torch.Generator().manual_seed(seed + i)
        d = (torch.rand(xb.shape, generator=g) * 2 - 1) * eps
        a = 2.5 * eps / steps
        for _ in range(steps):
            d.requires_grad_(True)
            xa = (xb + d).clamp(0, 1)
            loss = 0.0
            for m in models:
                box, s = C.person_conf(m, C.eot_transform(xa) if eot else xa)
                loss = loss + sponge_loss(box, s, masks, hide_player)
            gr, = torch.autograd.grad(loss, d)
            d = (d.detach() - a * gr.sign()).clamp(-eps, eps)
            d = (xb + d).clamp(0, 1) - xb
        out.append(((xb + d).clamp(0, 1) * 255).round() / 255)
    return torch.cat(out)


def rootkit_post(m, x):
    """Returns per-frame detections [K,5] (xyxy, conf), #candidates > 0.4, NMS ms (CPU, RootKit args)."""
    with torch.no_grad():
        out = m(x)
    p = out[0] if isinstance(out, (list, tuple)) else out
    dets, cands, ms = [], [], []
    for b in range(len(p)):
        pb = p[b:b + 1].clone()
        cands.append(int((pb[0, :, 4] * pb[0, :, 5] > 0.4).sum()))
        non_max_suppression(pb, 0.4, 0.4, 0, False, max_det=10)  # warm
        t = time.perf_counter()
        for _ in range(5):
            r = non_max_suppression(pb, 0.4, 0.4, 0, False, max_det=10)[0]
        ms.append(1000 * (time.perf_counter() - t) / 5)
        dets.append(torch.cat([r[:, :4], r[:, 4:5]], 1).numpy())
    return dets, cands, ms


def main():
    sp, _ = C.load_frames()
    test = sp["T"][:32]
    x = torch.stack([f.x for f in test])
    P = [f.persons for f in test]
    proxies = [C.load5(os.path.join(C.W, f"yolov5{s}.pt")) for s in "ns"]
    attackers = {s: C.load5(os.path.join(C.W, f"yolov5{s}.pt")) for s in "nsm"}
    inf_ms = {s: C.latency_ms(m) for s, m in attackers.items()}
    variants = {"clean": x}
    for eps in (8, 16):
        for hide in (False, True):
            name = f"sponge{'+hide' if hide else ''} eps={eps}/255"
            variants[name] = craft(proxies, x, P, eps / 255, hide, eot=False)
            print("crafted", name, flush=True)
    rows = []
    for vn, xv in variants.items():
        for an, m in attackers.items():
            dets, cands, ms = rootkit_post(m, xv)
            player_out = np.mean([bool(len(d) and _iou(d[:, :4], f.persons).max() > 0.3) for d, f in zip(dets, test)])
            ttks = []
            for i, (d, f) in enumerate(zip(dets, test)):
                ttks += simulate([d], f.persons, thr=0.4, gain=0.4, starts=12, seed=i, return_ttks=True)["ttks"]
            ttks = np.asarray(ttks)
            r = dict(frames=vn, attacker=f"v5{an}", cand=float(np.mean(cands)), cand_max=int(np.max(cands)),
                     nms_ms=float(np.mean(ms)), nms_ms_max=float(np.max(ms)), inf_ms=inf_ms[an],
                     player_in_top10=float(player_out), hit10=float((ttks <= 10).mean()))
            rows.append(r)
            print(json.dumps(r), flush=True)
    os.makedirs(os.path.join(C.ROOT, "out", "sponge5"), exist_ok=True)
    json.dump(rows, open(os.path.join(C.ROOT, "out", "sponge5", "results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
