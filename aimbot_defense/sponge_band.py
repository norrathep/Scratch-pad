"""Visible phantom flood with the real player kept *confidently detected* (but below the phantoms).

Proxies: YOLOv5 n/s/m (IC-style YOLOv5 ensemble), EOT on. Objective per frame:
  phantoms : every stride-8 cell / anchor outside players -> confidence >= 0.9 (hinge), small half-prior boxes
  player   : best box held in a "normal" band [0.6, 0.8] -> the cheater still sees the player detected,
             but the player never out-ranks the phantoms (otherwise "keep the top-1 box" would be a free fix)

Evaluation uses the RootKit AI-Aimbot post-processing (conf 0.4, IoU 0.4, max_det 10, closest-to-centre,
gain 0.4) and the cheater's cheap counter-measures:
  stock     max_det=10                (what RootKit ships)
  top1      max_det=1                 (keep only the most confident box)
  nocap     max_det=1000              (keep everything)
  thr0.85   conf 0.85, max_det=10     (raise the threshold)
Attackers: v5n/s/m (proxies) and v5l, v8n (never used by the defender).

  python -m aimbot_defense.sponge_band
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import torchvision

from . import capacity as C
from .aimsim import _iou, simulate
from .core import load_model as load_v8, raw as raw_v8, xywh2xyxy
from .sponge5 import non_max_suppression, sponge_masks

torch.set_num_threads(4)
BAND = (0.6, 0.8)
PH_MIN = 0.9
COUNTERS = {"stock": (0.4, 10), "top1": (0.4, 1), "nocap": (0.4, 1000), "thr0.85": (0.85, 10)}
OUT = os.path.join(C.ROOT, "out", "sponge_band")


def player_masks(persons):
    ms = []
    for x1, y1, x2, y2 in persons:
        ms.append((C.V5C[:, 0] > x1 - 4) & (C.V5C[:, 0] < x2 + 4) & (C.V5C[:, 1] > y1 - 4) & (C.V5C[:, 1] < y2 + 4))
    return ms


def band_loss(box, s, masks, pms):
    s = s.clamp(1e-6, 1 - 1e-6)
    tot = 0.0
    for b, ((ph, _, wh), plist) in enumerate(zip(masks, pms)):
        sp = s[b][ph]
        tot = tot - (torch.log(sp) * (sp < PH_MIN + 0.03)).sum() \
            + ((box[b][ph][:, 2:] - wh[ph]).abs() / wh[ph]).sum() + ((box[b][ph][:, :2] - C.V5C[ph]).abs() / 4).sum()
        for pm in plist:
            sl = s[b][pm]
            tot = tot - 4 * (torch.log(1 - sl) * (sl > BAND[1])).sum()      # no player box above the band
            top = sl.max()
            if top < BAND[0] + 0.05:                                          # best player box stays detected
                tot = tot - 4 * torch.log(top)
    return tot / len(s)


def craft(models, x, persons, eps, steps=60, bs=16, seed=0):
    out = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs]
        masks = [sponge_masks(p) for p in persons[i:i + bs]]
        pms = [player_masks(p) for p in persons[i:i + bs]]
        g = torch.Generator().manual_seed(seed + i)
        d = (torch.rand(xb.shape, generator=g) * 2 - 1) * eps
        a = 2.5 * eps / steps
        for _ in range(steps):
            d.requires_grad_(True)
            xa = (xb + d).clamp(0, 1)
            loss = 0.0
            for m in models:
                box, s = C.person_conf(m, C.eot_transform(xa))
                loss = loss + band_loss(box, s, masks, pms)
            gr, = torch.autograd.grad(loss, d)
            d = (d.detach() - a * gr.sign()).clamp(-eps, eps)
            d = (xb + d).clamp(0, 1) - xb
        out.append(((xb + d).clamp(0, 1) * 255).round() / 255)
        print(f"    {i + len(xb)}/{len(x)}", flush=True)
    return torch.cat(out)


def raw_preds(kind, m, x):
    """Returns per-image YOLOv5-style [N,6] tensors (xywh, obj, cls0) for NMS."""
    with torch.no_grad():
        if kind == "v5":
            out = m(x)
            p = out[0] if isinstance(out, (list, tuple)) else out
            return [p[b:b + 1, :, :6].clone() for b in range(len(p))]
        p = raw_v8(m, x)                                   # [B, 84, N]
        res = []
        for b in range(len(p)):
            q = torch.zeros(1, p.shape[2], 6)
            q[0, :, :4] = p[b, :4].T
            q[0, :, 4] = p[b, 4]
            q[0, :, 5] = 1.0
            res.append(q)
        return res


def post(pb, conf, max_det):
    t = time.perf_counter()
    r = non_max_suppression(pb, conf, 0.4, 0, False, max_det=max_det)[0]
    return torch.cat([r[:, :4], r[:, 4:5]], 1).numpy(), 1000 * (time.perf_counter() - t)


def evaluate(attackers, variants, test):
    rows = []
    for an, (kind, m) in attackers.items():
        lat = C.latency_ms(m) if kind == "v5" else 0.0
        for vn, xv in variants.items():
            preds = raw_preds(kind, m, xv)
            cands = [int((p[0, :, 4] * p[0, :, 5] > 0.4).sum()) for p in preds]
            # player's best confidence (uncapped detections overlapping the player)
            pconf = []
            for p, f in zip(preds, test):
                d, _ = post(p, 0.05, 1000)
                ov = _iou(d[:, :4], f.persons).max(1) if len(d) else np.zeros(0)
                pconf.append(float(d[ov > 0.5, 4].max()) if (ov > 0.5).any() else 0.0)
            for cn, (conf, md) in COUNTERS.items():
                dets, ms = zip(*[post(p, conf, md) for p in preds])
                player = np.mean([bool(len(d) and _iou(d[:, :4], f.persons).max() > 0.3) for d, f in zip(dets, test)])
                ttks = []
                for i, (d, f) in enumerate(zip(dets, test)):
                    ttks += simulate([d], f.persons, thr=0.0, gain=0.4, starts=12, seed=i, return_ttks=True)["ttks"]
                ttks = np.asarray(ttks)
                r = dict(attacker=an, frames=vn, counter=cn, cand=float(np.mean(cands)), nms_ms=float(np.mean(ms)),
                         inf_ms=lat, player_conf_median=float(np.median(pconf)), player_in_output=float(player),
                         hit10=float((ttks <= 10).mean()))
                rows.append(r)
                print(json.dumps(r), flush=True)
    return rows


def main():
    sp, _ = C.load_frames()
    test = sp["T"][:32]
    x = torch.stack([f.x for f in test])
    P = [f.persons for f in test]
    os.makedirs(OUT, exist_ok=True)
    proxies = [C.load5(os.path.join(C.W, f"yolov5{s}.pt")) for s in "nsm"]
    variants = {"clean": x}
    for eps in (8, 16):
        p = os.path.join(OUT, f"band_eps{eps}.pt")
        if os.path.exists(p):
            variants[f"band eps={eps}/255"] = torch.load(p)
            continue
        t = time.time()
        print(f"[craft] eps={eps}/255", flush=True)
        variants[f"band eps={eps}/255"] = craft(proxies, x, P, eps / 255, seed=eps)
        torch.save(variants[f"band eps={eps}/255"], p)
        print(f"  done in {time.time() - t:.0f}s", flush=True)
    attackers = {f"v5{s}": ("v5", C.load5(os.path.join(C.W, f"yolov5{s}.pt"))) for s in "nsml"}
    attackers["v8n"] = ("v8", load_v8(os.path.join(C.W, "yolov8n.pt")))
    rows = evaluate(attackers, variants, test)
    json.dump(rows, open(os.path.join(OUT, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
