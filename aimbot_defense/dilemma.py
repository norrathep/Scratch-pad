"""Dilemma loss: make the attacker's re-training objective fight itself.

When the cheater fine-tunes on harvested protected frames (true labels), the
detection loss contains two parts that the attacker wants to lower together:
  l_pos = -log s(player anchors)        -> keep detecting the player (recall)
  l_neg = -log(1 - s(phantom anchors))  -> suppress phantoms (fewer candidates,
                                           fewer false locks, less runtime)
A gradient step on l_pos + l_neg makes progress on both only if their
parameter gradients are not opposed. We therefore add to the flood cloak a
gradient-cancellation term (first-order surrogate of the bi-level objective
"after K fine-tuning steps the attacker still cannot have both"):

  L = L_flood(delta) + gamma * cos( grad_theta l_pos , grad_theta l_neg )   -> minimise (cos -> -1)

theta = all detector parameters (double-backprop through YOLOv8n). This is
the gradient-matching idea used in data poisoning (Witches' Brew), aimed at the
attacker's re-training rather than at a training-time backdoor.

  python -m aimbot_defense.dilemma
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .aimsim import aggregate, simulate
from .core import ANCH, ANCH_STRIDE, IMG, load_model, raw
from .flood import CAPS, detect_capped, flood_loss, phantom_mask
from .run import OUT, W, splits

torch.set_num_threads(4)


def player_mask(persons):
    m = torch.zeros(len(ANCH), dtype=torch.bool)
    for x1, y1, x2, y2 in persons:
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        m |= ((ANCH[:, 0] - cx).abs() < 0.3 * w) & ((ANCH[:, 1] - cy).abs() < 0.3 * h) & \
             (ANCH_STRIDE <= max(8.0, 0.5 * float(min(w, h))))
    return m


def head_params(m):
    # whole network: the last 1x1 layer alone is saturated (cos ~ -0.8 even on clean frames)
    ps = list(m.parameters())
    for p in ps:
        p.requires_grad_(True)
    return ps


def grad_cos(m, params, xa, pm, nm):
    p = raw(m, xa)
    s = p[:, 4].clamp(1e-6, 1 - 1e-6)
    l_pos = torch.stack([-torch.log(s[b][pm[b]]).mean() for b in range(len(pm)) if pm[b].any()]).mean()
    l_neg = torch.stack([-torch.log(1 - s[b][nm[b]]).mean() for b in range(len(nm))]).mean()
    gp = torch.autograd.grad(l_pos, params, create_graph=True)
    gn = torch.autograd.grad(l_neg, params, create_graph=True)
    gp, gn = torch.cat([g.flatten() for g in gp]), torch.cat([g.flatten() for g in gn])
    return torch.nn.functional.cosine_similarity(gp, gn, 0), p


def craft_dilemma(m, params, x, persons, eps=8 / 255, steps=40, gamma=300.0, seed=0, bs=8):
    out, coss = [], []
    for i in range(0, len(x), bs):
        xb, pb = x[i:i + bs], persons[i:i + bs]
        nm = [phantom_mask(p) for p in pb]
        pm = [player_mask(p) for p in pb]
        g = torch.Generator().manual_seed(seed + i)
        d = (torch.rand(xb.shape, generator=g) * 2 - 1) * eps
        a = 2.5 * eps / steps
        for _ in range(steps):
            d.requires_grad_(True)
            xa = (xb + d).clamp(0, 1)
            cos, p = grad_cos(m, params, xa, pm, nm)
            loss = flood_loss(p, nm, False) + gamma * cos
            gr, = torch.autograd.grad(loss, d)
            d = (d.detach() - a * gr.sign()).clamp(-eps, eps)
            d = (xb + d).clamp(0, 1) - xb
        out.append(((xb + d).clamp(0, 1) * 255).round() / 255)
        coss.append(float(cos.detach()))
        print(f"  {i + len(xb)}/{len(x)} cos={float(cos):+.3f}", flush=True)
    return torch.cat(out), float(np.mean(coss))


def baseline_cos(m, params, x, persons):
    nm = [phantom_mask(p) for p in persons]
    pm = [player_mask(p) for p in persons]
    with torch.enable_grad():
        c, _ = grad_cos(m, params, x, pm, nm)
    return float(c)


def main():
    from ultralytics import YOLO
    from .run import write_dataset
    from .flood import craft_flood
    sp, others = splits()
    m = load_model(os.path.join(W, "yolov8n.pt"))
    params = head_params(m)
    tr, test = sp["B"], sp["T"]
    xb, xt = torch.stack([f.x for f in tr]), torch.stack([f.x for f in test])
    pb, pt = [f.persons for f in tr], [f.persons for f in test]
    xf = craft_flood(m, xt, [phantom_mask(p) for p in pt], 8 / 255, seed=8)
    print("cos on plain-flood test frames:", baseline_cos(m, params, xf, pt), flush=True)
    bank = {f.name: [] for f in tr}
    for k in range(3):
        xa, c = craft_dilemma(m, params, xb, pb, seed=100 + 10 * k)
        print("train variant", k, "cos", c, flush=True)
        for f, xi in zip(tr, xa):
            bank[f.name].append((xi * 255).round().to(torch.uint8))
    bank = {n: torch.stack(v) for n, v in bank.items()}
    xd, cd = craft_dilemma(m, params, xt, pt, seed=8)
    print("test cos", cd, flush=True)
    for p in params:
        p.requires_grad_(False)
    rows = []

    def ev(model, aname, name, xv, caps=CAPS):
        for cap in caps:
            dets, n_cand, _ = detect_capped(model, xv, 0.4, cap)
            r = aggregate([simulate([d], f.persons, thr=0.4, seed=i) for i, (d, f) in enumerate(zip(dets, test))])
            r.update(defence=name, attacker=aname, cap=cap if cap else "none", cand=float(np.mean(n_cand)))
            rows.append(r)
            print(json.dumps(r), flush=True)

    ev(m, "v8n", "dilemma", xd)
    root = os.path.join(OUT, "ds", "dilemma_attacker")
    yaml = write_dataset(root, tr, bank, others)
    YOLO(os.path.join(W, "yolov8n.pt")).train(
        data=yaml, epochs=12, imgsz=IMG, batch=16, device="cpu", workers=2, optimizer="AdamW", lr0=0.001,
        warmup_epochs=1, mosaic=0.0, close_mosaic=0, hsv_h=0, hsv_s=0, hsv_v=0, scale=0.0, translate=0.0,
        fliplr=0.5, val=False, plots=False, amp=False, verbose=False, seed=7,
        project=os.path.join(OUT, "train"), name="dilemma_attacker", exist_ok=True)
    at = load_model(os.path.join(OUT, "train", "dilemma_attacker", "weights", "last.pt"))
    ev(at, "v8n_AT[dilemma]", "clean", xt, (None,))
    ev(at, "v8n_AT[dilemma]", "dilemma", xd)
    json.dump(rows, open(os.path.join(OUT, "dilemma_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
