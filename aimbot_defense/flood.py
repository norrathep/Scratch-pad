"""Does capping the candidate boxes defeat a phantom-flood (sponge) cloak?

Sponge/NMS-overload cloaks (Phantom Sponges, Overload) add many phantom boxes.
The obvious counter is to cap candidates (ultralytics: conf filter -> keep the
top `max_nms` by score -> NMS -> `max_det`). But if the phantoms out-score the
real player, a small cap keeps only phantoms and the player disappears. This
script measures that trade-off: for each cap k we report the real-player
recall, closed-loop hit@15, false locks, and the NMS cost the uncapped
attacker would pay.

  python -m aimbot_defense.flood
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import torchvision

from .aimsim import aggregate, simulate
from .core import ANCH, ANCH_STRIDE, IMG, eot_transform, load_model, raw, xywh2xyxy
from .run import OUT, W, splits

torch.set_num_threads(4)
CAPS = [1, 3, 10, 30, 100, 300, None]


def phantom_mask(persons, every=2, stride=8):
    """Lattice of stride-8 anchors (every `every`-th cell) outside real players."""
    n = IMG // stride
    idx = torch.arange(n * n)
    gy, gx = idx // n, idx % n
    m = (gy % every == 0) & (gx % every == 0)
    full = torch.zeros(len(ANCH), dtype=torch.bool)
    full[:n * n] = m
    for x1, y1, x2, y2 in persons:
        inside = (ANCH[:, 0] > x1 - 8) & (ANCH[:, 0] < x2 + 8) & (ANCH[:, 1] > y1 - 8) & (ANCH[:, 1] < y2 + 8)
        full &= ~inside
    return full


def flood_loss(p, masks, suppress_player):
    s = p[:, 4].clamp(1e-6, 1 - 1e-6)
    box = p[:, :4].transpose(1, 2)
    tgt_wh = torch.tensor([14.0, 28.0])  # small person-like boxes: neighbours 16 px apart never merge in NMS
    total = 0.0
    for b, m in enumerate(masks):
        att = -torch.log(s[b][m]).sum()
        att = att + ((box[b][m][:, 2:] - tgt_wh).abs() / tgt_wh).sum() \
                  + ((box[b][m][:, :2] - ANCH[m]).abs() / 8).sum()
        total = total + att
        if suppress_player:
            total = total - torch.log(1 - s[b][~m]).sum()
    return total / len(masks)


def craft_flood(m, x, masks, eps, steps=40, suppress_player=False, eot=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    d = (torch.rand(x.shape, generator=g) * 2 - 1) * eps
    a = 2.5 * eps / steps
    for _ in range(steps):
        d.requires_grad_(True)
        xa = (x + d).clamp(0, 1)
        loss = flood_loss(raw(m, eot_transform(xa) if eot else xa), masks, suppress_player)
        gr, = torch.autograd.grad(loss, d)
        d = (d.detach() - a * gr.sign()).clamp(-eps, eps)
        d = (x + d).clamp(0, 1) - x
    return ((x + d).clamp(0, 1) * 255).round() / 255


@torch.no_grad()
def detect_capped(m, x, thr, cap, iou=0.45):
    """ultralytics order: conf filter -> top-`cap` by score -> NMS. Also times NMS."""
    p = raw(m, x)
    out, n_cand, t_nms = [], [], []
    for b in range(p.shape[0]):
        sc = p[b, 4]
        keep = sc > thr
        boxes, sc = xywh2xyxy(p[b, :4].T)[keep], sc[keep]
        n_cand.append(len(sc))
        if cap is not None and len(sc) > cap:
            top = sc.topk(cap).indices
            boxes, sc = boxes[top], sc[top]
        t = time.perf_counter()
        k = torchvision.ops.nms(boxes, sc, iou) if len(sc) else torch.zeros(0, dtype=torch.long)
        t_nms.append(time.perf_counter() - t)
        out.append(torch.cat([boxes[k], sc[k, None]], 1).numpy())
    return out, n_cand, t_nms


def main():
    sp, _ = splits()
    test = sp["T"]
    m = load_model(os.path.join(W, "yolov8n.pt"))
    x = torch.stack([f.x for f in test])
    masks = [phantom_mask(f.persons) for f in test]
    t = time.perf_counter()
    with torch.no_grad():
        raw(m, x[:1])
    t_fwd = time.perf_counter() - t
    variants = {"clean": x}
    for eps in (8, 16):
        for sup in (False, True):
            name = f"flood{'+hide' if sup else ''} eps={eps}/255"
            variants[name] = craft_flood(m, x, masks, eps / 255, suppress_player=sup, seed=eps)
            print("crafted", name, flush=True)
    rows = []
    for name, xv in variants.items():
        for cap in CAPS:
            dets, n_cand, t_nms = detect_capped(m, xv, 0.4, cap)
            r = aggregate([simulate([d], f.persons, thr=0.4, seed=i) for i, (d, f) in enumerate(zip(dets, test))])
            r.update(defence=name, cap=cap if cap else "none", cand=float(np.mean(n_cand)),
                     kept=float(np.mean([len(d) for d in dets])), nms_ms=1000 * float(np.mean(t_nms)),
                     fwd_ms=1000 * t_fwd)
            rows.append(r)
            print(json.dumps(r), flush=True)
    json.dump(rows, open(os.path.join(OUT, "flood_results.json"), "w"), indent=1)




def adaptive():
    """Attacker fine-tunes YOLOv8n on harvested flood frames (true labels), then faces fresh flood frames."""
    from ultralytics import YOLO
    from .run import write_dataset
    sp, others = splits()
    m = load_model(os.path.join(W, "yolov8n.pt"))
    tr = sp["B"]
    xb = torch.stack([f.x for f in tr])
    mb = [phantom_mask(f.persons) for f in tr]
    bank = {f.name: [] for f in tr}
    for k in range(3):
        xa = craft_flood(m, xb, mb, 8 / 255, seed=100 + k)
        for f, xi in zip(tr, xa):
            bank[f.name].append((xi * 255).round().to(torch.uint8))
    bank = {n: torch.stack(v) for n, v in bank.items()}
    root = os.path.join(OUT, "ds", "flood_attacker")
    yaml = write_dataset(root, tr, bank, others)
    YOLO(os.path.join(W, "yolov8n.pt")).train(
        data=yaml, epochs=12, imgsz=IMG, batch=16, device="cpu", workers=2, optimizer="AdamW", lr0=0.001,
        warmup_epochs=1, mosaic=0.0, close_mosaic=0, hsv_h=0, hsv_s=0, hsv_v=0, scale=0.0, translate=0.0,
        fliplr=0.5, val=False, plots=False, amp=False, verbose=False, seed=7,
        project=os.path.join(OUT, "train"), name="flood_attacker", exist_ok=True)
    at = load_model(os.path.join(OUT, "train", "flood_attacker", "weights", "last.pt"))
    test = sp["T"]
    x = torch.stack([f.x for f in test])
    xf = craft_flood(m, x, [phantom_mask(f.persons) for f in test], 8 / 255, seed=8)  # defender proxy = base v8n
    rows = []
    for name, xv in (("clean", x), ("flood eps=8/255", xf)):
        for cap in CAPS:
            dets, n_cand, _ = detect_capped(at, xv, 0.4, cap)
            r = aggregate([simulate([d], f.persons, thr=0.4, seed=i) for i, (d, f) in enumerate(zip(dets, test))])
            r.update(defence=name, attacker="v8n_AT[flood]", cap=cap if cap else "none", cand=float(np.mean(n_cand)))
            rows.append(r)
            print(json.dumps(r), flush=True)
    # transfer: unseen YOLOv8s
    s = load_model(os.path.join(W, "yolov8s.pt"))
    for cap in (10, None):
        dets, n_cand, _ = detect_capped(s, xf, 0.4, cap)
        r = aggregate([simulate([d], f.persons, thr=0.4, seed=i) for i, (d, f) in enumerate(zip(dets, test))])
        r.update(defence="flood eps=8/255", attacker="v8s", cap=cap if cap else "none", cand=float(np.mean(n_cand)))
        rows.append(r)
        print(json.dumps(r), flush=True)
    json.dump(rows, open(os.path.join(OUT, "flood_adaptive_results.json"), "w"), indent=1)


if __name__ == "__main__":
    import sys
    adaptive() if sys.argv[1:] == ["adaptive"] else main()
