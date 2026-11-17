"""Universal (frame-agnostic) cloaks: one pattern, trained offline, added to every frame at zero runtime cost.

  UV_OBJ=band  : universal phantom flood with the player kept detected (player band [0.5, 0.7], phantoms >= 0.95)
  UV_OBJ=hide  : universal Invisibility-Cloak-style pattern (every person confidence -> 0)

Trained on splits A+B (280 person scenes) against YOLOv5 n/s/m with EOT; evaluated on the 80 held-out
test scenes with RootKit post-processing and the cheater's counter-measures (`sponge_band.evaluate`).

  UV_OBJ=band python -m aimbot_defense.universal5
"""
from __future__ import annotations

import json
import os
import time

import torch

from . import capacity as C
from . import sponge_band as S

torch.set_num_threads(4)
OBJ = os.environ.get("UV_OBJ", "band")
EPS = float(os.environ.get("UV_EPS", "8")) / 255
EPOCHS = int(os.environ.get("UV_EPOCHS", "12"))
OUT = os.path.join(C.ROOT, "out", f"paper_universal_{OBJ}")


def train(models, frames, bs=16, lr=0.5 / 255, seed=0):
    S.SB_OBJ = OBJ  # band_loss reads this
    g = torch.Generator().manual_seed(seed)
    d = (torch.rand(1, 3, C.IMG, C.IMG, generator=g) * 2 - 1) * EPS
    xs = torch.stack([f.x for f in frames])
    for ep in range(EPOCHS):
        perm = torch.randperm(len(xs), generator=g)
        tot = 0.0
        for i in range(0, len(xs), bs):
            idx = perm[i:i + bs]
            xb = xs[idx]
            fr = [frames[j] for j in idx.tolist()]
            masks = [S.sponge_masks(f.persons) for f in fr]
            pms = [S.player_masks(f.persons) for f in fr]
            d.requires_grad_(True)
            xa = (xb + d).clamp(0, 1)
            loss = 0.0
            for m in models:
                box, s = C.person_conf(m, C.eot_transform(xa))
                loss = loss + S.band_loss(box, s, masks, pms)
                box, s = C.person_conf(m, xa)
                loss = loss + S.band_loss(box, s, masks, pms)
            gr, = torch.autograd.grad(loss, d)
            d = (d.detach() - lr * gr.sign()).clamp(-EPS, EPS)
            tot += float(loss)
        print(f"  epoch {ep + 1}/{EPOCHS} loss {tot:.1f}", flush=True)
    return d.detach()


def main():
    os.makedirs(OUT, exist_ok=True)
    sp, _ = C.load_frames()
    proxies = [C.load5(os.path.join(C.W, f"yolov5{s}.pt")) for s in "nsm"]
    p = os.path.join(OUT, "delta.pt")
    if os.path.exists(p):
        d = torch.load(p)
    else:
        t = time.time()
        print(f"[train universal {OBJ}] eps={EPS * 255:.0f}/255", flush=True)
        d = train(proxies, sp["A"] + sp["B"])
        torch.save(d, p)
        print(f"  done in {time.time() - t:.0f}s", flush=True)
    test = sp["T"]
    x = torch.stack([f.x for f in test])
    xu = ((x + d).clamp(0, 1) * 255).round() / 255
    t = time.perf_counter()
    for _ in range(100):
        ((x[:1] + d).clamp(0, 1) * 255).round()
    add_ms = 1000 * (time.perf_counter() - t) / 100
    print(f"[runtime] applying the universal pattern: {add_ms:.3f} ms/frame (CPU, unoptimised)", flush=True)
    attackers = {f"v5{s}": ("v5", C.load5(os.path.join(C.W, f"yolov5{s}.pt"))) for s in "nsml"}
    attackers["v8n"] = ("v8", S.load_v8(os.path.join(C.W, "yolov8n.pt")))
    rows = S.evaluate(attackers, {"clean": x, f"universal {OBJ}": xu}, test)
    json.dump(dict(apply_ms=add_ms, rows=rows), open(os.path.join(OUT, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
