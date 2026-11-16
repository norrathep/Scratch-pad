"""IC Universal Cloak + "cloak estimation & subtraction" adaptive attack.

IC deploys a static universal perturbation delta_U on every frame, refined on
some frames by <=5 optimisation steps that start from delta_U. Because most of
the deployed signal is the same additive pattern in every frame, a cheater
who harvests N protected frames can estimate it without any clean frames:

    delta_hat = mean_i [ x_i - blur(x_i) ]      (high-pass residual average)

Natural-image high frequencies average towards 0 over varied frames while the
fixed pattern survives. The aimbot then runs on x - delta_hat.

  python -m aimbot_defense.universal
"""
from __future__ import annotations

import json
import os

import torch

from .aimsim import aggregate, simulate
from .core import detect_persons, gauss_blur, load_model, raw
from .craft import _loss
from .run import OUT, W, splits

EPS = 8 / 255
torch.set_num_threads(4)


def train_universal(m, frames, epochs=40, bs=16, lr=1 / 255):
    torch.manual_seed(0)
    d = (torch.rand(1, 3, 320, 320) * 2 - 1) * EPS
    xs = torch.stack([f.x for f in frames])
    for ep in range(epochs):
        perm = torch.randperm(len(xs))
        for i in range(0, len(xs), bs):
            x = xs[perm[i:i + bs]]
            d.requires_grad_(True)
            loss = _loss(raw(m, (x + d).clamp(0, 1)), [torch.zeros(0, 4)] * len(x))
            g, = torch.autograd.grad(loss, d)
            d = (d.detach() - lr * g.sign()).clamp(-EPS, EPS)
        if ep % 10 == 0:
            print("epoch", ep, float(loss), flush=True)
    return d.detach()


def q(x):
    return (x.clamp(0, 1) * 255).round() / 255


def main():
    sp, others = splits()
    m = load_model(os.path.join(W, "yolov8n.pt"))
    p = os.path.join(OUT, "universal.pt")
    if os.path.exists(p):
        dU = torch.load(p)
    else:
        dU = train_universal(m, sp["A"] + sp["B"])
        torch.save(dU, p)
    test = sp["T"]
    xt = torch.stack([f.x for f in test])
    # IC full pipeline: universal + <=5 refinement steps per frame starting at delta_U
    xr = refine(m, xt, dU, steps=5)
    # attacker harvest: every available frame (persons and others) carries delta_U (+ refinement)
    harvest = torch.stack([f.x for f in sp["A"] + sp["B"] + others])
    hu = q(harvest + dU)
    hr = refine(m, harvest, dU, steps=5)

    def est(h):
        return (h - gauss_blur(h, 1.5)).mean(0, keepdim=True)

    rows = []
    conds = {
        "universal": q(xt + dU),
        "universal - est(universal harvest)": q(q(xt + dU) - est(hu)),
        "IC refined": xr,
        "IC refined - est(refined harvest)": q(xr - est(hr)),
        "clean": xt,
    }
    hf_true = dU - gauss_blur(dU, 1.5)
    for name, h in (("universal", hu), ("refined", hr)):
        e = est(h)
        cos = torch.nn.functional.cosine_similarity(e.flatten(), hf_true.flatten(), 0)
        print(f"cosine(est from {name} harvest, true HF of delta_U) = {cos:.3f}")
        rows.append(dict(cond=f"cos(est[{name}], HF(delta_U))", value=float(cos)))
    for name, x in conds.items():
        dets = detect_persons(m, x, conf=0.05)
        for thr in (0.4, 0.15):
            r = aggregate([simulate([d], f.persons, thr=thr, seed=i) for i, (d, f) in enumerate(zip(dets, test))])
            rows.append(dict(cond=name, thr=thr, **r))
            print(name, thr, r, flush=True)
    json.dump(rows, open(os.path.join(OUT, "universal_results.json"), "w"), indent=1)


def refine(m, x, dU, steps=5, bs=16):
    out = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs]
        d = dU.expand_as(xb).clone()
        for _ in range(steps):
            d.requires_grad_(True)
            loss = _loss(raw(m, (xb + d).clamp(0, 1)), [torch.zeros(0, 4)] * len(xb))
            g, = torch.autograd.grad(loss, d)
            d = (d.detach() - (EPS / 4) * g.sign()).clamp(-EPS, EPS)
        out.append(q(xb + d))
    return torch.cat(out)


if __name__ == "__main__":
    main()
