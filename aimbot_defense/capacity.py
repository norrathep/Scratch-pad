"""Capacity dilemma: make *model size* (hence per-frame latency) the attacker's cost.

Setting matches Invisibility Cloak / AimTrap: the RootKit AI-Aimbot uses
original (anchor-based) YOLOv5, 320x320 capture, conf threshold 0.4, and a
proportional mouse controller with gain 0.4. With gain 0.4 the aimbot needs
several detections to converge, so

    time-to-hit (ms)  ~=  #controller steps  x  max(frame interval, inference latency)

i.e. a bigger detector directly slows the aimbot down.

Defender's loss ("encode the dilemma"): the cloak is optimised to defeat every
detector that is *fast enough to be worth using*, including versions of them
adversarially fine-tuned on harvested cloaked frames:

    round 1:  L = sum_{M in {v5n, v5s}}               L_hide(M, EOT(x + delta))
    round 2:  L = sum_{M in {v5n, v5s, v5n-AT, v5s-AT}} L_hide(M, EOT(x + delta))

v5m is deliberately left out: the defender concedes that a big enough model can
be retrained to see through the cloak, and makes sure it must be big (slow).

Stages (cached):
  python -m aimbot_defense.capacity craft1 | train1 | craft2 | train2 | eval
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
import torchvision

from .aimsim import simulate
from .core import IMG, ROOT, Frame, eot_transform, letterbox, xywh2xyxy

Y5 = os.path.join(ROOT, "third_party", "yolov5")
sys.path.insert(0, Y5)
OBJ = os.environ.get("CAP_OBJ", "hide")  # "hide" (IC-style) | "flood" (hide player + phantoms) | "dilemma" (flood + GC)
OUT = os.path.join(ROOT, "out", "capacity" if OBJ == "hide" else f"capacity_{OBJ}")
W = os.path.join(ROOT, "weights")
EPS, K_TEST = 8 / 255, 2
STEPS = 30 if OBJ == "hide" else 40     # flood needs more steps to grow phantoms and hide the player
FLOOD_OUT = os.path.join(ROOT, "out", "capacity_flood")
GAMMA = float(os.environ.get("CAP_GAMMA", "300"))  # weight of the gradient-conflict term (dilemma only)
EPOCHS = 5
GAIN = 0.4               # RootKit AI-Aimbot default aaMovementAmp (AimTrap App. I)
torch.set_num_threads(4)


# ------------------------------------------------------------------ data
def load_frames():
    base = os.path.join(ROOT, "data", "coco")
    P, O = [], []
    for line in open(os.path.join(ROOT, "data", "coco_sel.txt")):
        kind, name = line.split()
        img = cv2.cvtColor(cv2.imread(os.path.join(base, "images", "val2017", name + ".jpg")), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        arr = np.loadtxt(os.path.join(base, "labels", "val2017", name + ".txt"), ndmin=2).astype(np.float32)
        c, cx, cy, bw, bh = arr.T
        lab = np.stack([c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], 1)
        lb, boxes = letterbox(img, lab[:, 1:])
        lab = np.concatenate([lab[:, :1], boxes], 1)
        x = torch.from_numpy(lb).permute(2, 0, 1).float() / 255
        p = lab[lab[:, 0] == 0, 1:]
        p = p[(p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1]) / IMG ** 2 >= 0.015]
        (P if kind == "P" and len(p) else O).append(Frame(name, x, p, lab))
    return dict(A=P[:120], B=P[120:280], T=P[280:360]), O


# ----------------------------------------------------------------- models
def load5(path, fuse=True):
    from models.experimental import attempt_load
    m = attempt_load(path, device="cpu", fuse=fuse).float().eval()
    for q in m.parameters():
        q.requires_grad_(False)
    return m


def person_conf(m, x):
    """YOLOv5 inference head: [B, N, 85] = xywh, obj, cls. Person conf = obj * cls0 (as in YOLOv5 NMS)."""
    out = m(x)
    p = out[0] if isinstance(out, (list, tuple)) else out
    return p[..., :4], p[..., 4] * p[..., 5]


@torch.no_grad()
def detect5(m, x, conf=0.05, iou=0.45, bs=16):
    res = []
    for i in range(0, len(x), bs):
        box, s = person_conf(m, x[i:i + bs])
        for b in range(len(s)):
            k = s[b] > conf
            bb, ss = xywh2xyxy(box[b][k]), s[b][k]
            if len(ss):
                j = torchvision.ops.nms(bb, ss, iou)
                bb, ss = bb[j], ss[j]
            res.append(torch.cat([bb, ss[:, None]], 1).numpy())
    return res


def latency_ms(m, reps=20):
    x = torch.rand(1, 3, IMG, IMG)
    with torch.no_grad():
        for _ in range(5):
            m(x)
        t = time.perf_counter()
        for _ in range(reps):
            m(x)
    return 1000 * (time.perf_counter() - t) / reps


# ------------------------------------------------------------------ craft
def _v5_centres():
    out = []
    for st in (8, 16, 32):
        n = IMG // st
        gy, gx = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
        c = torch.stack([(gx + 0.5) * st, (gy + 0.5) * st], -1).reshape(-1, 2).float()
        out.append(c.repeat(3, 1))               # 3 anchors per cell, order (anchor, gy, gx)
    return torch.cat(out)                         # [6300, 2]


V5C = _v5_centres()
PH_EVERY = 3
PL_W = 4.0                          # weight on suppressing the real player relative to growing phantoms


def phantom_mask5(persons):
    """Phantom anchors: stride-8 level, anchor 1 (16x30, person-shaped), every PH_EVERY-th cell, >= 12 px from
    every player. Player anchors (all levels): centre inside a player box (+4 px) -> suppressed."""
    n = IMG // 8
    gy, gx = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    keep = ((gy % PH_EVERY == 1) & (gx % PH_EVERY == 1)).flatten()
    ph = torch.zeros(6300, dtype=torch.bool)
    ph[n * n: 2 * n * n] = keep
    pl = torch.zeros(6300, dtype=torch.bool)
    for x1, y1, x2, y2 in persons:
        near = (V5C[:, 0] > x1 - 12) & (V5C[:, 0] < x2 + 12) & (V5C[:, 1] > y1 - 12) & (V5C[:, 1] < y2 + 12)
        ph &= ~near
        pl |= (V5C[:, 0] > x1 - 4) & (V5C[:, 0] < x2 + 4) & (V5C[:, 1] > y1 - 4) & (V5C[:, 1] < y2 + 4)
    return ph, pl


PH_WH = torch.tensor([14.0, 28.0])  # spaced 24 px apart and 14 px wide: phantoms never merge in NMS


def cloak_loss(box, s, masks):
    s = s.clamp(1e-6, 1 - 1e-6)
    if OBJ == "hide":
        return -torch.log(1 - s).sum() / len(s)
    total = 0.0
    for b, (ph, pl) in enumerate(masks):
        sp_, sl = s[b][ph], s[b][pl]
        # hinges: stop pushing phantoms once confident, and player anchors once well below threshold
        total = total - (torch.log(sp_) * (sp_ < 0.8)).sum() - PL_W * (torch.log(1 - sl) * (sl > 0.05)).sum() \
            + ((box[b][ph][:, 2:] - PH_WH).abs() / PH_WH).sum() + ((box[b][ph][:, :2] - V5C[ph]).abs() / 8).sum()
    return total / len(s)


def grad_cos5(m, xa, masks):
    """cos( d l_pos / d theta , d l_neg / d theta ) over all weights of m (double backprop).
    l_pos: attacker's loss for detecting the real player; l_neg: attacker's loss for scoring phantoms as background.
    cos -> -1 means any weight update that lowers one raises the other (to first order)."""
    params = [q for q in m.parameters() if q.requires_grad]
    _, s = person_conf(m, xa)
    s = s.clamp(1e-6, 1 - 1e-6)
    lp = torch.stack([-torch.log(s[b][pl]).mean() for b, (ph, pl) in enumerate(masks) if pl.any()]).mean()
    ln = torch.stack([-torch.log(1 - s[b][ph]).mean() for b, (ph, pl) in enumerate(masks) if ph.any()]).mean()
    gp = torch.cat([g.flatten() for g in torch.autograd.grad(lp, params, create_graph=True)])
    gn = torch.cat([g.flatten() for g in torch.autograd.grad(ln, params, create_graph=True)])
    return torch.nn.functional.cosine_similarity(gp, gn, 0)


def craft_hide(models, x, seed, persons=None, bs=16, gc_models=None, log=None):
    out = []
    rng = np.random.default_rng(seed)
    for i in range(0, len(x), bs):
        xb = x[i:i + bs]
        masks = [phantom_mask5(p) for p in persons[i:i + bs]] if OBJ != "hide" else None
        g = torch.Generator().manual_seed(seed + i)
        d = (torch.rand(xb.shape, generator=g) * 2 - 1) * EPS
        a = 2.5 * EPS / STEPS
        for _ in range(STEPS):
            d.requires_grad_(True)
            xa = (xb + d).clamp(0, 1)
            loss = 0.0
            for m in models:
                box, s = person_conf(m, eot_transform(xa))
                loss = loss + cloak_loss(box, s, masks)
            if gc_models:  # one random checkpoint of the simulated retraining trajectory per step (unbiased)
                cos = grad_cos5(gc_models[rng.integers(len(gc_models))], xa, masks)
                loss = loss + GAMMA * cos
            gr, = torch.autograd.grad(loss, d)
            d = (d.detach() - a * gr.sign()).clamp(-EPS, EPS)
            d = (xb + d).clamp(0, 1) - xb
        out.append(((xb + d).clamp(0, 1) * 255).round().to(torch.uint8))
        msg = f"    {i + len(xb)}/{len(x)}"
        if gc_models:
            xq = out[-1].float() / 255
            c = [float(grad_cos5(m, xq, masks).detach()) for m in gc_models]
            msg += " cos per checkpoint " + " ".join(f"{v:+.2f}" for v in c)
            if log is not None:
                log.append(c)
        print(msg, flush=True)
    return torch.cat(out)


def bank_file(rnd, split):
    return os.path.join(OUT, "banks", f"r{rnd}_{split}.pt")


def stage_craft(rnd):
    sp, _ = load_frames()
    os.makedirs(os.path.join(OUT, "banks"), exist_ok=True)
    proxies = [load5(os.path.join(W, f"yolov5{s}.pt")) for s in "ns"]
    gc_models = None
    if rnd == 2 and OBJ == "dilemma":
        # identical proxies to the flood round 2 (ablation: only the gradient-conflict term differs)
        proxies += [load5(os.path.join(FLOOD_OUT, "train", f"r1_defender_{s}", "weights", "last.pt")) for s in "ns"]
        traj = os.path.join(OUT, "train", "traj_defender_n", "weights")
        gc_models = [load5(os.path.join(W, "yolov5n.pt"), fuse=False)] + \
                    [load5(os.path.join(traj, f"epoch{e}.pt"), fuse=False) for e in (0, 2, 4)]
        for m in gc_models:
            for q in m.parameters():
                q.requires_grad_(True)
    elif rnd == 2:
        proxies += [load5(model_file("defender", s, 1)) for s in "ns"]
    todo = [("T", K_TEST), ("B", 1)] + ([("A", 1)] if rnd == 1 else [])
    for split, K in todo:
        p = bank_file(rnd, split)
        if os.path.exists(p):
            continue
        t = time.time()
        print(f"[craft r{rnd}] {split} with {len(proxies)} proxies", flush=True)
        x = torch.stack([f.x for f in sp[split]])
        cos_log = []
        bank = torch.stack([craft_hide(proxies, x, seed=1000 * rnd + 100 * k + (split == "T"),
                                         persons=[f.persons for f in sp[split]], gc_models=gc_models, log=cos_log)
                            for k in range(K)], 1)
        torch.save({f.name: bank[i] for i, f in enumerate(sp[split])}, p)
        if cos_log:
            json.dump(cos_log, open(os.path.join(OUT, f"gc_cos_r{rnd}_{split}.json"), "w"))
        print(f"  done in {time.time() - t:.0f}s", flush=True)


# ------------------------------------------------------------------ train
HYP = dict(lr0=0.001, lrf=0.1, momentum=0.937, weight_decay=0.0005, warmup_epochs=1.0, warmup_momentum=0.8,
           warmup_bias_lr=0.01, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0,
           fl_gamma=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0, scale=0.0, shear=0.0,
           perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=0.0, mixup=0.0, copy_paste=0.0)


def model_file(who, size, rnd):
    return os.path.join(OUT, "train", f"r{rnd}_{who}_{size}", "weights", "last.pt")


def stage_train(rnd):
    import yaml as pyyaml
    from .run import write_dataset
    sp, others = load_frames()
    hyp = os.path.join(OUT, "hyp.yaml")
    os.makedirs(OUT, exist_ok=True)
    pyyaml.safe_dump(HYP, open(hyp, "w"))
    jobs = [("attacker", s, "B") for s in "nsm"] + ([("defender", s, "A") for s in "ns"] if rnd == 1 else [])
    for who, size, split in jobs:
        if os.path.exists(model_file(who, size, rnd)):
            continue
        root = os.path.join(OUT, "ds", f"r{rnd}_{who}")
        if not os.path.exists(os.path.join(root, "data.yaml")):
            write_dataset(root, sp[split], torch.load(bank_file(rnd, split)), others[:80])
            shutil.rmtree(os.path.join(root, "labels", "train.cache"), ignore_errors=True)
        t = time.time()
        print(f"[train r{rnd}] {who} v5{size}", flush=True)
        cmd = [sys.executable, os.path.join(Y5, "train.py"), "--img", str(IMG), "--batch", "16", "--epochs", str(EPOCHS),
               "--data", os.path.join(root, "data.yaml"), "--weights", os.path.join(W, f"yolov5{size}.pt"),
               "--hyp", hyp, "--optimizer", "AdamW", "--device", "cpu", "--workers", "2", "--noval", "--noplots",
               "--project", os.path.join(OUT, "train"), "--name", f"r{rnd}_{who}_{size}", "--exist-ok", "--seed", "7"]
        r = subprocess.run(cmd, cwd=Y5, capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-3000:], r.stderr[-3000:])
            raise SystemExit(f"training failed: {who} {size}")
        print(f"  done in {time.time() - t:.0f}s", flush=True)


# ------------------------------------------------------------------- eval
def stage_eval():
    sp, _ = load_frames()
    test = sp["T"]
    clean = {f.name: (f.x[None] * 255).round().to(torch.uint8) for f in test}
    banks = {"none": clean}
    for rnd in (1, 2):
        if os.path.exists(bank_file(rnd, "T")):
            banks[f"cloak r{rnd}"] = torch.load(bank_file(rnd, "T"))
    attackers = {f"v5{s}": os.path.join(W, f"yolov5{s}.pt") for s in "nsm"}
    for rnd in (1, 2):
        for s in "nsm":
            if os.path.exists(model_file("attacker", s, rnd)):
                attackers[f"v5{s}-AT{rnd}"] = model_file("attacker", s, rnd)
    rows = []
    for an, ap in attackers.items():
        m = load5(ap)
        lat = latency_ms(m)
        for bn, bank in banks.items():
            at_rnd = an.split("-AT")[1] if "-AT" in an else None
            if at_rnd and bn != "none" and bn != f"cloak r{at_rnd}":
                continue  # an AT model is evaluated on the cloak round it was trained against (+ clean)
            x = torch.cat([bank[f.name] for f in test]).float() / 255
            dets = detect5(m, x)
            per, ttks, k = [], [], 0
            for i, f in enumerate(test):
                kk = len(bank[f.name])
                r = simulate(dets[k:k + kk], f.persons, thr=0.4, gain=GAIN, starts=12, seed=i, return_ttks=True)
                k += kk
                ttks += r.pop("ttks")
                per.append(r)
            ttks = np.asarray(ttks)
            n_dets = float(np.mean([(d[:, 4] > 0.4).sum() for d in dets]))  # boxes the aimbot sees per frame
            row = dict(attacker=an, frames=bn, latency_ms=lat, n_dets=n_dets, det=float(np.mean([r["det"] for r in per])),
                       fake=float(np.mean([r["fake"] for r in per])), steps_median=float(np.median(ttks)),
                       hit_steps10=float((ttks <= 10).mean()), hit_steps60=float((ttks <= 60).mean()), ttks=ttks.tolist())
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "ttks"}), flush=True)
    json.dump(rows, open(os.path.join(OUT, "results.json"), "w"))


if __name__ == "__main__":
    {"craft1": lambda: stage_craft(1), "craft2": lambda: stage_craft(2), "train1": lambda: stage_train(1),
     "train2": lambda: stage_train(2), "eval": stage_eval}[sys.argv[1]]()
