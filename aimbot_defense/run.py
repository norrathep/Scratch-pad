"""Experiment pipeline (CPU-friendly, cached per stage).

  python -m aimbot_defense.run craft            # round-1 cloaks (proxy = YOLOv8n)
  python -m aimbot_defense.run train            # adversarial fine-tuning (attacker + defender-side)
  python -m aimbot_defense.run craft2           # round-2 cloaks vs {YOLOv8n, defender's own AT model}
  python -m aimbot_defense.run eval             # closed-loop aimbot metrics -> out/results.md

Splits of the person images: A = defender's private set (used to train an
anticipated-attacker proxy), B = attacker's harvested frames (used for
adversarial fine-tuning), T = held-out test scenes.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import zlib

import cv2
import numpy as np
import torch

from .aimsim import aggregate, simulate
from .core import IMG, PREPROC, ROOT, detect_persons, load_coco128, load_model
from .craft import craft, sample_targets

OUT = os.path.join(ROOT, "out")
W = os.path.join(ROOT, "weights")
CONFIGS = {
    "ic_hide":   dict(kind="hide",  eot=False, steps=10),
    "hide_eot":  dict(kind="hide",  eot=True,  steps=30),
    "decoy_eot": dict(kind="decoy", eot=True,  steps=30),
    "drift_eot": dict(kind="drift", eot=True,  steps=30),
}
K_TEST, K_TRAIN, EPS = 4, 3, 8 / 255
torch.set_num_threads(4)


def splits():
    persons, others = load_coco128()
    rng = random.Random(0)
    rng.shuffle(persons)
    n = len(persons) // 3
    return dict(A=persons[:n], B=persons[n:2 * n], T=persons[2 * n:]), others


def make_bank(models, frames, cfg, K, seed, bs=16):
    """Returns {name: uint8 [K,3,H,W]}."""
    jobs = [(f, k) for f in frames for k in range(K)]
    out = {f.name: [None] * K for f in frames}
    for i in range(0, len(jobs), bs):
        chunk = jobs[i:i + bs]
        x = torch.stack([f.x for f, _ in chunk])
        fakes = [sample_targets(f.persons, cfg["kind"], random.Random(zlib.crc32(f'{seed}/{f.name}/{k}'.encode())))
                 for f, k in chunk]
        xa = craft(models, x, fakes, EPS, cfg["steps"], cfg["eot"], seed=seed + i)
        for (f, k), xi in zip(chunk, xa):
            out[f.name][k] = (xi * 255).round().to(torch.uint8)
        print(f"  {i + len(chunk)}/{len(jobs)}", flush=True)
    return {n: torch.stack(v) for n, v in out.items()}


def bank_path(cfg, split, rnd=1):
    return os.path.join(OUT, "banks", f"r{rnd}_{cfg}_{split}.pt")


def stage_craft(rnd=1, cfgs=None):
    sp, _ = splits()
    os.makedirs(os.path.join(OUT, "banks"), exist_ok=True)
    for name in cfgs or CONFIGS:
        cfg = CONFIGS[name]
        if rnd == 1:
            models = [load_model(os.path.join(W, "yolov8n.pt"))]
            todo = [("T", K_TEST), ("A", K_TRAIN), ("B", K_TRAIN)]
        else:
            models = [load_model(os.path.join(W, "yolov8n.pt")),
                      load_model(model_path(name, "defender"))]
            todo = [("T", K_TEST)]
        for split, K in todo:
            p = bank_path(name, split, rnd)
            if os.path.exists(p):
                continue
            t = time.time()
            print(f"[craft r{rnd}] {name} {split}", flush=True)
            torch.save(make_bank(models, sp[split], cfg, K, seed=1000 * rnd + (split == "T")), p)
            print(f"  done in {time.time() - t:.0f}s", flush=True)


def model_path(cfg, who):
    return os.path.join(OUT, "train", f"{cfg}_{who}", "weights", "last.pt")


def write_dataset(root, frames, bank, others):
    for sub in ("images", "labels"):
        os.makedirs(os.path.join(root, sub, "train"), exist_ok=True)

    def put(name, img_u8, labels):
        cv2.imwrite(os.path.join(root, "images", "train", name + ".png"),
                    cv2.cvtColor(img_u8.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR))
        with open(os.path.join(root, "labels", "train", name + ".txt"), "w") as fh:
            for c, x1, y1, x2, y2 in labels:
                fh.write(f"{int(c)} {(x1 + x2) / 2 / IMG:.6f} {(y1 + y2) / 2 / IMG:.6f} "
                         f"{(x2 - x1) / IMG:.6f} {(y2 - y1) / IMG:.6f}\n")

    for f in frames:  # protected copies (true labels: the attacker labels real players) + clean copy
        for k, xi in enumerate(bank[f.name]):
            put(f"{f.name}_p{k}", xi, f.all_labels)
        put(f"{f.name}_clean", (f.x * 255).round().to(torch.uint8), f.all_labels)
    for f in others:
        put(f"{f.name}_o", (f.x * 255).round().to(torch.uint8), f.all_labels)
    yaml = os.path.join(root, "data.yaml")
    from ultralytics import YOLO
    from ultralytics.utils import YAML
    names = YOLO(os.path.join(W, "yolov8n.pt")).names
    YAML.save(yaml, dict(path=root, train="images/train", val="images/train", names=names))
    return yaml


def stage_train(cfgs=None, epochs=12):
    from ultralytics import YOLO
    sp, others = splits()
    for name in cfgs or CONFIGS:
        for who, split in (("attacker", "B"), ("defender", "A")):
            if os.path.exists(model_path(name, who)):
                continue
            root = os.path.join(OUT, "ds", f"{name}_{who}")
            yaml = write_dataset(root, sp[split], torch.load(bank_path(name, split)), others)
            t = time.time()
            print(f"[train] {name} {who}", flush=True)
            YOLO(os.path.join(W, "yolov8n.pt")).train(
                data=yaml, epochs=epochs, imgsz=IMG, batch=16, device="cpu", workers=2,
                optimizer="AdamW", lr0=0.001, warmup_epochs=1, mosaic=0.0, close_mosaic=0,
                hsv_h=0, hsv_s=0, hsv_v=0, scale=0.0, translate=0.0, fliplr=0.5,
                val=False, plots=False, amp=False, verbose=False, seed=zlib.crc32(who.encode()) & 0xff,
                project=os.path.join(OUT, "train"), name=f"{name}_{who}", exist_ok=True)
            print(f"  done in {time.time() - t:.0f}s", flush=True)


def stage_eval():
    sp, _ = splits()
    test = sp["T"]
    attackers = {"v8n": os.path.join(W, "yolov8n.pt"), "v8s": os.path.join(W, "yolov8s.pt")}
    for name in CONFIGS:
        if os.path.exists(model_path(name, "attacker")):
            attackers[f"v8n_AT[{name}]"] = model_path(name, "attacker")
    models = {k: load_model(v) for k, v in attackers.items()}
    banks = {"none": {f.name: (f.x[None] * 255).round().to(torch.uint8) for f in test}}
    for name in CONFIGS:
        for rnd in (1, 2):
            p = bank_path(name, "T", rnd)
            if os.path.exists(p):
                banks[f"{name}" + ("" if rnd == 1 else "+r2")] = torch.load(p)
    rows = []
    for bname, bank in banks.items():
        for aname, m in models.items():
            # an AT attacker only exists for the defence it was trained against
            if "AT[" in aname and aname.split("[")[1][:-1] != bname.replace("+r2", "") and bname != "none":
                continue
            for pre, fn in PREPROC.items():
                torch.manual_seed(0)
                dets = {}
                for f in test:
                    x = fn(bank[f.name].float() / 255)
                    dets[f.name] = detect_persons(m, x, conf=0.05)
                for thr in (0.4, 0.15):
                    per = [simulate(dets[f.name], f.persons, thr=thr, seed=i) for i, f in enumerate(test)]
                    r = dict(defence=bname, attacker=aname, preproc=pre, thr=thr, **aggregate(per))
                    rows.append(r)
                    print(json.dumps(r), flush=True)
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    st = sys.argv[1]
    cfgs = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    {"craft": lambda: stage_craft(1, cfgs), "craft2": lambda: stage_craft(2, cfgs),
     "train": lambda: stage_train(cfgs), "eval": stage_eval}[st]()
