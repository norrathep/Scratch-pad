"""Core utilities for the aimbot-defense testbed.

Everything here works on 320x320 letterboxed RGB frames in [0,1] (CHW float
tensors) and on YOLOv8 raw outputs of shape [B, 4+nc, N] (cx, cy, w, h in
pixels, then per-class sigmoid scores). Class 0 is "person", which stands in
for a player model in an FPS frame.
"""
from __future__ import annotations

import glob
import io
import math
import os
import random
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

IMG = 320
STRIDES = (8, 16, 32)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- data
@dataclass
class Frame:
    name: str
    x: torch.Tensor                       # [3, IMG, IMG] float in [0,1]
    persons: np.ndarray                   # [P, 4] xyxy pixels (letterboxed)
    all_labels: np.ndarray = field(default=None)  # [L, 5] cls, xyxy pixels


def letterbox(img: np.ndarray, boxes_xyxy: np.ndarray):
    h, w = img.shape[:2]
    r = IMG / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (IMG - nh) // 2, (IMG - nw) // 2
    out = np.full((IMG, IMG, 3), 114, np.uint8)
    out[top:top + nh, left:left + nw] = img
    b = boxes_xyxy.copy().astype(np.float32)
    if len(b):
        b[:, [0, 2]] = b[:, [0, 2]] * r + left
        b[:, [1, 3]] = b[:, [1, 3]] * r + top
    return out, b


def load_coco128(min_person_area: float = 0.01):
    """Returns (person_frames, other_frames). Person frames keep only images
    with at least one reasonably sized person (stand-in for an enemy player)."""
    base = os.path.join(ROOT, "data", "coco128")
    persons, others = [], []
    for ip in sorted(glob.glob(os.path.join(base, "images", "train2017", "*.jpg"))):
        name = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(base, "labels", "train2017", name + ".txt")
        img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        lab = np.zeros((0, 5), np.float32)
        if os.path.exists(lp):
            arr = np.loadtxt(lp, ndmin=2).astype(np.float32)
            if arr.size:
                c, cx, cy, bw, bh = arr.T
                lab = np.stack([c, (cx - bw / 2) * w, (cy - bh / 2) * h,
                                (cx + bw / 2) * w, (cy + bh / 2) * h], 1)
        lb_img, lb_boxes = letterbox(img, lab[:, 1:])
        lab_lb = np.concatenate([lab[:, :1], lb_boxes], 1) if len(lab) else lab
        x = torch.from_numpy(lb_img).permute(2, 0, 1).float() / 255.0
        p = lab_lb[lab_lb[:, 0] == 0, 1:] if len(lab_lb) else np.zeros((0, 4), np.float32)
        if len(p):
            area = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1]) / IMG ** 2
            p = p[area >= min_person_area]
        f = Frame(name, x, p, lab_lb)
        (persons if len(p) else others).append(f)
    return persons, others


# -------------------------------------------------------------------------- model
def load_model(path: str):
    from ultralytics import YOLO
    m = YOLO(path).model.float().eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def raw(model, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    return out[0] if isinstance(out, (list, tuple)) else out


def anchor_centers(device="cpu") -> torch.Tensor:
    pts = []
    for s in STRIDES:
        n = IMG // s
        ys, xs = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
        pts.append(torch.stack([(xs + 0.5) * s, (ys + 0.5) * s], -1).reshape(-1, 2))
    return torch.cat(pts).float().to(device)  # [N, 2]


ANCH = anchor_centers()
ANCH_STRIDE = torch.cat([torch.full(((IMG // s) ** 2,), float(s)) for s in STRIDES])


def xywh2xyxy(b: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


@torch.no_grad()
def detect_persons(model, x: torch.Tensor, conf: float = 0.4, iou: float = 0.45):
    """x: [B,3,H,W]. Returns list of [K,5] arrays (xyxy, conf) for class person."""
    p = raw(model, x)
    res = []
    for b in range(p.shape[0]):
        boxes = xywh2xyxy(p[b, :4].T)
        sc = p[b, 4]
        keep = sc > conf
        boxes, sc = boxes[keep], sc[keep]
        if len(sc):
            k = torchvision.ops.nms(boxes, sc, iou)
            boxes, sc = boxes[k], sc[k]
        res.append(torch.cat([boxes, sc[:, None]], 1).numpy())
    return res


# ----------------------------------------------------------------- preprocessing
def jpeg(x: torch.Tensor, q: int = 75) -> torch.Tensor:
    out = []
    for xi in x:
        im = Image.fromarray((xi.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        out.append(torch.from_numpy(np.array(Image.open(buf))).permute(2, 0, 1).float() / 255)
    return torch.stack(out)


def down_up(x: torch.Tensor, f: float = 0.5) -> torch.Tensor:
    h = x.shape[-1]
    s = max(8, int(h * f))
    return F.interpolate(F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False),
                         size=(h, h), mode="bilinear", align_corners=False)


def gauss_blur(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    k = int(2 * math.ceil(2 * sigma) + 1)
    t = torch.arange(k).float() - k // 2
    g = torch.exp(-t ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    w1 = g.view(1, 1, 1, k).repeat(3, 1, 1, 1)
    w2 = g.view(1, 1, k, 1).repeat(3, 1, 1, 1)
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), w1, groups=3)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), w2, groups=3)


PREPROC = {
    "none": lambda x: x,
    "gauss8": lambda x: (x + torch.randn_like(x) * 8 / 255).clamp(0, 1),
    "jpeg75": lambda x: jpeg(x, 75),
    "down0.5": lambda x: down_up(x, 0.5),
    "blur1": lambda x: gauss_blur(x, 1.0),
}


def eot_transform(x: torch.Tensor) -> torch.Tensor:
    """Differentiable random purification used during crafting (EOT)."""
    r = random.random()
    if r < 0.25:
        x = x + torch.randn_like(x) * random.uniform(0, 10) / 255
    elif r < 0.5:
        x = down_up(x, random.uniform(0.45, 0.9))
    elif r < 0.75:
        x = gauss_blur(x, random.uniform(0.5, 1.3))
    return x.clamp(0, 1)
