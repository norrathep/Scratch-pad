# Aimbot defence exploration

Exploring defences against visual (screen-capture + detector) aimbots that are more robust to adaptive attackers than
Invisibility Cloak (USENIX Security 2024), with the goal of *slowing the aimbot down* (closed-loop time-to-hit)
rather than only blinding it per frame.

* `docs/options.md` – analysis of IC, option space, pilot results, evaluation protocol.
* `aimbot_defense/` – CPU testbed (YOLOv8 + COCO128 persons as a stand-in for game frames):
  * `craft.py` – PGD crafting with `hide` (IC objective), `decoy`, `drift` objectives (+EOT).
  * `aimsim.py` – closed-loop aimbot simulator (closest-box targeting, smoothed mouse, time-to-hit).
  * `run.py` – pipeline: `craft` → `train` (adversarial fine-tuning) → `craft2` (AT-aware round 2) → `eval`.
  * `universal.py` – IC universal cloak + cloak estimation/subtraction attack.

Setup:
```
python3 -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision && pip install ultralytics
mkdir -p data weights
curl -L -o data/coco128.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip && (cd data && unzip coco128.zip)
curl -L -o weights/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
curl -L -o weights/yolov8s.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt
python -m aimbot_defense.run craft && python -m aimbot_defense.run train && \
python -m aimbot_defense.run craft2 && python -m aimbot_defense.run eval
python -m aimbot_defense.universal
```
