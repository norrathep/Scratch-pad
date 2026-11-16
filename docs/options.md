# Beyond Invisibility Cloak: slowing visual aimbots down, under adaptive attackers

Working notes. Scope: proactive, perturbation-based defences against visual (screen-capture + detector) aimbots, building on
*Invisibility Cloak* (IC; Sun et al., USENIX Security 2024) and the follow-up *AimTrap* (Wang et al., CCS 2026).

---

## 1. What IC does

* **Universal Cloak.** One static l∞ perturbation (ε = 8/255), trained offline on an ensemble of proxy detectors (YOLOv5 family, NanoDet), with BCE pushing **every** box confidence to 0 (Eq. 3).
* **Refinement.** On some frames, up to 5 Adam-like steps against one lightweight proxy, *starting from* the universal cloak. Frames are refined at intervals; the remaining frames just get the universal cloak.
* **Deployment.** Inside the render pipeline: server-side for cloud gaming, TEE for local clients.
* **Adaptive evaluation (§5.6).** (i) Gaussian noise before/after the cloak: < 5 % DSR drop. (ii) Fine-tune YOLOv5n, then adversarially train it on 20 % of the protected data: DSR falls to **81.7 %** on CS2 and stays at 100 % on CF.
* **Self-declared limitation (§6).** "If the cheater collects sufficient adversarial samples and adopts more targeted fine-tuning strategies, our defense may become vulnerable … we can design *inducive perturbations* (such as identifying additional wrong targets), which we leave as future work."

AimTrap does the "inducive" part with 3D adversarial textures: camouflage textures on players plus honeypot textures in the map. It makes the **threshold-dilemma** argument: lowering the threshold to recover cloaked players also triggers more honeypot locks. Its adaptive-attack section is qualitative. It has no adversarial-training experiment.

## 2. Where IC is weak

| # | Weakness | Why it matters |
|---|---|---|
| W1 | **The attacker moves second.** Every protected frame is visible to the cheater, who can harvest it, label it and fine-tune on it. This is the Fawkes/LowKey situation described in Radiya-Dixit & Tramèr, *Data poisoning won't save you from facial recognition*. | Any *fixed* perturbation family will eventually be trained away. The defender's real levers are **cost** (data, labels, GPU time) and **perishability** (rotation), not "unbreakable noise". |
| W2 | **The per-frame DSR metric overstates protection.** An aimbot only needs a few good frames within ~100–250 ms. | If frames are independent, 81.7 % per-frame DSR means P(≥1 detection in 15 frames) = 1 − 0.817¹⁵ ≈ **95 %**. Under the paper's own AT result the aimbot still mostly works. The metric that matters is closed-loop **time-to-hit**. |
| W3 | **The "hide" objective is the easiest one for an adaptive attacker to undo.** It only pushes confidence down. Lowering the threshold, or adversarial training that restores recall, attacks it directly. | IC fixes θ = 0.4 and never sweeps the attacker's threshold. |
| W4 | **The deployed signal is mostly one static pattern.** The universal cloak is on every frame, and refinements only take a few steps away from it. | A cheater can estimate the pattern by averaging high-pass residuals over harvested frames and subtract it, with no clean frames needed (we test this in §5.3). |
| W5 | **The adaptive attacks evaluated are weak.** Only Gaussian noise and one small AT run. No JPEG, resize, blur or diffusion purification (cf. noisy-upscaling attacks on Glaze-style protections), no EOT/BPDA, no temporal filtering, no change of architecture. | Reviewers of a follow-up will ask for all of these. |

## 3. Reframing the goal

A cheater wants **superhuman reaction time and precision**. Human visual reaction is about 200–250 ms (≈ 12–15 frames at 60 FPS). An unprotected aimbot gets there in a few frames.

> **Defence goal:** push the aimbot's *time-to-hit* distribution back to human level, or make its locks unreliable, and turn the cheater's counter-measures into **latency**.

This loosens the requirement. The cloak does not have to be unbreakable. It has to make every way of working around it cost time. Temporal filtering, multi-frame confirmation, ensembles, purification and bigger robust models all add delay, and delay is exactly what removes the aimbot's advantage. Call this a **latency tax**. False locks on phantoms are also detection evidence, so prevention and detection reinforce each other.

## 4. Option space

Ranked by how promising each looks. Status: **[P]** = prototyped in `aimbot_defense/`, **[I]** = idea only.

### O1. Per-frame misdirection, "decoy" / "drift" [P] (recommended starting point)
Keep the IC machinery (l∞ ε = 8/255, per-frame PGD), but change the objective from *hide* to *misdirect*:
* **decoy:** suppress the real player **and** synthesise a confident phantom player 1.0–1.6 body-sizes away;
* **drift:** keep a detection on the player but move its centre (the aim point) off the hitbox.

The direction is **re-sampled every frame**. A closed-loop aimbot (closest box to the crosshair, smoothed mouse motion) gets dragged somewhere different each frame, so it oscillates instead of converging. That is the slowdown.

Why it may resist adaptive attacks better than *hide*:
* Lowering the threshold does not help, because phantoms are high-confidence (AimTrap's dilemma, here in screen space).
* Temporal consistency filters (track confirmation over n frames) can reject frame-varying phantoms, but they add n frames of latency: the latency tax.
* Every lock on a phantom is a behavioural signal: the cheater snaps onto empty space where no human would aim.

Open risk: adversarial training on decoy frames may teach the model to ignore phantoms. That is the main thing to measure.

### O2. Compute slowdown via sponge/NMS overload (Phantom Sponges, Overload) [I] (not recommended as a primary)
Flooding NMS with candidate boxes does inflate latency, and it is the literal reading of "slow down". But the adaptive counter is trivial: cap pre-NMS top-k / `max_det` (an aimbot needs only a handful of boxes), or switch to an NMS-free detector (YOLOv10, RT-DETR). Defences already exist (Underload, CVPR 2024). It is fine as a by-product of O1 (decoys add boxes), but not as the core mechanism.

### O3. Fine-tuning-aware cloaks: iterated best response or bi-level optimisation [P, cheap version]
Simulate the attacker inside the defender's loop:
* *Cheap version (prototyped as "r2"):* the defender fine-tunes its own proxy on its own protected frames, then crafts against the ensemble {original, defender-AT proxy}. This is one round of iterated best response. It can be extended to a population of AT'd proxies (PSRO-style).
* *Bi-level version:* optimise the universal component / generator G so that it still works after k attacker fine-tuning steps on G's outputs (unrolled or MAML-style), as in robust unlearnable examples (REM, ICLR 2022) and anti-fine-tuning work (e.g. non-fine-tunable learning).

### O4. Moving target: keyed perturbation families plus rotation [I]
Per-session or per-patch secret seeds into a perturbation *generator*, drawn from deliberately diverse families: different frequency bands, spatial structure, objectives (hide/decoy/drift) and colour channels. The research question is the **cross-family AT transfer matrix**: if AT on family F1 does not transfer to F2, harvested data is perishable, and the attacker's retraining cadence must beat the rotation period. Measure the attacker's data-efficiency curve (frames needed to recover hit@15) against rotation period.

### O5. Poisoning the harvest: a shortcut trap [I] (speculative)
Embed a key-dependent, easily learnable pattern correlated with player boxes. An attacker who fine-tunes on harvested frames latches onto the shortcut (as with unlearnable examples or backdoors). After rotating the key, the fine-tuned model is *worse* than the base model. Risk: while a key is live, the pattern is a positive cue that helps the attacker. Only worth it combined with fast rotation (O4).

### O6. Purification-robust budgets [P: EOT]
EOT over noise, JPEG, resize and blur during crafting (prototyped). Beyond l∞: perceptual budgets (JND or contrast masking, so larger ε in textured regions and smaller in flat sky), low-frequency or colour/shading perturbations, or perturbations in render space (shaders, textures, lighting, as in AimTrap). These survive purifiers because they are *content*. They are still open to AT, so they pair with O3/O4.

### O7. Temporal complementarity at high refresh rates [I] (speculative, needs a safety study)
Alternate δ and −δ, or more generally zero-sum sequences, at 144–240 Hz. Human vision integrates them away, which allows a much larger per-frame ε. The attacker must average ≥ 2 frames, paying latency and motion blur, or capture at a lower rate. Risks are perceptibility (phantom-array effect, saccades) and photosensitivity, so it needs careful human studies.

### O8. Controller-aware decoy placement [I] (cheap extension of O1)
The defender knows the crosshair is at the screen centre, so it can place decoys to **win target selection**: closer to the crosshair than the real player. It can also choose per-frame offsets that make a smoothed controller (EMA/PID) oscillate, for example alternating directions at a period matched to typical smoothing gains.

## 5. Pilot experiment (CPU, toy scale)

**Setup.** COCO128 person images as a stand-in for game frames: 48 scenes with ≥ 1 sizeable person, split into A (defender private) / B (attacker harvest) / T (test), 16 each. Frames are letterboxed to 320 × 320. Defender proxy: YOLOv8n. Attackers: YOLOv8n (same weights, so white-box transfer), YOLOv8s (different size), and YOLOv8n adversarially fine-tuned (12 epochs) on B's protected frames with true labels. ε = 8/255, quantised to 8 bit. Each test scene gets a bank of 4 independently crafted variants, and a fresh one is drawn every simulated frame.

**Closed-loop simulator** (`aimsim.py`). The crosshair starts 50–110 px from a player. Each frame: detect, pick the box closest to the crosshair, move 50 % of the way to its centre, and count a hit when the crosshair is inside the player's central hitbox. Up to 60 frames (1 s at 60 FPS); 12 starts per scene.

**Defences:** `ic_hide` (IC-style objective, 10 PGD steps), `hide_eot`, `decoy_eot`, `drift_eot` (30 steps with EOT), and `+r2` variants (O3, crafted against {YOLOv8n, defender's AT'd YOLOv8n}).

**Adaptive attacks:** preprocessing (Gaussian σ = 8/255, JPEG-75, 0.5× down-up, blur σ = 1), threshold lowering (0.4 → 0.15), adversarial fine-tuning, and universal-cloak subtraction (§5.3).

RESULTS_PLACEHOLDER

## 7. Suggested evaluation protocol for the real study

* **Attacker knowledge:** K0 unaware; K1 knows the algorithm but not the keys or proxies; K2 white-box on the deployed generator (extracted from the client).
* **Attacks:** A1 purification (noise, JPEG, resize, blur, median, diffusion/upscaler); A2 threshold sweep; A3 temporal filtering (n-frame confirmation, EMA, Kalman); A4 AT with harvested frames ∈ {100, 1k, 10k} and label sources {manual, pseudo-labels from purified frames}; A5 universal-pattern estimation and subtraction; A6 architecture swap (NMS-free, bigger, transformer); A7 detect-and-route (a classifier flags protected frames and routes them to a robust model).
* **Metrics:** time-to-hit distribution and hit@150/250 ms (primary); false-lock rate (evidence); per-frame DSR (for comparability with IC); attacker cost (frames, GPU-hours, *added inference latency*); defender cost (ms/frame, LPIPS/SSIM, user study).
* **Data:** real game captures (CS2 / CF, or open-source shooters) with ground-truth boxes from the engine. Run real aimbot code (the open-source YOLO-based ones IC uses) in a closed loop on recorded demos.

## References (as used here)
* C. Sun et al. *Invisibility Cloak: Proactive Defense Against Visual Game Cheating.* USENIX Security 2024. https://www.usenix.org/conference/usenixsecurity24/presentation/sun-chenxin
* J. Wang et al. *Shoot the Honey, Cloak the Player (AimTrap).* CCS 2026, arXiv:2606.25734.
* E. Radiya-Dixit, S. Hong, N. Carlini, F. Tramèr. *Data Poisoning Won't Save You From Facial Recognition.* ICLR 2022, arXiv:2106.14851.
* A. Shapira et al. *Phantom Sponges: Exploiting Non-Maximum Suppression to Attack Deep Object Detectors.* WACV 2023, arXiv:2205.13618.
* E.-C. Chen et al. *Overload: Latency Attacks on Object Detection for Edge Devices.* CVPR 2024.
* *Underload: Defending against Latency Attacks for Object Detectors on Edge Devices.* arXiv:2412.02171.
* S. Fu et al. *Robust Unlearnable Examples (REM).* ICLR 2022.
* R. Hönig et al. *Adversarial Perturbations Cannot Reliably Protect Artists From Generative AI.* ICLR 2025.
