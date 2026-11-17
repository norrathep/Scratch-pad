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

### 1.1 What IC and AimTrap evaluate on

| | Invisibility Cloak (USENIX Sec '24) | AimTrap (CCS '26) |
|---|---|---|
| Aimbot | RootKit-Org *AI-Aimbot* (open source, YOLOv5-based) | the same RootKit AI-Aimbot, defaults: 640×640 aim window, conf 0.4, `aaMovementAmp` = 0.4 (a proportional controller moving 40 % of the offset per step), headshot mode |
| Cheating models | YOLOv5 n/s/m/l/x and n6–x6, NanoDet-Plus-m (anchor-free); transfer tests on YOLOv3n, YOLOv8n, Faster R-CNN, RTMDet | proxy YOLOv5n, main attacker **YOLOv5s**; transfer to YOLOv5 n/s/m/l, YOLOv8n, Faster R-CNN, NanoDet |
| Proxies | YOLOv5n / NanoDet (refinement); ensembles of YOLOv5 models (universal cloak) | YOLOv5n, or an ensemble of YOLOv5n+s with MI+TI |
| Data | ACVC-CS2 (14,820 frames, 9 maps) and ACVC-CF (57,330 frames, 7 maps): real match footage cropped to **320×320** | a custom CS2 map (Hammer editor) with 16 honeypot textures and 4 generated player models; real-game runs on CS2 |
| Metric | per-frame Defence Success Rate (no box ≥ 0.4), FPS, #queries, SSIM, user study (122 people) | per-frame decoy / evasion success, uniqueness, trajectory-classifier detection accuracy |
| Hardware | Xeon 5418Y + RTX 4090 | i7-13700K + RTX 4090 / 5090 |
| Adaptive | random noise; fine-tune + AT YOLOv5n on 20 % of the data | threshold argument; qualitative discussion of retraining |

Our capacity experiment (§5.7) therefore uses the original anchor-based YOLOv5 n/s/m (ultralytics v7.0 weights), 320×320 frames, conf 0.4 and a controller gain of 0.4.

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


All numbers are closed-loop **hit@15**: the fraction of engagements where the aimbot reaches a real hitbox within 250 ms. Lower is better for the defender. The test set is 16 scenes × 12 starts. Raw data: `out/results.json`, `out/universal_results.json`.

### 5.1 White-box attacker (YOLOv8n): purification and threshold lowering (thr = 0.4 unless noted)

| defence | no preproc | thr 0.15 | Gauss σ=8/255 | JPEG-75 | down-up 0.5× | blur σ=1 |
|---|---|---|---|---|---|---|
| none | 0.86 | 0.92 | 0.86 | 0.86 | 0.81 | 0.87 |
| `ic_hide` (IC objective, no EOT) | **0.09** | 0.10 | 0.16 | 0.37 | **0.62** | **0.56** |
| `hide_eot` | **0.00** | 0.06 | 0.06 | 0.06 | **0.06** | **0.06** |
| `decoy_eot` | 0.51 | 0.55 | 0.67 | 0.77 | 0.60 | 0.56 |
| `drift_eot` | 0.85 | 0.81 | 0.88 | 0.92 | 0.86 | 0.87 |

### 5.2 Adversarially fine-tuned attacker, YOLOv8n-AT trained on harvested frames of the *same* defence (thr = 0.4)

| defence | AT model on clean frames | round-1 cloak | round-2 cloak (+r2, O3) |
|---|---|---|---|
| `ic_hide` | 0.77 | 0.68 | 0.52 |
| `hide_eot` | 0.81 | 0.50 | 0.50 |
| `decoy_eot` | 0.64 | 0.67 | 0.74 |
| `drift_eot` | 0.48 | 0.62 | 0.79 |

Transfer to an unseen YOLOv8s: **every** defence scores 0.81–0.94, i.e. no protection.

### 5.3 Universal-cloak estimation and subtraction (§2 W4)
* From 112 harvested protected frames, the high-pass residual average recovers the universal cloak with **cosine 0.96** to its true high-frequency component. With IC-style 5-step refinement on top, it is still **0.92**. No clean frames are needed.
* At this toy scale the universal cloak itself did not generalise: hit@15 is 0.86, the same as clean. The refined cloak (white-box) gave 0.06, and subtracting the estimate raised it to 0.19 (0.25 at thr 0.15). A universal cloak trained on large data, as IC's is, would carry more of the effect and so lose more to subtraction. This needs testing at scale.

### 5.4 What the pilot says (16 test scenes, so read as a direction, not a result)
1. **IC-style cloaks without EOT are fragile to free purification**, which IC did not test. A 0.5× down-up resize takes hit@15 from 0.09 to 0.62, and a σ = 1 blur takes it to 0.56. Crafting with EOT removes this at the same ε: 0.06 across all purifiers. Any follow-up should use EOT as the baseline, not IC as published.
2. **Adversarial fine-tuning largely undoes the plain IC objective.** The AT model scores 0.68 under the cloak against 0.77 on clean frames. The EOT version keeps more protection (0.50 against 0.81). This is the trend IC's own Table 10 hints at.
3. **Naive per-frame misdirection did *not* beat hide, and AT neutralises it** (decoy 0.67 against 0.64 clean). The key lesson is that **zero-mean, per-frame random offsets are averaged out by the aimbot's own mouse smoothing**: drift fails even white-box (0.85). For a "slowdown" defence, the misdirection has to be **temporally correlated**: a consistent bias per engagement, or a decoy placed to win target selection (O8). Independent per-frame noise does not work.
4. **Single-proxy cloaks do not transfer** to YOLOv8s at all. Proxy ensembles (as in IC) are mandatory, and every comparison here is white-box or near-white-box.
5. **One round of iterated best response (O3 "+r2") helped the plain IC objective under AT** (0.68 → 0.52) but did nothing for the others. It is cheap and worth scaling up, but the evidence is weak.
6. **Confounds:** the AT runs used very little data (16 scenes), which also degraded clean performance (0.48–0.81 against 0.86). Compare each AT row with its own clean column. COCO is not game footage. Small n.

### 5.5 Phantom flood against box capping (`flood.py`)
Question: if the cheater caps candidate boxes to defeat a sponge cloak, do they lose the player? The flood cloak puts confident 14×28 px phantoms on a lattice of stride-8 anchors outside the player (ε = 8/255, white-box YOLOv8n). The cap keeps the top-k boxes by score before NMS, the same order ultralytics uses.

| cap k | 1 | 3 | 10 | 30 | 100 | 300 | none |
|---|---|---|---|---|---|---|---|
| player detected (clean = 0.81–0.88) | 0.25 | 0.31 | 0.38 | 0.50 | 0.75 | 0.88 | 0.88 |
| hit@15 (clean = 0.83–0.86) | 0.41 | 0.43 | 0.33 | 0.22 | 0.17 | 0.15 | 0.14 |
| false locks / engagement | 33 | 32 | 38 | 44 | 48 | 49 | 50 |

* **Yes, capping costs the player.** With k ≤ 10, the player survives the cap in only 25–38 % of frames.
* **But the runtime lever is negligible.** On CPU, NMS over the ~340 candidates takes **0.3 ms against ~95 ms for the forward pass**. Hundreds of boxes do not slow a desktop aimbot. Overload-style slowdowns need tens of thousands of boxes on weak hardware.
* **What actually slows the aimbot is target selection.** Even uncapped, with the player still detected, hit@15 falls from 0.86 to **0.14**, because there is almost always a phantom closer to the crosshair. Dense phantoms work far better than one decoy (0.51). This is the useful form of "slow it down".
* **Adaptive attacks:**
  * *Retraining on flood frames largely removes it.* Candidates fall from 341 to 10, and hit@15 is 0.56 against 0.69 for the same retrained model on clean frames.
  * *It does not transfer to YOLOv8s* (0.88–0.91).

### 5.6 Dilemma loss: can retraining be forced to choose between "detect the player" and "drop the phantoms"? (`dilemma.py`)
**Idea.** The attacker's fine-tuning lowers l_pos (player anchors scored as person) and l_neg (phantom anchors scored as background, which also means fewer candidates, fewer false locks and less runtime) together. If ∇θ l_pos = −c·∇θ l_neg, then to first order any parameter update, with any optimiser, that improves one makes the other worse. So we add a gradient-cancellation term (as in gradient-matching poisoning, cf. Witches' Brew) to the flood cloak:

  L = L_flood(δ) + γ · cos(∇θ l_pos(x+δ), ∇θ l_neg(x+δ)) → minimise, θ = all YOLOv8n parameters (double backprop).

Note: measured on the last 1×1 class layer only, this cosine is saturated even on clean frames (−0.81), because it reduces to feature similarity. It has to be measured over the whole network. Whole-network cosine: −0.12 on clean frames, −0.35 to −0.40 for the plain flood, **−0.86** with the dilemma term.

| | white-box v8n, no cap | v8n, cap 10 | v8n-AT (12 epochs), its own clean | v8n-AT on cloaked frames | phantom candidates after AT |
|---|---|---|---|---|---|
| plain flood | 0.14 | 0.33 | 0.69 | 0.56 | 341 → 10 |
| dilemma flood | **0.09** | 0.21 | 0.71 | 0.61 | 241 → 6 |

**Result: negative.** The dilemma term helps a little before retraining. Retraining still removes the phantoms (6 candidates remain) and recovers the player just as well as against the plain flood: the residual protection is 0.10 against 0.13. A first-order conflict at the attacker's starting weights θ0 does not survive 12 epochs: after a few steps the backbone learns features that separate phantoms from players.

**Why, and what a real version needs.**
1. *There is no information-theoretic dilemma.* Players and phantoms are distinguishable in the frame (a human can tell), so an unbounded attacker can always learn both. Any guarantee must be against a **bounded** attacker: a model-capacity or latency budget, N harvested frames, and the time until rotation.
2. *The conflict must hold along the attacker's whole training trajectory, not just at θ0.* Next step: enforce the cosine term (or the unrolled post-fine-tuning objective S = recall × (1 − phantom survival)) at a **population of checkpoints**: θ0, the defender's own simulated-AT checkpoints after 1, 3 and 12 epochs, and earlier rounds' AT models (PSRO-style). Then iterate.
3. *Move "runtime" from NMS to model capacity.* NMS cost is negligible (§5.5), so the realistic cost axis is the capacity, and therefore the inference latency, needed to separate phantoms from players. The testable version of the dilemma: the attacker can retrain a small fast model (n/s) to get detection and no phantoms only with a model large enough to push its latency past the human-reaction budget. Measure the post-AT hit@15 against model size (v8n → s → m → l) and the latency at each size.

### 5.7 Capacity dilemma: model size as the attacker's cost (`capacity.py`)
**Setup (matches IC/AimTrap).**
* Detectors: original anchor-based YOLOv5 n/s/m (v7.0 weights), 320×320, conf 0.4, controller gain 0.4 (the RootKit AI-Aimbot defaults).
* Data: 400 COCO val2017 person images. A = 120 (defender), B = 160 (attacker harvest), T = 80 (test). The attacker adds 80 person-free images when retraining.
* Cloak: hide + EOT, ε = 8/255.
* Training: 5 epochs, AdamW, true labels, clean copies included.
* Cost: aiming steps to a hit × per-frame latency. Latency was measured on this 4-core CPU at batch 1: n ≈ 14–18 ms, s ≈ 33–36 ms, m ≈ 66–71 ms.

**The loss encodes the dilemma.** It defeats every model fast enough to be worth using, including retrained versions; YOLOv5m is deliberately excluded.
* Round 1: proxies = {v5n, v5s}.
* Round 2: proxies = {v5n, v5s, defender-retrained v5n, defender-retrained v5s}. The defender's models are trained on its own split A.

Metric: hit within 10 aiming steps.

| attacker | clean | round-1 cloak | round-2 cloak |
|---|---|---|---|
| v5n (off the shelf) | 0.83 | 0.00 | 0.00 |
| v5s (off the shelf) | 0.87 | 0.00 | 0.00 |
| v5m (off the shelf, not a proxy) | 0.92 | 0.71 | 0.70 |
| v5n retrained on harvested frames of that round | 0.78 / 0.83 | 0.55 | **0.38** |
| v5s retrained | 0.87 / 0.88 | 0.68 | **0.65** |
| v5m retrained | 0.90 / 0.82 | 0.85 | **0.78** |

(The "clean" column for retrained rows lists the round-1 / round-2 model.)

Time budget (CPU latencies; a step costs max(latency, 1000/144 ms)): P(hit ≤ 150 ms) under the round-2 cloak is 0.37 for n, 0.57 for s and 0.46 for m. Clean n and s give 0.83–0.84.

**Reading.**
1. **The size ordering holds.** After retraining against the round-2 cloak, recovery grows with capacity: n 0.38 < s 0.65 < m 0.78. Protection left (clean − cloaked): n 0.45, s 0.23, m 0.04. Only the large model fully sees through.
2. **The dilemma term works where it is applied hardest.** Adding the defender's own retrained n to the loss cut the attacker's retrained n from 0.55 to 0.38. It transfers across independent retraining runs on disjoint data.
3. **The weak point is v5s.** It is in the loss, but after retraining it recovers to 0.65 against 0.88 clean. A 2× latency step (n → s) buys the attacker most of the way back. On this CPU, the best attacker under a 150 ms budget is retrained s (0.57) rather than m (0.46), against 0.84 without the cloak. That is a real reduction, but not "the cheater must go slow".
4. **Hardware decides whether the latency tax bites.** On a gaming GPU, n/s/m all run in a few ms, so m's cost is negligible there. The dilemma only bites on CPU/laptop hardware, a GPU shared with the game, or if the escape model has to be far bigger (l/x, ensembles, purifiers). The next step is to measure on the target hardware.
5. **Caveats:** a single seed, small data (160 harvested frames), 5 epochs, COCO rather than game frames, and one round of iteration.

**Next:**
* Make the loss harder for v5s: a population of retrained checkpoints (several seeds and epochs) and more rounds.
* Add the phantom-flood term, so a retrained small model must also reject dozens of confident phantoms.
* Extend to l/x and measure GPU latency under game load.

### 5.8 Adding phantoms to the capacity dilemma (`CAP_OBJ=flood python -m aimbot_defense.capacity ...`)
**Objective.** The same pipeline as §5.7, but the cloak suppresses the real player (weight ×4) **and** grows confident 14×28 px phantoms. They sit on stride-8 anchor 1 (16×30), every 3rd cell, at least 12 px from any player. Hinge terms stop pushing anchors that are already confident or suppressed; 40 PGD steps with EOT.

The first version suppressed every non-phantom anchor and failed on YOLOv5: phantom confidence reached only ≈ 0.10, because neighbouring anchors share features and YOLOv5's score is obj × cls. Suppressing only the player's anchors fixed it.

Metric: hit within 10 steps; "≤150 ms" uses CPU latency per step (n 17, s 35, m 70 ms).

| attacker | frames | hide | hide ≤150 ms | flood | flood ≤150 ms | boxes/frame (flood) | false locks (flood) |
|---|---|---|---|---|---|---|---|
| v5n | clean | 0.83 | 0.83 | 0.83 | 0.83 | 1.5 | 2.6 |
| v5n | r1 / r2 | 0.00 / 0.00 | 0.00 | 0.03 / 0.03 | 0.03 | **32 / 22** | **43 / 39** |
| v5s | r1 / r2 | 0.00 / 0.00 | 0.00 | 0.04 / 0.04 | 0.04 | **52 / 38** | **46 / 44** |
| v5m (not a proxy) | r1 / r2 | 0.71 / 0.70 | 0.39 / 0.40 | 0.75 / 0.72 | 0.44 | 3.0 | 7–8 |
| v5n retrained | r1 | 0.55 | 0.53 | 0.56 | 0.55 | 0.7 | 1.6 |
| v5s retrained | r1 | 0.68 | 0.60 | 0.71 | 0.66 | 1.1 | 3.3 |
| v5m retrained | r1 | 0.85 | 0.50 | 0.85 | 0.49 | 1.5 | 4.6 |
| v5n retrained | r2 (clean: hide 0.83, flood 0.74) | **0.38** | 0.37 | 0.47 | 0.46 | 0.6 | 2.6 |
| v5s retrained | r2 (clean: 0.88 / 0.84) | 0.65 | 0.57 | 0.62 | 0.57 | 1.2 | 6.0 |
| v5m retrained | r2 (clean: 0.82 / 0.85) | 0.78 | 0.46 | 0.79 | 0.46 | 1.5 | 3.8 |

**Reading.**
1. **Against cheats that don't adapt, phantoms are a big win for detection, not for prevention.** n and s are blinded about as well as by hide (0.03–0.04 against 0.00). They also lock onto empty space **38–46 times per engagement** (≈ 3 on clean frames), which is unambiguous evidence of cheating, as in AimTrap's honeypots. Most real cheaters run stock models, so this matters in practice.
2. **Retraining wipes the phantoms out completely, in both rounds.** After retraining, the model sees 0.6–1.5 boxes per frame, at or *below* clean levels. The retrained models end up at the same hit rates as against hide alone (within ±0.1). Round 2 did not help the phantoms survive.
3. **Why: phantoms are dense, free negative labels for the attacker.** Every harvested frame carries dozens of confident "person" regions that the attacker's true labels mark as background. That is the strongest and easiest supervision signal in the whole retraining set. Hide, by contrast, forces the attacker to *recover* positives under the perturbation. Phantoms on background are therefore the easiest part of a cloak for an adaptive attacker to undo. This matches the YOLOv8 pilot (§5.5) and the failed first-order dilemma loss (§5.6).
4. **The size ordering is unchanged** (n < s < m after retraining), and so is the weak point (retrained s ≈ 0.62–0.65). Phantoms do not add a capacity lever.

**What could make phantoms survive retraining (untested ideas).**
* **Remove the free negatives.** Phantoms that the attacker cannot label as background without also labelling real players: for example, phantoms *attached to* players (extra heads or limbs that shift the aim point), so "suppress the phantom" and "detect the player" share the same pixels.
* **Make the negatives perishable.** Phantom layouts that change per session or patch, rotated faster than the attacker can harvest and retrain. This still gives free negatives each time, but costs a retraining cycle per rotation.
* **Use phantoms purely as a detector of stock cheats, and keep hide for prevention.** That is AimTrap's split, here in screen space.

### 5.9 Multi-checkpoint gradient conflict ("dilemma") in the YOLOv5 pipeline (`CAP_OBJ=dilemma CAP_GAMMA=1000`)
**Design: an exact ablation of flood round 2.** The proxies are identical {v5n, v5s, flood-r1 defender-retrained v5n and v5s}, and so are the seeds, data and attacker pipeline. One term is added to the loss:

  γ · cos(∇θ l_player, ∇θ l_phantom), γ = 1000.

Here θ = all weights of a YOLOv5n checkpoint. The checkpoint is drawn at random each PGD step from a simulated retraining trajectory: the base v5n, then the defender's own v5n retrained on flood-r1 frames after 1, 3 and 5 epochs (split A, disjoint from the attacker's split B).

**The term took effect.** Gradient cosine on the test cloaks, averaged over the 80 scenes:

| weights evaluated at | clean | flood r2 | dilemma |
|---|---|---|---|
| base v5n | −0.58 | −0.50 | **−0.76** |
| defender retrained, epoch 1 / 3 / 5 | −0.62 / −0.68 / −0.63 | −0.65 / −0.69 / −0.71 | **−0.91 / −0.92 / −0.94** |
| *attacker's* final retrained v5n (never seen by the defender) | −0.61 | −0.67 | **−0.83** |

The conflict transfers to the attacker's own trajectory: the cosine is still −0.83 at the end of *their* retraining.

**Outcome.** Hit within 10 steps; b = boxes per frame, fk = false locks per engagement.

| attacker | hide | flood | dilemma |
|---|---|---|---|
| stock v5n | 0.00 | 0.03 (b 22, fk 39) | 0.16 (b 8, fk 25) |
| stock v5s | 0.00 | 0.04 (b 38, fk 44) | 0.06 (b 27, fk 39) |
| stock v5m | 0.70 | 0.72 | 0.77 |
| retrained v5n (clean) | 0.38 (0.83) | 0.47 (0.74) | **0.49** (0.78), b 0.8, fk 4.9 |
| retrained v5s (clean) | 0.65 (0.88) | 0.62 (0.84) | **0.61** (0.83) |
| retrained v5m (clean) | 0.78 (0.82) | 0.79 (0.85) | **0.83** (0.86) |

**Result: negative again.** The attacker removed the phantoms (0.8 boxes per frame) and recovered the player as well as against plain flood: the protection left is 0.29 against 0.27. That happened *even though the gradients were still strongly opposed at the attacker's final weights* (−0.83). The only gains are about 2× more false locks for retrained n (4.9 against 2.6). The cost is weaker protection against stock n (0.16 against 0.03), because the gradient term takes budget from the flood and hide terms.

**Why a gradient-level dilemma cannot work (the key lesson).**
* Two gradients with cos > −1 always admit a direction that lowers **both** losses: the cone {d : d·g_pos < 0, d·g_neg < 0} is non-empty unless the gradients are exactly anti-parallel. At cos = −0.83 the non-opposed component is sin(arccos 0.83) ≈ 0.56 of the gradient, which is plenty of room in 1.9 M dimensions.
* This is the standard picture from multi-task learning (e.g. PCGrad): conflicting gradients slow joint learning and bias the path, but they do not prevent it. The attacker's optimiser simply follows the joint-descent cone.
* A real dilemma has to exist at the **function level**: no detector within the attacker's capacity budget can both detect players and reject phantoms on these frames. Phantoms are background + an ε = 8/255 pattern, so that is false for any reasonable detector. The evidence for it being false is exactly what we measured: retraining separates them every time.

**Implications for the project.**
1. *Drop gradient-geometry dilemmas.* They are measurable and transferable (−0.83 on unseen attacker weights is itself a nice finding), but they do not bind.
2. *A binding dilemma needs a representational argument.* Either the phantom and the player must be indistinguishable *in the input* to a capacity-bounded model (phantoms attached to or overlapping the player, sharing its pixels), or the separating function must be expensive, which is the capacity/latency route of §5.7. Both are claims about *functions*, so test them by measuring the best post-retraining hit rate per model size, not gradient cosines.
3. *Phantoms remain valuable as a tripwire* against stock and non-adaptive cheats (25–44 false locks per engagement).

### 5.10 How much latency can an "energy" cloak force? (bound for the narrative)
Only the post-processing is input-dependent; the CNN costs the same on every frame. Measured on CPU (4 threads, 320×320, batch 1):
* **Inference:** v5n 22.8 ms, v5s 36.5 ms, v5m 58.0 ms.
* **YOLOv5 `non_max_suppression`** (conf 0.4, IoU 0.45, `max_det` 300), with N candidates above threshold:

| N candidates | 2 | 50 | 300 | 1000 | 3000 | 6300 (every anchor) |
|---|---|---|---|---|---|---|
| **stacked on the player** (NMS keeps 1 box, so the output looks normal) | 0.25 ms | 0.26 | 0.35 | 0.76 | 1.37 | **1.93 ms** |
| **spread** (candidates survive as visible boxes) | 0.21 ms | 0.24 | 0.62 | 2.88 | 15.1 | **30.5 ms** (300 boxes out) |

* **A *stealthy* energy cloak can add at most ~1.7 ms per frame** (≈ 8 % of v5n), and only if *every* anchor clears the threshold. Our ε = 8/255 floods reached 20–50 candidates, i.e. ≈ 0.3 ms.
* **A visible flood can add up to ~30 ms**, but the overlay then shows hundreds of boxes, and capping candidates (`max_nms` / `max_det` / top-k) removes the cost.
* **Conclusion:** compute or energy slowdown cannot carry the "slow the aimbot" claim at this scale. It is a minor side effect at best.

## 6. Recommended next steps
1. **Move to real game frames and proxy ensembles** (CS2 / CF captures, or an open-source shooter with engine ground truth). Keep EOT on by default.
2. **Redesign the slowdown objective to be temporally correlated** (O8). Options: (a) per-engagement consistent aim bias, so the aimbot converges confidently onto a point just off the hitbox; (b) a decoy closer to the crosshair than the real player, kept stable for several frames and then moved, so the controller keeps re-acquiring; (c) offsets matched to the smoothing gain to cause overshoot. Measure the time-to-hit *distribution*, plus false locks as evidence.
3. **Make the adaptive loop the core of the method** (O3 + O4). Iterate defender-AT and attacker-AT rounds (PSRO-style population), and measure the **cross-family transfer matrix** for rotating perturbation families. The paper's claim would then be about attacker *cost* (frames and retraining time to regain hit@15 below human level) against rotation period, rather than "unbreakable".
4. **Add the missing adaptive attacks** from §7: temporal filtering (n-frame confirmation; this measures the latency tax directly), universal-cloak subtraction at scale, architecture swap, and diffusion/upscaler purification.

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
