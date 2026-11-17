# JCSSE 2027 submission plan

**Venue facts** (jcsse2027.mfu.ac.th):
* 4–6 pages A4, IEEE conference format, ~150-word abstract.
* Double-blind: anonymise this repo and any links. An arXiv version must use a different title.
* Generative-AI use must be declared at submission. This includes the experiment code and drafting done with Claude.
* Deadlines: **round 1 = 20 Nov 2026** (notification 8 Jan), **round 2 = 15 Jan 2027** (notification 5 Mar). Conference 16–18 Jun 2027, Chiang Rai.

## Working title
*Phantom Flooding: Imperceptible Decoys that Misdirect Visual Aimbots*

Keep "energy" out of the title. Compute cost is measured and reported, but at achievable budgets it is ≤ 1 ms (§5.10–5.13 of `options.md`), so it cannot be the headline.

## Claim (one sentence)
An imperceptible (ε = 8/255) perturbation that keeps the real player *detected at normal confidence* but adds *higher-confidence* phantom players. It exploits the aimbot's own post-processing (keep top-k by confidence, aim at the box closest to the crosshair), so a stock aimbot misses, and every cheap fix costs the cheater recall or produces ban evidence.

## Contributions
1. **Method:** the phantom-flood loss with a player-confidence band. Phantoms ≥ 0.95; player best box in [0.5, 0.7]; EOT; YOLOv5 n/s/m ensemble (the IC/AimTrap model family).
2. **Closed-loop evaluation against the real RootKit AI-Aimbot pipeline** (conf 0.4, IoU 0.4, `max_det` 10, closest to centre, gain 0.4). The metric is the hit rate within 10 aiming steps, not per-frame DSR, with the argument for why per-frame DSR overstates protection: 81.7 % DSR still gives ≈ 95 % detection within 15 frames.
3. **Counter-measure analysis:** top-1, keep-all and raised-threshold counters, the false-lock evidence they leave, and the measured NMS cost.

## Experiments (ε = 8/255, 80 test scenes)
| # | experiment | status |
|---|---|---|
| E1 | Main table: clean / IC-style hide / phantom (tuned) on v5n, v5s, v5m (proxies) and v5l, v8n (held out). Metrics: hit@10, player kept in output, player median conf, candidates, NMS ms | running (`out/paper_band_tuned`, `out/paper_hide`) |
| E2 | Counter-measure table (stock / top1 / nocap / thr0.85) | same run |
| E3 | Ablation: untuned weights, no clean pass (`out/paper_band_base`) | same run |
| E4 | Perceptibility: PSNR / SSIM / LPIPS of cloaked frames, plus an example figure | to do (cheap) |
| E5 | False-lock (honeypot) counts per engagement for stock models | have for flood (§5.8); add for E1 frames |

## Limitations to state explicitly
* **An adaptive cheater who retrains** on harvested frames strips the phantoms (§5.8–5.9). Report it as a limitation, with the numbers.
* **Weak transfer** to unseen architectures (v5l, v8n).
* **COCO person images** stand in for game frames. This is the biggest reviewer risk for a game paper.
* CPU-only latency numbers.

## Out of scope for this paper
The capacity dilemma (§5.7), gradient conflict (§5.6, §5.9), rotation, real-game deployment and a user study. These are candidates for a longer follow-up.

## Round 1 or round 2?
* **Round 1 (20 Nov):** feasible only with COCO frames. Experiments finish today, leaving about 2 days of writing.
* **Round 2 (15 Jan): recommended.** It allows a small real game-frame set (e.g. self-recorded CS2/CrossFire footage at 320×320 with labels), ε = 16 results, and one more held-out model (NanoDet), which address the two main reviewer risks.
