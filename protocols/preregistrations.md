# Pre-registrations

The sections below are copied verbatim from the project's execution log. Each was committed to the development repository before the run it governs. That repository's history is not part of this snapshot, so the commit timestamps can be checked only on request.

## L. Phase 2: multi-seed replication, P1b vs v2 (PRE-REGISTRATION, 2026-11-11, before any seed-1/2 result)

**PI decisions:**
- decoder FiLM rejected (K16);
- P1b (v2 + ordered DGT, direct supervision) locked as the primary model.

**Arms**, run sequentially in this order, same commit, Stage-1 schedule (100k steps, AdamW, cosine, batch 4 × 2):
1. P1b seed 1;
2. v2 seed 1;
3. P1b seed 2;
4. v2 seed 2.

Launch record: `runs/phase2/phase2_launch.json` (SHA-256 `f6e9839e…453f`).

**Criterion (§D Stage-2 rule, fixed now).** P1b is called *contributing* on a dataset only if, **for every seed s ∈ {0, 1, 2}**, the paired 517-image mean ΔPSNR₈ (P1b_s − v2_s) is **≥ 0.2 dB** and its 95% bootstrap CI (B = 2000, seed 20261107) **excludes 0**, in the same direction.
- Primary dataset: UIEB (the seed-0 candidate, K13).
- LSUI and ALL are reported under the same rule.
- Otherwise the result is "not shown to contribute".

Seed 0 is already measured: UIEB +0.613 [+0.091, +1.108]; LSUI +0.002; ALL +0.107.

**Evaluation:**
- each checkpoint gets the 517-image evaluation as it completes, with `last.pt` at step 100k only;
- pairing is within seed (P1b_s vs v2_s);
- the across-seed summary reports each seed's interval and the mean of the per-seed differences.

**Note.** `--val-subset` sampling depends on `--seed`, so the training-time monitoring subsets differ across seeds and are not comparable. No decision uses them. The test split is not accessed.

### L1. P1b seed 1: audit and 517-image evaluation (2026-11-11)

**Run:** `runs/phase2/P1b_s1/20261111_070454`, commit `6639e5a`, seed 1.
- Exit 0; wall 32,281 s (0.374 GPU-days).
- 100,000 contiguous records; 41 fp16 overflow skips; 0 non-finite losses.
- `last.pt` SHA-256 `140f0e4f6cf60bbf7e7f186acae6f10fa3923f617654ebefd8bd97db4d888f88`; strict load OK.
- `best.pt` (archival only): step 82,000.

**Telemetry** (mean of the last 1k steps), closely reproducing seed 0 (K12):
- β_D (R, G, B): 0.946, 0.921, 0.920;
- β_B (R, G, B): 1.782, 1.782, 1.704;
- mean t̂D: 0.638 (by step: 0.895 at 1k, 0.789 at 5k, 0.707 at 20k, 0.665 at 50k);
- ordering: no ties.

**517-image evaluation:**
- `phase2_eval517.py`, commit `8acc82b`; output `runs/diagnostics/P2_EVAL517_P1b_s1_20261111_160445`; 403 s.
- Parity PASS (0.0048 dB); all 517 images ran on CUDA.
- Per-image CSV SHA-256 `0a1b679a8d4c954f650deba18138a89c128b934a570da1a69bc30a0404003559`.
- The pairing mode was validated first by reproducing K13 (P1b_s0 − v2_s0) exactly.

| Dataset | PSNR₈ | SSIM₈ |
|---|---|---|
| LSUI | 28.266 [27.781, 28.733] | — |
| UIEB | 22.902 [21.874, 23.933] | — |
| ALL | 27.343 [26.902, 27.769] | 0.9112 |

Seed 0 for reference: LSUI 28.333, UIEB 22.623, ALL 27.350. The within-seed comparison (P1b_s1 − v2_s1) awaits v2 seed 1.

### L2. v2 seed 1: interruption, resume, audit, and the seed-1 paired result (2026-11-12)

**Interruption:**
- At 00:31:42 IST Windows Update forced a restart (System event 1074, TrustedInstaller, "Operating System: Upgrade (Planned)").
- The run was killed at logged step 92,913. It was healthy up to that point: 0 non-finite losses, 37 skips.
- **PI decision: Option A, resume.**
  - The pre-resume `last.pt` (step 92,000, RNG state present) was archived as `last_step092000_preresume.pt` (SHA-256 `d6119c55…76f8`).
  - Resumed on the identical commit `5e38d7c` (src diff 0 lines) with `--seed 1`; record in `runs/phase2/phase2_resume_v2_s1.json`.
  - Restored: model, optimizer, GradScaler, step and RNG. **Not restored:** the DataLoader shuffle position after step 92,000, so the last 8% of steps (lr 8.8e-6 → 1e-6) used a different sample order than an uninterrupted run would have.
  - **This arm is a resumed run. Disclose it in any write-up.**

**Audit:**
- Exit 0. Wall time ≈ 29,520 s (to step 92,000) + 2,561 s (resume).
- Steps 1–92,000 come from `train_log.jsonl`; the 913 records for 92,001–92,913 are superseded. Steps 92,001–100,000 come from `train_log.resume1.jsonl`. **Together they are contiguous 1–100,000.**
- 41 skips; 0 non-finite losses; GradScaler final 131,072.
- `last.pt` (step 100,000) SHA-256 `9030037adcb124706f1577bb8c194f5ad3358ff2373e6c2c99858a0604875dc0`; strict load OK; 4,152,052 parameters; v2.
- `best.pt` (archival only): step 76,000.
- Last 1k steps: mean t̂D 0.573, t̂B 0.796.
- Step-100k subset (seed-1 subset `2d00cea3…`, the same as P1b seed 1): PSNR 26.879, LSUI 27.508, UIEB 23.810, SSIM 0.9085. P1b seed 1 on the same subset: 27.476, 28.163, 24.119.

**517-image evaluation:**
- Output `runs/diagnostics/P2_EVAL517_v2_s1_20261112_022006`; 342 s.
- Parity PASS (0.0011 dB); all 517 images ran on CUDA.
- Per-image CSV SHA-256 `b112eef4…d39f`.
- Results: LSUI 28.045 [27.570, 28.510]; UIEB 22.311 [21.343, 23.344]; ALL 27.058 [26.637, 27.487]; SSIM 0.9100.
- For comparison, v2 seed 0 scored ALL 27.243.

**Seed-1 paired result, P1b_s1 − v2_s1** (copy: `reports/experiments/phase2/pair_P1b_s1_minus_v2_s1.json`):

| Dataset | ΔPSNR₈ | ΔSSIM₈ | §L per-seed test (≥ 0.2 dB and CI excludes 0) |
|---|---|---|---|
| LSUI | +0.221 [+0.004, +0.438], d_z +0.09 | −0.0003 [−0.0016, +0.0008] | met (lower bound +0.004) |
| UIEB | **+0.592 [+0.107, +1.084]**, d_z +0.25 | **+0.0083 [+0.0022, +0.0142]** | **met** |
| ALL | +0.285 [+0.094, +0.480], d_z +0.12 | +0.0012 [−0.0002, +0.0026] | met |

**Status against the §L criterion** (needs all of seeds 0, 1 and 2):

| Dataset | Seed 0 (K13) | Seed 1 | Seed 2 |
|---|---|---|---|
| UIEB | met (+0.613 [+0.091, +1.108]) | met | **not run** (held by PI) |
| LSUI | not met (+0.002) | met | not run |
| ALL | not met (+0.107) | met | not run |

**No conclusion is drawn before seed 2.**
- UIEB is consistent in direction and size across the two seeds.
- The LSUI/ALL difference between seeds coincides with v2 seed 1 scoring 0.19 dB below v2 seed 0 overall (27.058 vs 27.243). P1b is stable across seeds (27.350 vs 27.343).

### L3. P1b seed 2: completion audit (2026-11-12)

**Run:** `runs/phase2/P1b_s2/20261112_082323`, commit `35b893d`, seed 2.
- Ran entirely inside Windows active hours (08:00–00:00). Windows Update was not paused; no interruption occurred.
- Exit 0; wall 32,355 s (0.374 GPU-days).
- 100,000 contiguous records; 42 fp16 overflow skips; 0 non-finite losses or loss terms; GradScaler final 32,768.
- `last.pt` (step 100,000) SHA-256 `684b5a324ab71e1a0d004168af563c03e436f96c56c3401bfd9ed9b489956430`; strict load OK; 4,284,413 parameters.
- `best.pt` (archival only): step 64,000.

**Telemetry** (mean of the last 1k steps):
- β_D (R, G, B): 0.915, 0.888, 0.888;
- β_B (R, G, B): 1.777, 1.777, 1.654;
- mean t̂D 0.644 (by step: 0.896 at 1k, 0.793 at 5k, 0.731 at 20k, 0.684 at 50k), mean t̂B 0.420.

**Jerlov ordering:**
- 0 violations and 0 exact float32 ties in 100,000 steps.
- Smallest per-sample gaps: β_D R − G 4.2e-6 and G − B 6.1e-6; β_B R − G 7.2e-7 and G − B 0.022.
- Mean gaps over the last 1k steps: β_D R − G 0.026 and G − B 0.0006; β_B R − G 0.0001 and G − B 0.123.
- As in seeds 0 and 1, the constraint is binding: β_D,G ≈ β_D,B and β_B,R ≈ β_B,G.
- Barrier maxima: 3.4e-3 (upper) and 1.4e-3 (lower).

**Step-100k subset** (seed-2 subset `f28c5292…`; not comparable with other seeds' subsets):
- PSNR 26.509, LSUI 27.498, UIEB 21.678, SSIM 0.8967, NLL −1.282, corr 0.308.

**Pending:** the 517-image evaluation, and v2 seed 2 (held by the PI).

### L4. Seed 2 and the pre-registered Gate G1 verdict (2026-11-13)

**v2 seed 2 audit** (`runs/phase2/v2_s2/20261112_174530`, commit `5f59b5f`; the src code is identical to all Phase 2 runs; Windows Update paused):
- Exit 0; wall 32,127 s (0.372 GPU-days).
- 100,000 contiguous records; 42 skips; 0 non-finite losses.
- `last.pt` SHA-256 `0549d7445b07629e412cd0dbbcc95ac5ca1eadba1deb98e7351e3eba72d23816`; strict load OK; v2.
- `best.pt` (archival only): step 84,000.
- Last 1k steps: mean t̂D 0.570, t̂B 0.797.
- Step-100k subset (seed-2 subset, the same as P1b seed 2): v2 26.331 (LSUI 27.379, UIEB 21.218); P1b 26.509 (27.498, 21.678).

**517-image evaluations:**

| Arm | LSUI | UIEB | ALL | SSIM ALL | Per-image CSV SHA-256 |
|---|---|---|---|---|---|
| P1b seed 2 (parity 0.0054 dB) | 28.168 | 22.765 | 27.238 | 0.9112 | `b1c1bc2f…488c` |
| v2 seed 2 (all CUDA) | 28.015 | 22.055 | 26.989 | 0.9085 | `bd6beb4c…23d2` |

**Seed-2 paired result, P1b_s2 − v2_s2:**

| Dataset | ΔPSNR₈ | d_z | ΔSSIM₈ |
|---|---|---|---|
| LSUI | +0.154 [−0.049, +0.375] | +0.07 | +0.0016 [+0.0004, +0.0029] |
| UIEB | **+0.711 [+0.146, +1.311]** | +0.26 | +0.0079 [+0.0021, +0.0144] |
| ALL | +0.250 [+0.063, +0.450] | +0.11 | +0.0027 [+0.0013, +0.0041] |

**Gate G1: pre-registered §L verdict.** The rule: for every seed, ΔPSNR₈ ≥ 0.2 dB and the 95% CI excludes 0, in the same direction.

| Dataset | Seed 0 | Seed 1 | Seed 2 | Mean of per-seed Δ | **Verdict** |
|---|---|---|---|---|---|
| **UIEB** | +0.613 [+0.091, +1.108] ✓ | +0.592 [+0.107, +1.084] ✓ | +0.711 [+0.146, +1.311] ✓ | +0.639 | **CONTRIBUTING (criterion met in all 3 seeds)** |
| LSUI | +0.002 [−0.210, +0.220] ✗ | +0.221 [+0.004, +0.438] ✓ | +0.154 [−0.049, +0.375] ✗ | +0.126 | not shown to contribute |
| ALL | +0.107 [−0.085, +0.300] ✗ | +0.285 [+0.094, +0.480] ✓ | +0.250 [+0.063, +0.450] ✓ | +0.214 | not shown to contribute (fails in seed 0) |

**Statement within the evidence:** on the 517-image validation split, adding the ordered depth-grounded transmission (DGT) to v2 improves UIEB PSNR₈ by +0.59 to +0.71 dB in each of 3 seeds. Each per-seed CI excludes 0, and the result meets the pre-registered ≥ 0.2 dB rule. No such effect is shown on LSUI or overall.

**Not claimed / caveats:**
- **Validation, not held-out.** This split was used repeatedly for decisions; the test-split confirmation (roadmap phase F) is still required.
- **v2 seed 1 is a resumed run** (L2).
- **No multiplicity correction** across the comparisons made in this project.
- **No physical identification:** the Jerlov ordering is binding (L3).
- 89 UIEB images.

GPU accounting: Phase 2 (4 runs + resume + evaluations) ≈ 1.53 GPU-days; cumulative ≈ 4.65 of 7.

## M. Phase C: global scene prior (PRE-REGISTRATION, 2026-11-13, before any Phase C training)

**PI decision:** the Phase C upgrade is a foundation-model scene prior (the novelty option), not EMA + larger crops.

**Design** (implemented as a **global** descriptor; the reasoning was given to the PI):
- **Descriptor:** g = [CLS, mean of patch tokens] from the frozen Depth-Anything-V2-Small encoder's last layer (768-d), computed once per **full** image offline (`src/data/precompute_scene_features.py`; same pinned model and weights SHA as the depth maps).
- **Mapping:** h = GELU(Linear(768 → 64)), feeding
  1. a channel FiLM on the bottleneck input, f ← f·(1 + γ) + β, and
  2. an additive term on the 9 pre-activations of the ordered-DGT head.
- **Initialisation:** both output layers are zero-initialised. No decoder injection (avoids the P1c bypass, K15/K16).
- **Size:** +116,361 trainable parameters (model 4,400,774 ≤ 5M); the encoder is frozen and offline.
- **Motivation:** training crops (256 px) then see their whole image's context, and the DGT β is no longer estimated from crop-local features only.

**Verification** (`src/diagnostics/scene_check.py`, CPU): ALL PASS.
- Bit-identical to P1b seed 0 at zero init for any g (random inputs and 4 real images at native size); the 1e-3 control breaks it.
- Gradient routing at init is as designed, and after the output layers are non-zero g changes β.
- The ordering is preserved and all input contracts hold.
- `film_check` and integration check A (v1/v2 bit-identical to HEAD) still pass.

**Run:**
- P1d = `--phys-conditioning v2 --dgt --dgt-ordered --scene-prior`, seed 0, Stage-1 schedule, 100k steps, trained from scratch (not warm-started), `last.pt` only.
- Matched comparator: P1b seed 0 (`5b22ea7c…`).

**Gate G2 (fixed now):** adopt P1d over P1b only if the paired 517-image ΔPSNR₈ (P1d_s0 − P1b_s0) is **≥ +0.2 dB with the 95% bootstrap CI (B = 2000, seed 20261107) excluding 0, on ALL or on UIEB.** Otherwise P1b stays the model. Either outcome is reported.
- **Single seed:** any adoption is provisional until replicated, which is outside the current budget unless extended.

## O. Phase D: external baselines (PRE-REGISTRATION, 2026-11-13, before any baseline training)

**PI decisions:**
- one **unified recipe** for all baselines;
- U-shape evaluated with 256×256 resize and bilinear upsampling back to native size (the adapter's `resize256` mode, bicubic; change it to bilinear before U-shape is evaluated if the PI wants it strictly);
- Spectroformer uses the default variant (`Final_model_AGSSF.mymodel`);
- run order: Five A⁺ → U-shape → Spectroformer → PhaseFormer *or* WWE-UIE → PUGAN, then P1e (report section P, to be pre-registered).

**Unified recipe** (`baselines/adapters/train_baseline.py`):
- **Data:** the same as SS-PGN: `train.PairedCropDataset`, 4,135 training images, 256 crops, rot90 + flip, mask.
- **Objective:** `SSPGNLoss.recon` = L1 + 0.2·(1 − SSIM) + 0.1·wavelet. These are the SS-PGN reconstruction terms; baselines have no physics or NIG outputs.
- **Optimiser:** AdamW (0.9, 0.9), lr 5e-4, 1k warmup, cosine to 1e-6, 100k steps, batch 4 × 2, clip 1.0, fp16 AMP, seed 0. The schedule is identical to SS-PGN's.
- **Not used:** official GAN, perceptual or colour-space losses and official schedules. This is an *architecture comparison under an identical recipe*, and will be stated as such.
- **Fallback (fixed now):** if a baseline's 500-step pilot diverges at lr 5e-4, it uses lr 2e-4 and this is reported.

**Selection and evaluation:**
- `last.pt` at step 100k only.
- 517-image native-resolution evaluation with `registry.predict` and the same metric code.
- Paired bootstrap against P1b seed 0 and v2 seed 0 (B = 2000, seed 20261107).
- No baseline result changes any SS-PGN design decision.

**Budget:** about 1.9 GPU-days remain. Each baseline's measured pilot step time sets its projected cost. The PI is told before the cumulative budget would be exceeded.

### O1. Precision exception (disclosed) and Five A⁺ launch (2026-11-13)

**Exception:** Five A⁺ uses FFT/phase operations (`torch.angle`) that CUDA does not implement in fp16 ("angle_cuda not implemented for Half" at step 1). It is therefore trained in **fp32** (`--no-amp`). Every other part of the recipe is unchanged. The same applies to any baseline that fails the same way (Spectroformer uses `torch.fft`); each instance will be recorded.

**Five A⁺ pilot** (500 steps, fp32): stable; L1 0.312 → 0.118; validation 18.47 dB at step 500 (lr only at 2.5e-4 by then, warmup 1k); 0.093 s/step, so 100k ≈ 2.6 h (0.11 GPU-days). Divergence at lr 5e-4 is checked during the full run; the §O fallback applies if it diverges.

### O2. Phase D decisions and automation (PI, 2026-11-13)

**Decisions:**
- WWE-UIE (WACV 2026) is selected over PhaseFormer.
- **PUGAN is excluded:** its official repository omits the pretrained auxiliary physical-parameter networks (`Par/model/*.pth`) its generator requires, so leak-free training from scratch on our split is not possible with the released code.
- FUnIE-GAN is excluded: the local copy is not a pinned git clone.
- U-shape native evaluation: bilinear (antialiased) 256² downscale and bilinear upscale back (registry updated).

**Automation:** `baselines/adapters/phase_d_driver.py`.
- Sequence: Five A⁺ evaluation and pairing, then for each of U-shape → Spectroformer → WWE-UIE: a 500-step pilot (fp32 on a half-precision error, per O1; lr 2e-4 on divergence, per O), then a budget check, a 100k run, and evaluation and pairing.
- **The driver stops** if a launch would leave less than the 0.38 GPU-day P1e reserve (starting budget 1.88 GPU-days).
- Evaluation: `baselines/adapters/eval517_baseline.py`, with the same protocol and gates as the SS-PGN evaluations. Status log: `runs/baselines/phase_d_status.json`.

## P. P1e: regularised scene prior for the DGT β head only (PRE-REGISTRATION, 2026-11-13, before training)

**Motivation:** P1d failed G2 (M1). Its training loss fell about 30% below P1b's while validation got worse, consistent with the 768-d descriptor acting as an image identifier through the bottleneck FiLM. Validation images whose descriptors resemble training ones improved (+0.67 dB in the top similarity quartile).

**Design** (PI decision 2026-11-13; flag `--scene-prior-pca 16`, off by default):
- **Training-only PCA** to 16-d (`src/data/fit_scene_pca.py`):
  - fitted on the 4,135 training descriptors only, whitened;
  - 58.1% of variance retained;
  - `data/scene/.../pca16_train.npz` is bound to the training-file SHA-256;
  - validation and test are transformed only, never refitted.
- **Descriptor routed only to the 9 ordered-DGT pre-activations** (`scene_mode = dgt_only`). There is no bottleneck or decoder FiLM path, so g cannot alter features except through β and B∞.
- **Training-only regularisation:**
  - per-sample descriptor dropout p = 0.5, setting it to the training mean;
  - Gaussian noise σ = 0.25 in whitened units. σ is an assistant choice; the PI specified "Gaussian noise" without a value.
- **Initialisation and size:** zero-initialised output layer; +1,673 parameters (model 4,286,086).

**Verification** (`src/diagnostics/p1e_check.py`, CPU): ALL PASS.
- Bit-identical to P1b seed 0 at zero init in eval **and** train mode.
- The bottleneck input is provably independent of g.
- Measured dropout fraction 0.497 over 4,000 draws; noise std 0.253.
- Eval mode is deterministic.
- PCA store returns whitened 16-d vectors.
- The first control design (`scene_dgt.weight[0,0]`) was inert (its hidden unit was ~0 for the control input). It was redesigned to `scene_dgt.bias[0]`, which then tripped correctly.
- `scene_check`, `film_check` and integration check A still pass (P1b/P1d/v1/v2 defaults unchanged).

**Run:** `--phys-conditioning v2 --dgt --dgt-ordered --scene-prior-pca 16`, seed 0, Stage-1 schedule, 100k steps, `last.pt`, started after the Phase D queue, using the 0.38 GPU-day reserve.

**Gate G2 (unchanged):** adopt over P1b only if the paired 517-image ΔPSNR₈ (P1e_s0 − P1b_s0) is ≥ +0.2 dB with the 95% CI excluding 0, on ALL or UIEB. Otherwise P1b remains the model. Single seed; provisional.

## Q. Phase F: single test-split evaluation (PRE-REGISTRATION, 2026-11-17, before the test split is read)

**Context:**
- Gate G2 failed on 2026-11-17, so **P1b_s0 is the flagship model**. P1e: ΔPSNR₈ vs P1b_s0 ALL −0.111 [−0.286, +0.058], UIEB +0.097 [−0.384, +0.600].
- Phase D is closed with 3 retrained baselines.
- Phase E (zero-shot) is complete.
- The test split (`data/splits/lsui_uieb_all_d6_seed0/test_manifest.txt`, 517 pairs) has never been read by any training, selection or evaluation code. It is opened **once**, by `src/diagnostics/phase_f_test.py --split test`, after this section is committed.

### Frozen model list (roadmap F1; each `last.pt` at step 100000, SHA-256 pinned in the script)

| Group | Models |
|---|---|
| Flagship | **P1b_s0** (`5b22ea7c…`) |
| Seed replicates | P1b_s1 (`140f0e4f…`), P1b_s2 (`684b5a32…`) |
| v2 (no DGT) | v2_s0 (`8a50ea4d…`), v2_s1 (`9030037a…`), v2_s2 (`0549d744…`) |
| G2 comparator | P1e_s0 (`f0ac978c…`) |
| Stage-1 ablations | v1_retrain (`42e1b674…`), A2_no_physics (`62f7604c…`), A3_v2_no_Lphys (`91a4fdea…`), A4_v2_no_NIG (`f70fb83c…`) |
| Retrained baselines | Five A+ (`61cf5174…`), U-shape (`8057943e…`), WWE-UIE (`c0e92616…`) |

Not tested:
- P1, P1c and P1d (rejected designs);
- Spectroformer (excluded for cost);
- external pretrained checkpoints (never on LSUI/UIEB).

### Protocol
Identical to the 517 validation evaluation:
- fp32, native resolution, PadCrop to the model multiple;
- CUDA with CPU-on-OOM fallback, allocator capped at 0.50;
- GPU-vs-CPU parity on 4 images per model (|ΔPSNR₈| < 0.01 dB) with the shared ±8 checkerboard failure control.

Depth and the P1e scene descriptor for test images are computed exactly as in precomputation:
- Depth-Anything-V2-Small @5426e4f, fp16, P1/P99, stored as float16.
- Descriptor = [CLS, mean patch token], projected by the training-only PCA-16.

A fail-closed preprocessing check runs before the test split is opened: on 8 validation images, on-the-fly depth and descriptors must reproduce the stored training-time values (depth max |Δ| ≤ 2e-3; PCA-16 max |Δ| ≤ 5e-2).

### Metrics and groups
- **Per image:** PSNR₈, SSIM₈, LPIPS (alex, pinned: lpips 0.1.4, lin `df73285e…`, AlexNet `7be5be79…`), UIQM, UCIQE.
- **Groups:** LSUI, UIEB, ALL (stratified), and the **near-duplicate-filtered** LSUI subset with its ALL.

### Near-duplicate rule (fixed now, computed before any model output)
- A test **input** is a near-duplicate of training if:
  - its 64-bit dHash is within Hamming distance ≤ 8 of a **training** input, and
  - the pair has pixel PSNR at 128×128 ≥ 25 dB.
- This is the same rule as `src/diagnostics/overlap_audit.py`.
- Near-duplicates are excluded from the "LSUI_dedup" and "ALL_dedup" groups. Full-split numbers are always reported as well.

### Statistics
- Paired bootstrap of (model − P1b_s0) per image: B = 2000, seed 20261107, percentile 95% CI, d_z, stratified for ALL.
- Seed-matched P1b − v2 for seeds 0/1/2, plus their mean.

### Primary claims (confirmatory, two-sided 95% CI)
1. P1b_s0 vs each retrained baseline on ALL PSNR₈.
2. P1b vs v2 seed-matched on UIEB PSNR₈ (Gate G1 replicated on test).

Everything else is descriptive.

### Qualitative grid
6 test units (4 LSUI, 2 UIEB) are drawn with `numpy.random.default_rng(20261117)` from the manifest order **before** any output exists. Their outputs are saved for all models in the grid.

## S. F3: conformal calibration of the NIG uncertainty + 3-seed ensemble (PRE-REGISTRATION, 2026-11-17, before F3 reads any test image)

**Scope.**
- Roadmap F3 was planned before any test access.
- No retraining; the frozen P1b_s0, P1b_s1 and P1b_s2 `last.pt` (SHA as in section Q).
- **The Phase F single-model test numbers (section R) remain the primary locked result.**
- Every ensemble number on test is **post-hoc / supplementary**.

**Test access.** `src/diagnostics/f3_conformal.py --split test --confirm-f3` reads the test split once more, only for F3. Its lock file is `runs/phase_f/F3_TEST_OPENED.json`, which refuses a second F3 read. No point prediction is tuned on test.

**Predictors.**
- (i) P1b_s0: μ and σ = NIG total predictive std, σ² = β(1+ν)/(ν(α−1)).
- (ii) Ensemble E3: μ = mean of the three seeds; σ² = mean(σ_i²) + Var_i(μ_i).

**Data.**
- Per image, 4096 pixel positions (1024 for underwater_dark, for memory) drawn with `numpy.random.default_rng(crc32(unit))`, all 3 channels.
- Residual r = |y − μ| in [0, 1] units; σ as above.
- Sets: validation (517, calibration), test (517), EUVP underwater_dark (5550 pairs, zero-shot).

**Calibration on validation only.** Split conformal with nominal levels 1 − a ∈ {0.80, 0.90, 0.95}.
- **Adaptive:** score s = r/σ, q̂ = the ⌈(n+1)(1−a)⌉/n empirical quantile, interval μ ± q̂σ.
- **Reference A:** uncalibrated NIG Gaussian, μ ± z₍₁₋ₐ/₂₎σ.
- **Reference B:** non-adaptive constant width, μ ± q̂_r, with q̂_r the conformal quantile of r.

**Reported on test and underwater_dark.**
- Marginal pixel coverage.
- Per-image coverage: mean and 5th percentile.
- Mean half-width ×255 (sharpness).
- Spearman ρ(σ, r) (adaptivity).

**Caveat stated in advance.** Pixels within an image are not exchangeable. The finite-sample guarantee holds for exchangeable calibration and test draws; pixel-level coverage is therefore reported as an empirical marginal quantity, together with its per-image distribution.

**Ensemble point accuracy.**
- PSNR₈ and SSIM₈ of E3 vs. P1b_s0 on validation (decision set) and underwater_dark (zero-shot), paired bootstrap B = 2000, seed 20261107.
- The test row is post-hoc.
- **E3 is used for the uncertainty claim only if its validation coverage/sharpness or PSNR beats P1b_s0's on validation.**

**Also in this step** (no test access):
- Zero-shot evaluation of external pretrained checkpoints, each with its own official protocol:
  - MARINE-Net-LSUI on underwater_dark and C60;
  - Five A+ official α = 0.1 / 0.4 on underwater_dark and C60;
  - UIR-PolyKernel and PhaseFormer-UFO120 on C60 only (training data unverifiable for underwater_dark).
- A CPU OpenCV downstream check on saved outputs for C60 plus 200 underwater_dark images (rng 20261117): SIFT/ORB RANSAC inliers against a fixed synthetic homography, and Canny edge density and mean edge-component length.
