# Data-overlap audit (2026-11-13)

**Script:** `src/diagnostics/overlap_audit.py` (CPU only; the test manifest is never read). **Results:** `overlap_audit_20261113.json`.

**Method:**
- A 64-bit difference hash of every training input (4,135) is compared with each candidate image.
- Every hit within Hamming distance 8 is verified at the pixel level (128² bilinear, PSNR).
- The positive control passed: training images matched themselves.

## 1. Candidate zero-shot sets vs OUR training split

| Set | n | Exact hash | Verified duplicates (≥ 35 dB) | Near (≥ 25 dB) | Usable for zero-shot? |
|---|---|---|---|---|---|
| EUVP test_samples (official EUVP test) | 515 | 387 | **409** | 410 | **No.** These images are inside LSUI (e.g. `test_p0_` = `LSUI:1991`). |
| EUVP Paired/underwater_scenes | 2,315 | 1,332 | **1,435** | 1,443 | **No.** |
| EUVP Paired/underwater_imagenet | 4,970 | 17 | 18 | 18 | Yes, after excluding the flagged images |
| EUVP Paired/underwater_dark | 6,120 | 0 | **0** | 0 | **Yes** (preferred paired zero-shot set) |
| UIEB challenge_60 (unpaired) | 60 | 0 | 0 | 0 | **Yes** (no reference: no-reference metrics and uncertainty only) |

The EUVP copy is a Kaggle mirror (`pamuduranasinghe/euvp-dataset`, metadata licence CC0-1.0), not the official IRVLab distribution. Cite the original EUVP paper and its licence terms.

## 2. Our validation split vs our training split (near-duplicate frames)

- 20 of 428 LSUI validation images have a training neighbour at ≥ 25 dB pixel-PSNR, typically adjacent frames with nearby LSUI IDs (e.g. val 1109 vs train 1111 at 31.0 dB).
- **0 of 89 UIEB** validation images do.
- An exploratory pass with a wider hash window (≤ 12) found 29 LSUI at ≥ 25 dB and 8 at ≥ 28 dB.
- **Effect:** absolute LSUI validation numbers are slightly optimistic for all models. Paired comparisons are unaffected in expectation, because every model sees the same images. The UIEB result (Gate G1) is unaffected.
- **The test split was not checked** (rule: touched once). The same split procedure means similar LSUI near-frames are likely there. Recommendation: before Phase F, report LSUI test results also with near-duplicate frames excluded.

## 3. Pretrained checkpoints vs our validation split (name overlap with published training lists)

| Published training list (UDnet repo) | Overlap with our validation |
|---|---|
| `splits/V2/LSUI_train.txt` (3,160) | **320 / 428** LSUI |
| `splits/V2/UIEB_train.txt` (738) | **76 / 89** UIEB |
| `splits/UIEB/train800.txt` (the standard UIEB-800 training list) | **81 / 89** UIEB |

**Consequence:** any checkpoint trained on the standard UIEB-800 / LSUI-train lists saw 75–91% of our validation images, and most likely a similar share of our test images. **Pretrained ("Tier-2") checkpoints must not be scored on our validation or test splits.** They can only be compared fairly on sets that neither they nor we trained on (e.g. UIEB challenge_60, EUVP underwater_dark, after checking each checkpoint's training data).
