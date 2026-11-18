# External baselines: workspace, provenance and compatibility (2026-11-10)

**Status:** repositories cloned and audited, adapters written, CPU dry run passed. **No baseline has been trained or evaluated.** The work was done on CPU only (`CUDA_VISIBLE_DEVICES=-1`, with `torch.cuda.is_available() == False` asserted) while Phase 1c trained on the GPU (PID 19680, untouched). No package was installed or changed.

## Layout
- `repos/`: shallow clones of the official repositories, pinned below. They are not committed (`.gitignore`).
- `adapters/registry.py`: builds each architecture from its clone (random init) and wraps `forward()` and the native-resolution input-size handling.
- `adapters/data.py`: our split, crop dataset and PSNR₈/SSIM₈ code, **imported from `src/`** rather than re-implemented, so the protocol is identical by construction.
- `adapters/cpu_dry_run.py`: the CPU dry run. Its report is `cpu_dry_run_report.json`.

## Repositories

| Key | Paper | Repository (commit) | Licence |
|---|---|---|---|
| `ushape` | U-shape Transformer, Peng et al., IEEE TIP 2023 | github.com/LintaoPeng/U-shape_Transformer_for_Underwater_Image_Enhancement (`41f98976`) | MIT |
| `fiveaplus` | Five A⁺ Network, Jiang et al., BMVC 2023 | github.com/Owen718/FiveAPlus-Network (`c15e2234`) | **none stated** |
| `spectroformer` | Spectroformer, Khan et al., WACV 2024 | github.com/Mdraqibkhan/Spectroformer (`0a0c17b3`) | MIT (README) |

**How each was identified as official:**
- U-shape: the README calls it "the official PyTorch implementation", and the account belongs to the first author. The arXiv abstract page itself gives no code link.
- Five A⁺: the README says "office implementation"; the repository was found by web search.
- Spectroformer: the README names the authors and links the WACV paper.

**WaterFormer:** no official code repository was found (web search, 2026-11-10). It is not included.

## CPU dry run (`cpu_dry_run_report.json`)

| Key | Declared params | Used in forward | Accepted input sizes | One training step on a real 256 crop (batch 1) | Native prediction (168×299 UIEB image) |
|---|---|---|---|---|---|
| `ushape` | 31,589,775 | 31,391,203 | **256×256 only** | finite; 485/495 tensors receive gradients | OK via `resize256` |
| `fiveaplus` | 8,990 | 8,923 | any H, W | finite; 54/57 | OK |
| `spectroformer` | 2,511,081 | 2,246,411 | **multiples of 8** | finite; 431/511 | OK (padded to ×8) |

"Used in forward" excludes parameters of modules that are declared but never called in `forward()`. These are dead code in the released implementations (Spectroformer: e.g. the 4th SFCA, `refinement`, `ups1`/`ups2`/`ups_4`; U-shape: `feature_to_rgb`, `Conv`, `linear_encoding`).

**Environment:**
- All model forward dependencies are present: torch 2.7.1, einops 0.7.0, timm 1.0.3, scikit-image 0.26, thop, kornia.
- Missing, but not needed for our training loop: `pytorch_lightning` (Five A⁺ `train.py`) and `ptflops` (its runtime script).
- The pinned `requirements.txt` of U-shape (torch 1.8.1, timm 0.4.12, einops 0.3.0) was **not** installed. The model runs on the current stack.

## Protocol alignment (standardized split and metrics)
- **Data:** train 4,135 (3,423 LSUI + 712 UIEB) and val 517 (428 + 89), with counts asserted. The test manifest is refused.
- **Training input:** `train.PairedCropDataset`, the same as SS-PGN: 256 random crops; reflect-pad plus mask for small images; rot90 and flip; RGB in [0, 1].
- **Evaluation:** native resolution, uint8-rounded outputs, per-image PSNR₈/SSIM₈ and paired bootstrap, using the same functions as S1/P1/P1b.

## Open decisions (need PI input before any baseline training)
1. **U-shape input size.**
   - The architecture fixes 256×256 (learned positional encoding over 16×16 patches).
   - Its official test also resizes the *ground truth* to 256, so its published PSNR is not native-resolution.
   - The adapter currently uses `resize256`: resize the input to 256, then resize the output back and score at native resolution. This penalises U-shape relative to fully convolutional models.
   - Alternatives: overlapping 256-tiling (our T1 showed tiling costs SS-PGN about −1.33 dB on UIEB), or interpolating the positional encoding (modifies the model).
2. **Spectroformer architecture.**
   - `mymodel` is defined in 5 files.
   - `Train.py` imports a module (`final_model`) that does not exist.
   - The released `best.pth` pickles `model_without_CA.mymodel`, an ablation. This was established by reading the pickle's class references without unpickling.
   - The adapter defaults to `Final_model_AGSSF.mymodel`, assumed to be the full model. This should be confirmed against the paper.
3. **Training recipe per baseline.**
   - Options: (a) each baseline's official loss, optimizer and schedule, with our data, crops and step budget; or (b) a common recipe.
   - The official recipes differ widely:
     - U-shape: GAN + multi-colour-space loss, batch 1, 800 epochs;
     - Five A⁺: Adam 4e-4, CyclicLR, 400 epochs;
     - Spectroformer: GAN + VGG + MS-SSIM, lr 3e-5, batch 1.
   - The budget (≈ 0.37 GPU-days for each 100k-step SS-PGN run) must be set before launching.
4. **Pretrained weights are not used.** They were trained on author splits that may overlap our validation/test images.
5. **Five A⁺ has no licence file.** Using it to produce comparison numbers is normal research practice; redistributing its code or weights would need permission.
