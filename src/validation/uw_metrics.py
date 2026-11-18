"""
No-reference underwater metrics: UIQM and UCIQE.

These are vectorised ports of the two most widely reused public implementations. Published
UIQM/UCIQE numbers come from inconsistent implementations, so every method in a comparison
must be scored with THIS code; never mix with numbers copied from papers.

UIQM  (Panetta et al., IEEE JOE 2016): port of FUnIE-GAN Evaluation/uqim_utils.py
      (Islam et al. 2020, MIT licence), written for Python 2. Port decisions:
        * block counts k1 = W // 10, k2 = H // 10 (Python-2 integer division, which the original relied on);
        * alpha-trimmed mean keeps the original slice x[T_L + 1 : K - T_R];
        * UISM blue weight: the original has 0.144 (typo; the paper's luminance weight is 0.114).
          variant="funie" (default) reproduces 0.144 for comparability with papers that
          used this code; variant="paper" uses 0.114.
UCIQE (Yang & Sowmya, IEEE TIP 2015): port of JOU-UIP/UCIQE UCIQE.py, the authors' group repository.
        * OpenCV 8-bit Lab (L*255/100, a+128, b+128) divided by 255;
        * "var_chr" = sqrt(mean|1 - (mean_C / C)^2|), as in the reference (not a plain std);
        * luminance contrast from a 65536-bin histogram of L over [min, max], first index with
          cdf > 0.01 and cdf >= 0.99, (idx - 1) / 65535, as in the reference.

Input for both: HxWx3 uint8 RGB.

Self-check against the reference code (downloaded separately):
  python src/validation/uw_metrics.py --ref-uiqm <dir with uqim_utils.py> --ref-uciqe <dir with UCIQE.py> img1 img2 ...
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy import ndimage


# --------------------------------------------------------------------------
# UIQM
# --------------------------------------------------------------------------
def _mu_a(x, aL=0.1, aR=0.1):
    x = np.sort(x.astype(np.float64))
    K = x.size
    tl, tr = math.ceil(aL * K), math.floor(aR * K)
    return float(x[int(tl + 1):int(K - tr)].sum() / (K - tl - tr))


def _uicm(x):
    R, G, B = (x[..., i].ravel().astype(np.float64) for i in range(3))
    RG, YB = R - G, (R + G) / 2 - B
    m_rg, m_yb = _mu_a(RG), _mu_a(YB)
    s_rg, s_yb = np.mean((RG - m_rg) ** 2), np.mean((YB - m_yb) ** 2)
    return -0.0268 * math.sqrt(m_rg ** 2 + m_yb ** 2) + 0.1586 * math.sqrt(s_rg + s_yb)


def _sobel(c):
    mag = np.hypot(ndimage.sobel(c, 0), ndimage.sobel(c, 1))
    mx = np.max(mag)
    return mag * (255.0 / mx) if mx > 0 else mag


def _blocks(x, ws):
    k1, k2 = x.shape[1] // ws, x.shape[0] // ws
    if k1 == 0 or k2 == 0:
        raise ValueError(f"image {x.shape[:2]} smaller than block size {ws}")
    x = x[:k2 * ws, :k1 * ws]
    if x.ndim == 2:
        b = x.reshape(k2, ws, k1, ws).transpose(0, 2, 1, 3).reshape(k2, k1, -1)
    else:
        b = x.reshape(k2, ws, k1, ws, x.shape[2]).transpose(0, 2, 1, 3, 4).reshape(k2, k1, -1)
    return b.max(-1).astype(np.float64), b.min(-1).astype(np.float64), k1, k2


def _eme(x, ws):
    mx, mn, k1, k2 = _blocks(x, ws)
    ok = (mn != 0) & (mx != 0)
    val = np.log(mx[ok] / mn[ok]).sum()
    return 2.0 / (k1 * k2) * val


def _uism(x, variant):
    lam = (0.299, 0.587, 0.144 if variant == "funie" else 0.114)
    return sum(l * _eme(_sobel(x[..., i]) * x[..., i], 10) for i, l in enumerate(lam))


def _uiconm(x, ws=10):
    mx, mn, k1, k2 = _blocks(x, ws)
    top, bot = mx - mn, mx + mn
    ok = (top != 0) & (bot != 0) & ~np.isnan(top) & ~np.isnan(bot)
    r = top[ok] / bot[ok]
    return -1.0 / (k1 * k2) * float((r * np.log(r)).sum())


def uiqm(img_u8: np.ndarray, variant: str = "funie") -> dict:
    if img_u8.dtype != np.uint8 or img_u8.ndim != 3 or img_u8.shape[2] != 3:
        raise ValueError(f"uiqm expects HxWx3 uint8 RGB, got {img_u8.dtype} {img_u8.shape}")
    x = img_u8.astype(np.float32)
    a, b, c = _uicm(x), _uism(x, variant), _uiconm(x, 10)
    return {"uiqm": 0.0282 * a + 0.2953 * b + 3.5753 * c, "uicm": a, "uism": b, "uiconm": c}


# --------------------------------------------------------------------------
# UCIQE
# --------------------------------------------------------------------------
def uciqe(img_u8: np.ndarray) -> dict:
    if img_u8.dtype != np.uint8 or img_u8.ndim != 3 or img_u8.shape[2] != 3:
        raise ValueError(f"uciqe expects HxWx3 uint8 RGB, got {img_u8.dtype} {img_u8.shape}")
    lab = cv2.cvtColor(np.ascontiguousarray(img_u8[..., ::-1]), cv2.COLOR_BGR2LAB)
    lum, a, b = lab[..., 0] / 255, lab[..., 1] / 255, lab[..., 2] / 255
    chr_ = np.sqrt(a ** 2 + b ** 2)
    sat = chr_ / np.sqrt(chr_ ** 2 + lum ** 2)
    aver_sat, aver_chr = float(np.mean(sat)), float(np.mean(chr_))
    var_chr = float(np.sqrt(np.mean(np.abs(1 - (aver_chr / chr_) ** 2))))
    nbins = 65536                                  # reference: lum is float, so 65536 bins
    hist, _ = np.histogram(lum, nbins)
    cdf = np.cumsum(hist) / np.sum(hist)
    lo, hi = np.where(cdf > 0.01)[0][0], np.where(cdf >= 0.99)[0][0]
    con_lum = (hi - 1) / (nbins - 1) - (lo - 1) / (nbins - 1)
    return {"uciqe": 0.4680 * var_chr + 0.2745 * con_lum + 0.2576 * aver_sat,
            "uciqe_var_chr": var_chr, "uciqe_con_lum": con_lum, "uciqe_aver_sat": aver_sat}


# --------------------------------------------------------------------------
# Self-check against reference implementations
# --------------------------------------------------------------------------
def _check_against_refs(ref_uiqm_dir, ref_uciqe_dir, images, crop=160):
    import importlib.util
    import sys
    from PIL import Image

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    ok = True
    if ref_uiqm_dir:
        ref = load("ref_uqim", f"{ref_uiqm_dir}/uqim_utils.py")
        # Python-3 shim for the reference's float block counts (int division, as in Python 2)
        _orig_eme, _orig_uiconm = ref.eme, ref._uiconm

        def eme_py3(x, window_size):
            k1, k2 = x.shape[1] // window_size, x.shape[0] // window_size
            w = 2. / (k1 * k2); x = x[:window_size * k2, :window_size * k1]; val = 0
            for l in range(k1):
                for k in range(k2):
                    blk = x[k * window_size:window_size * (k + 1), l * window_size:window_size * (l + 1)]
                    mx, mn = np.max(blk), np.min(blk)
                    if mn == 0.0 or mx == 0.0: val += 0
                    else: val += math.log(mx / mn)
            return w * val

        def uiconm_py3(x, window_size):
            k1, k2 = x.shape[1] // window_size, x.shape[0] // window_size
            w = -1. / (k1 * k2); x = x[:window_size * k2, :window_size * k1]; val = 0
            for l in range(k1):
                for k in range(k2):
                    blk = x[k * window_size:window_size * (k + 1), l * window_size:window_size * (l + 1), :]
                    mx, mn = np.max(blk), np.min(blk); top, bot = mx - mn, mx + mn
                    if math.isnan(top) or math.isnan(bot) or bot == 0.0 or top == 0.0: val += 0.0
                    else: val += math.pow(top / bot, 1) * math.log(top / bot)
            return w * val
        ref.eme, ref._uiconm = eme_py3, uiconm_py3
        for p in images:
            with Image.open(p) as im:
                a = np.array(im.convert("RGB"))[:crop, :crop]     # crop: reference is pure-Python slow
            r, o = ref.getUIQM(a), uiqm(a, "funie")["uiqm"]
            good = abs(r - o) < 1e-4 * max(1, abs(r))
            ok &= good
            print(f"  UIQM  {p.split('/')[-1][:30]:30s} ref={r:.6f} ours={o:.6f} {'OK' if good else 'MISMATCH'}")
    if ref_uciqe_dir:
        ref = load("ref_uciqe", f"{ref_uciqe_dir}/UCIQE.py")
        for p in images:
            with Image.open(p) as im:
                a = np.array(im.convert("RGB"))
            r, o = ref.uciqe(1, p), uciqe(a)["uciqe"]
            good = abs(r - o) < 1e-9 * max(1, abs(r))
            ok &= good
            print(f"  UCIQE {p.split('/')[-1][:30]:30s} ref={r:.8f} ours={o:.8f} {'OK' if good else 'MISMATCH'}")
    return ok


if __name__ == "__main__":
    import argparse
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-uiqm", default=None)
    ap.add_argument("--ref-uciqe", default=None)
    ap.add_argument("images", nargs="+")
    a = ap.parse_args()
    sys.exit(0 if _check_against_refs(a.ref_uiqm, a.ref_uciqe, a.images) else 1)
