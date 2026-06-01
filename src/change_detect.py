

from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg


# ---------------------------------------------------------------------------
# Coherence estimation
# ---------------------------------------------------------------------------

def estimate_coherence(img1: np.ndarray, img2: np.ndarray,
                        window_az: int = 5, window_rg: int = 20) -> np.ndarray:
    """
    Compute complex coherence magnitude between two co-registered SLC images.

    Parameters
    ----------
    img1, img2   : complex64 arrays of the same shape [az, rg]
    window_az    : averaging window in azimuth (samples)
    window_rg    : averaging window in range (samples)

    Returns
    -------
    coherence : float32 [az, rg]  values in [0, 1]
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")

    print(f"[Coherence] window {window_az}az × {window_rg}rg  "
          f"on shape {img1.shape}")

    # Cross-product and individual powers
    cross = img1 * np.conj(img2)   # complex
    pow1  = np.abs(img1)**2
    pow2  = np.abs(img2)**2

    # Spatial averaging (uniform filter over each dimension)
    sz = (window_az, window_rg)
    cross_avg = (uniform_filter(cross.real, size=sz)
                 + 1j * uniform_filter(cross.imag, size=sz)).astype(np.complex64)
    pow1_avg  = uniform_filter(pow1, size=sz).astype(np.float32)
    pow2_avg  = uniform_filter(pow2, size=sz).astype(np.float32)

    # Coherence magnitude
    denom = np.sqrt(pow1_avg * pow2_avg) + 1e-10
    coherence = (np.abs(cross_avg) / denom).astype(np.float32)
    coherence = np.clip(coherence, 0.0, 1.0)

    print(f"  Mean coherence : {coherence.mean():.3f}")
    print(f"  Std coherence  : {coherence.std():.3f}")
    return coherence


# ---------------------------------------------------------------------------
# Change map
# ---------------------------------------------------------------------------

def make_change_map(coherence: np.ndarray,
                    threshold: float = 0.3,
                    out_png: Path = cfg.OUT_CHANGE_PNG) -> np.ndarray:
    """
    Generate and save a change detection map.

    Pixels below `threshold` are classified as 'changed' (red).
    Pixels above are 'stable' (grey).

    Parameters
    ----------
    coherence : float32 [az, rg]
    threshold : coherence threshold for change classification
    out_png   : output path

    Returns
    -------
    change_mask : bool [az, rg]  True = changed pixel
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    change_mask = coherence < threshold
    pct_changed = 100.0 * change_mask.sum() / change_mask.size
    print(f"[Change map] threshold={threshold}  changed={pct_changed:.1f}%")

    # ---- Visualisation ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="black")
    fig.suptitle(f"Coherence change detection — {cfg.AOI_NAME}",
                 color="white", fontsize=12, y=1.02)

    # Coherence image
    im0 = axes[0].imshow(coherence, cmap="RdYlGn", vmin=0, vmax=1,
                          interpolation="nearest", aspect="auto")
    axes[0].set_title("Coherence γ", color="white")
    plt.colorbar(im0, ax=axes[0]).set_label("γ", color="white")

    # Histogram of coherence
    axes[1].hist(coherence.ravel(), bins=100, color="#4DA6FF", edgecolor="none")
    axes[1].axvline(threshold, color="red", linewidth=1.5, label=f"threshold={threshold}")
    axes[1].set_title("Coherence histogram", color="white")
    axes[1].set_xlabel("γ", color="white")
    axes[1].set_ylabel("count", color="white")
    axes[1].tick_params(colors="white")
    axes[1].legend(facecolor="#222", labelcolor="white")
    axes[1].set_facecolor("#111")

    # Change map: red = changed, grey = stable
    cmap_map = np.zeros((*coherence.shape, 3), dtype=np.uint8)
    cmap_map[~change_mask] = [120, 120, 120]   # stable → grey
    cmap_map[change_mask]  = [220, 50,  50]    # changed → red
    axes[2].imshow(cmap_map, interpolation="nearest", aspect="auto")
    axes[2].set_title(f"Change mask  (red = γ < {threshold})", color="white")

    for ax in axes:
        ax.tick_params(colors="white")
        ax.set_facecolor("black")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close()
    print(f"  Saved → {out_png}")

    return change_mask


# ---------------------------------------------------------------------------
# Coregistration (simplified)
# ---------------------------------------------------------------------------

def coregister(img1: np.ndarray, img2: np.ndarray,
                max_shift: int = 20) -> np.ndarray:
    """
    Coregister img2 to img1 using cross-correlation of magnitude images.
    Returns img2 shifted to best align with img1.

    For production, use SNAP or pyroSAR for sub-pixel coregistration.
    """
    from scipy.signal import fftconvolve
    from scipy.ndimage import shift as nd_shift

    mag1 = np.abs(img1).astype(np.float32)
    mag2 = np.abs(img2).astype(np.float32)

    # Crop to a manageable size for cross-correlation
    h, w = min(512, mag1.shape[0]), min(512, mag1.shape[1])
    m1 = mag1[:h, :w]
    m2 = mag2[:h, :w]

    # Normalised cross-correlation
    corr = fftconvolve(m1 - m1.mean(), m2[::-1, ::-1] - m2.mean(), mode='same')
    cy, cx = np.unravel_index(np.argmax(corr), corr.shape)
    dy = cy - h // 2
    dx = cx - w // 2

    # Clamp to max_shift
    dy = int(np.clip(dy, -max_shift, max_shift))
    dx = int(np.clip(dx, -max_shift, max_shift))

    print(f"[Coregister] best shift: az={dy}, rg={dx}")
    img2_shifted = nd_shift(img2.real, [dy, dx]) + 1j * nd_shift(img2.imag, [dy, dx])
    return img2_shifted.astype(np.complex64)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def run_change_detection(img1_path: str, img2_path: str,
                          coregister_images: bool = True,
                          window_az: int = 5, window_rg: int = 20,
                          threshold: float = 0.3) -> None:
    print(f"\n{'='*60}")
    print(f"Change detection: {Path(img1_path).name} vs {Path(img2_path).name}")
    print(f"{'='*60}")

    img1 = np.load(img1_path).astype(np.complex64)
    img2 = np.load(img2_path).astype(np.complex64)

    if img1.shape != img2.shape:
        # Crop to common size
        h = min(img1.shape[0], img2.shape[0])
        w = min(img1.shape[1], img2.shape[1])
        img1, img2 = img1[:h, :w], img2[:h, :w]
        print(f"  Cropped to common shape: {h}×{w}")

    if coregister_images:
        img2 = coregister(img1, img2)

    coh    = estimate_coherence(img1, img2, window_az, window_rg)
    change = make_change_map(coh, threshold)
    print(f"\nDone. Outputs in {cfg.RESULTS}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("img1", help="First focused image (.npy)")
    parser.add_argument("img2", help="Second focused image (.npy)")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--no-coreg", action="store_true")
    args = parser.parse_args()

    run_change_detection(
        args.img1, args.img2,
        coregister_images=not args.no_coreg,
        threshold=args.threshold
    )
