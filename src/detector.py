

from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for scripts
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def multilook(complex_img: np.ndarray,
              looks_az: int = 1, looks_rg: int = 1) -> np.ndarray:
    """
    Multi-look by averaging power over az×rg blocks.
    Reduces speckle; output is real float32 power.

    Crop to integer multiple of look factors.
    """
    if looks_az == 1 and looks_rg == 1:
        return np.abs(complex_img)**2

    naz, nrg = complex_img.shape
    naz_ml = (naz // looks_az) * looks_az
    nrg_ml = (nrg // looks_rg) * looks_rg
    img = complex_img[:naz_ml, :nrg_ml]

    power = np.abs(img)**2
    power = power.reshape(naz_ml // looks_az, looks_az,
                          nrg_ml // looks_rg, looks_rg)
    return power.mean(axis=(1, 3)).astype(np.float32)


def to_db(power: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Convert linear power to dB scale."""
    return (10.0 * np.log10(power + eps)).astype(np.float32)


def clip_percentile(img: np.ndarray,
                    low: float = cfg.DISPLAY_LOW_PCT,
                    high: float = cfg.DISPLAY_HIGH_PCT) -> np.ndarray:
    """Clip to [low, high] percentile and normalise to [0, 1]."""
    vmin, vmax = np.percentile(img, [low, high])
    return np.clip((img - vmin) / (vmax - vmin + 1e-8), 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Main detection + save
# ---------------------------------------------------------------------------

def detect_and_save(focused: np.ndarray,
                    out_png: Path = cfg.OUT_DETECTED_PNG,
                    looks_az: int = 3, looks_rg: int = 1,
                    save_phase: bool = True) -> np.ndarray:
    """
    Detect, multi-look, log-scale, and save visualisation.

    Parameters
    ----------
    focused   : complex64 focused image [az, rg]
    out_png   : output PNG path
    looks_az  : azimuth multi-look factor (default 3 → ~square pixels for IW)
    looks_rg  : range multi-look factor
    save_phase: also save phase image

    Returns
    -------
    display : float32 [az_ml, rg_ml] normalised to [0,1] for display
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Detect] multi-look {looks_az}az × {looks_rg}rg")
    power   = multilook(focused, looks_az, looks_rg)
    db_img  = to_db(power)
    display = clip_percentile(db_img)

    print(f"  Output shape  : {display.shape}")
    print(f"  dB range      : {db_img.min():.1f} – {db_img.max():.1f} dB")
    print(f"  Display range : [{np.percentile(db_img, cfg.DISPLAY_LOW_PCT):.1f}, "
          f"{np.percentile(db_img, cfg.DISPLAY_HIGH_PCT):.1f}] dB")

    # ---- Backscatter image ----
    fig, ax = plt.subplots(figsize=(12, 10), facecolor="black")
    ax.imshow(display, cmap="gray", interpolation="nearest", aspect="auto")
    ax.set_title(f"Sentinel-1 SAR — {cfg.AOI_NAME}\n"
                 f"(backscatter, {looks_az}×{looks_rg} looks, log scale)",
                 color="white", fontsize=11, pad=8)
    ax.set_xlabel("Range (samples)", color="white")
    ax.set_ylabel("Azimuth (samples)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    # Colourbar
    sm = plt.cm.ScalarMappable(cmap="gray",
                                norm=Normalize(vmin=np.percentile(db_img, cfg.DISPLAY_LOW_PCT),
                                               vmax=np.percentile(db_img, cfg.DISPLAY_HIGH_PCT)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Backscatter (dB)", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close()
    print(f"  Saved backscatter → {out_png}")

    # ---- Phase image ----
    if save_phase:
        phase = np.angle(focused)
        phase_norm = (phase + np.pi) / (2 * np.pi)   # [0,1]

        fig2, ax2 = plt.subplots(figsize=(12, 10), facecolor="black")
        ax2.imshow(phase_norm, cmap="hsv", interpolation="nearest", aspect="auto")
        ax2.set_title(f"Sentinel-1 — {cfg.AOI_NAME}  (phase)",
                      color="white", fontsize=11, pad=8)
        ax2.set_xlabel("Range (samples)", color="white")
        ax2.set_ylabel("Azimuth (samples)", color="white")
        ax2.tick_params(colors="white")
        plt.tight_layout()
        plt.savefig(cfg.OUT_PHASE_PNG, dpi=150, bbox_inches="tight",
                    facecolor="black", edgecolor="none")
        plt.close()
        print(f"  Saved phase      → {cfg.OUT_PHASE_PNG}")

    return display


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("focused_npy", help=f"Path to focused .npy array")
    parser.add_argument("--looks-az", type=int, default=3)
    parser.add_argument("--looks-rg", type=int, default=1)
    args = parser.parse_args()

    focused = np.load(args.focused_npy).astype(np.complex64)
    detect_and_save(focused, looks_az=args.looks_az, looks_rg=args.looks_rg)
