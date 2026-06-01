

from pathlib import Path
import sys

import numpy as np
from scipy.signal import get_window

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from reader import BurstData


# ---------------------------------------------------------------------------
# Azimuth reference function
# ---------------------------------------------------------------------------

def _az_reference_spectrum(naz: int, prf: float,
                             K_a_vec: np.ndarray,
                             doppler_centroid: float) -> np.ndarray:
    """
    Build the azimuth matched filter for all range bins.

    Parameters
    ----------
    naz              : number of azimuth samples (FFT length)
    prf              : pulse repetition frequency (Hz)
    K_a_vec          : azimuth FM rate for each range bin, shape [nrg]
    doppler_centroid : Doppler centroid frequency (Hz)

    Returns
    -------
    H_az : complex64 array [naz, nrg]
    """
    f_az = np.fft.fftfreq(naz, d=1.0 / prf)   # [naz]
    f_az += doppler_centroid                    # shift to centroid

    # Outer product: f_az² over range bins with different K_a
    phase = np.pi * f_az[:, None]**2 / K_a_vec[None, :]  # [naz, nrg]
    return np.exp(1j * phase).astype(np.complex64)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def azimuth_compress(rcmc_data: np.ndarray,
                     burst: BurstData,
                     window: str = cfg.AZIMUTH_WINDOW) -> np.ndarray:
    """
    Apply azimuth compression to RCMC data (Range-Doppler domain).

    Parameters
    ----------
    rcmc_data : complex64 [az_lines, rg_samples]  (Range-Doppler domain,
                output of rcmc.py — azimuth FFT already applied)
    burst     : BurstData
    window    : window function applied along azimuth before filtering

    Returns
    -------
    focused : complex64 [az_lines, rg_samples]
              Fully focused SAR image in slant-range geometry.
              Take np.abs() to get magnitude, angle() for phase.
    """
    naz, nrg = rcmc_data.shape
    print(f"[Azimuth compress] {naz}×{nrg}")

    # ---- Azimuth FM rate (range-dependent) ----
    #   K_a(r) = 2 * v² / (λ * r)
    r_vec = burst.slant_range_vec   # [nrg]
    K_a   = 2.0 * cfg.SAT_VELOCITY**2 / (burst.wavelength * r_vec)  # [nrg]

    print(f"  K_a range: {K_a.min():.0f} – {K_a.max():.0f} Hz/s")

    # ---- Azimuth window ----
    if window.lower() == "none":
        win = np.ones(naz, dtype=np.float32)
    else:
        win = get_window(window, naz).astype(np.float32)

    windowed = rcmc_data * win[:, np.newaxis]

    # ---- Azimuth matched filter ----
    H_az = _az_reference_spectrum(naz, burst.prf, K_a, burst.doppler_centroid)

    filtered = windowed * np.conj(H_az)

    # ---- Azimuth IFFT → image domain ----
    focused = np.fft.ifft(filtered, axis=0).astype(np.complex64)

    print(f"  Focused. Peak magnitude: {np.abs(focused).max():.2f}")
    return focused


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, time
    from reader import read_burst

    parser = argparse.ArgumentParser()
    parser.add_argument("safe")
    parser.add_argument("--rcmc",  default=str(cfg.NP_RCMC))
    parser.add_argument("--save",  action="store_true")
    args = parser.parse_args()

    burst = read_burst(args.safe)
    rcmc_data = np.load(args.rcmc) if Path(args.rcmc).exists() else None

    if rcmc_data is None:
        from range_compress import range_compress
        from rcmc import rcmc as do_rcmc
        rc = range_compress(burst)
        rcmc_data = do_rcmc(rc, burst)

    t0 = time.time()
    focused = azimuth_compress(rcmc_data, burst)
    print(f"  Time: {time.time()-t0:.1f}s")

    if args.save:
        np.save(cfg.NP_FOCUSED, focused)
        print(f"  Saved → {cfg.NP_FOCUSED}")
