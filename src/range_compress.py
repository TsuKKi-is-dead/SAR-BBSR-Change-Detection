

from pathlib import Path
import sys

import numpy as np
from scipy.signal import get_window

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from reader import BurstData


# ---------------------------------------------------------------------------
# Reference chirp construction
# ---------------------------------------------------------------------------

def _reference_chirp_spectrum(nrg: int, chirp_bw: float,
                               chirp_duration: float, range_fs: float) -> np.ndarray:
    """
    Build the reference chirp spectrum H(f) for a linear FM pulse.

    H(f) = rect(f / B) * exp(-j*pi*f^2 / K_r)

    where K_r = B / tau is the chirp rate (Hz/s).

    Parameters
    ----------
    nrg           : number of range samples (FFT length)
    chirp_bw      : chirp bandwidth B (Hz)
    chirp_duration: pulse duration tau (s)
    range_fs      : range sampling frequency (Hz)

    Returns
    -------
    H : complex64 array of length nrg
    """
    K_r = chirp_bw / chirp_duration          # chirp rate
    f   = np.fft.fftfreq(nrg, d=1.0 / range_fs)  # frequency axis

    # Rectangular envelope (passband = chirp BW)
    rect = (np.abs(f) <= chirp_bw / 2).astype(np.float32)

    H = rect * np.exp(-1j * np.pi * f**2 / K_r)
    return H.astype(np.complex64)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def range_compress(burst: BurstData,
                   window: str = cfg.RANGE_WINDOW) -> np.ndarray:
    """
    Apply range compression to all azimuth lines of a burst.

    Parameters
    ----------
    burst  : BurstData from reader.py
    window : window function name (scipy.signal.get_window) or 'none'

    Returns
    -------
    rc : complex64 array [az_lines, rg_samples]
         Range-compressed data. Each target is now a sharp peak in range.
    """
    raw  = burst.raw_iq                      # [az, rg] complex64
    naz, nrg = raw.shape

    print(f"[Range compress] {naz} × {nrg}  chirp BW={burst.chirp_bw/1e6:.1f} MHz")

    # ---- Range window ----
    if window.lower() == "none":
        win = np.ones(nrg, dtype=np.float32)
    else:
        win = get_window(window, nrg).astype(np.float32)

    # ---- Reference chirp spectrum ----
    H_ref = _reference_chirp_spectrum(
        nrg, burst.chirp_bw, burst.chirp_duration, burst.range_fs
    )

    # ---- Process all azimuth lines (vectorised) ----
    # Apply window, FFT, multiply by H*(f), IFFT
    windowed = raw * win[np.newaxis, :]           # broadcast window over az
    S_f      = np.fft.fft(windowed, axis=1)       # FFT along range
    S_mf     = S_f * np.conj(H_ref)[np.newaxis, :]# matched filter
    rc       = np.fft.ifft(S_mf, axis=1).astype(np.complex64)

    print(f"  Done. Peak SNR gain ≈ {10*np.log10(nrg):.1f} dB (theoretical)")
    return rc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, time
    from reader import read_burst

    parser = argparse.ArgumentParser()
    parser.add_argument("safe", help="Path to .SAFE product")
    parser.add_argument("--save", action="store_true",
                        help=f"Save result to {cfg.NP_RANGE_COMPRESSED}")
    args = parser.parse_args()

    burst = read_burst(args.safe)
    t0 = time.time()
    rc = range_compress(burst)
    print(f"  Time: {time.time()-t0:.1f}s")

    if args.save:
        np.save(cfg.NP_RANGE_COMPRESSED, rc)
        print(f"  Saved → {cfg.NP_RANGE_COMPRESSED}")
