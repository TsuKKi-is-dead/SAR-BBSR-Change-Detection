

from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from reader import BurstData


# ---------------------------------------------------------------------------
# Sinc kernel interpolation (used when RCMC_INTERP == 'sinc')
# ---------------------------------------------------------------------------

def _sinc_interp_row(row: np.ndarray, shift: np.ndarray,
                     half_width: int = cfg.SINC_HALF_WIDTH) -> np.ndarray:
    """
    Shift each sample in a row by a (possibly different) fractional amount
    using a windowed-sinc kernel.

    Parameters
    ----------
    row        : 1D complex array (one Doppler-frequency line)
    shift      : 1D real array of shifts in samples, shape == row.shape
    half_width : sinc kernel half-width (larger = more accurate, slower)

    Returns
    -------
    out : 1D complex array, same length as row
    """
    n = len(row)
    out = np.zeros(n, dtype=np.complex64)

    for j in range(n):
        src = j - shift[j]        # source location in original row
        k0  = int(np.floor(src)) - half_width + 1
        k1  = k0 + 2 * half_width

        # Clamp to valid indices
        ks = np.arange(k0, k1)
        valid = (ks >= 0) & (ks < n)
        ks_v = ks[valid]

        # Windowed sinc weights
        dx  = src - ks_v
        w   = np.sinc(dx) * np.blackman(2 * half_width)[valid]
        out[j] = np.dot(row[ks_v], w)

    return out


# ---------------------------------------------------------------------------
# Core RCMC
# ---------------------------------------------------------------------------

def rcmc(range_compressed: np.ndarray,
         burst: BurstData,
         interp: str = cfg.RCMC_INTERP) -> np.ndarray:
    """
    Apply Range Cell Migration Correction.

    Parameters
    ----------
    range_compressed : complex64 [az_lines, rg_samples]  (after range compression)
    burst            : BurstData (for PRF, wavelength, sat velocity, range vec)
    interp           : 'sinc' or 'linear' (map_coordinates order=1)

    Returns
    -------
    rcmc_data : complex64 [az_lines, rg_samples]  in Range-Doppler domain
                (azimuth FFT applied; ready for azimuth compression)
    """
    naz, nrg = range_compressed.shape
    print(f"[RCMC] {naz}×{nrg}  interp={interp}")

    # ---- Step 1: Azimuth FFT → Range-Doppler domain ----
    rdc = np.fft.fft(range_compressed, axis=0).astype(np.complex64)

    # Doppler frequency axis (centred on Doppler centroid)
    f_az = np.fft.fftfreq(naz, d=1.0 / burst.prf)   # [naz]
    f_az += burst.doppler_centroid                    # shift to centroid

    # Range (slant distance) to each sample [nrg]
    r_vec = burst.slant_range_vec   # metres

    # ---- Step 2: Compute range migration ΔR for each (f_az, r) ----
    #
    #  ΔR(f_a, r) = r * [1/sqrt(1 - (f_a*λ/(2v))²) - 1]
    #
    # Shape: [naz, nrg]
    ratio = (f_az[:, None] * burst.wavelength / (2 * cfg.SAT_VELOCITY))**2
    ratio = np.clip(ratio, 0, 0.99)   # avoid sqrt of negative
    scale = 1.0 / np.sqrt(1.0 - ratio) - 1.0
    delta_r = r_vec[None, :] * scale   # metres, shape [naz, nrg]

    # Convert ΔR (metres) → range samples
    delta_samp = delta_r / burst.range_spacing   # shape [naz, nrg]

    print(f"  Max migration: {np.max(np.abs(delta_r)):.1f} m  "
          f"({np.max(np.abs(delta_samp)):.1f} samples)")

    # ---- Step 3: Interpolate to correct range bins ----
    rcmc_out = np.zeros_like(rdc)

    if interp == "sinc":
        # Per-row sinc interpolation (slower but accurate)
        for i in range(naz):
            if i % 500 == 0:
                print(f"  RCMC row {i}/{naz} ...", end="\r")
            rcmc_out[i] = _sinc_interp_row(rdc[i], delta_samp[i])
    else:
        # scipy map_coordinates — linear interpolation (faster)
        rg_idx = np.arange(nrg)[None, :] - delta_samp   # [naz, nrg]
        az_idx = np.arange(naz)[:, None] * np.ones((1, nrg))

        # Real and imaginary parts separately (map_coordinates is real-only)
        coords = np.array([az_idx.ravel(), rg_idx.ravel()])
        r_part = map_coordinates(rdc.real, coords, order=1,
                                 mode='constant', cval=0).reshape(naz, nrg)
        i_part = map_coordinates(rdc.imag, coords, order=1,
                                 mode='constant', cval=0).reshape(naz, nrg)
        rcmc_out = (r_part + 1j * i_part).astype(np.complex64)

    print(f"\n  RCMC complete.")
    return rcmc_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, time
    from reader import read_burst

    parser = argparse.ArgumentParser()
    parser.add_argument("safe")
    parser.add_argument("--rc",   default=str(cfg.NP_RANGE_COMPRESSED),
                        help="Path to range-compressed .npy (skip re-computing)")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    burst = read_burst(args.safe)
    rc    = np.load(args.rc) if Path(args.rc).exists() else None

    if rc is None:
        from range_compress import range_compress
        rc = range_compress(burst)

    t0 = time.time()
    out = rcmc(rc, burst)
    print(f"  Time: {time.time()-t0:.1f}s")

    if args.save:
        np.save(cfg.NP_RCMC, out)
        print(f"  Saved → {cfg.NP_RCMC}")
