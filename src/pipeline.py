

import argparse
import time
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg


# ---------------------------------------------------------------------------
# Timer helper
# ---------------------------------------------------------------------------
class Timer:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self._t = time.time()
        print(f"\n{'─'*60}")
        print(f"  STAGE: {self.name}")
        print(f"{'─'*60}")
        return self
    def __exit__(self, *_):
        print(f"  ✓ {self.name} done in {time.time()-self._t:.1f}s")


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def run_pipeline(safe_path: str,
                 safe_path2: str = None,
                 subswath: str = cfg.SUBSWATH,
                 burst_index: int = cfg.BURST_INDEX,
                 looks_az: int = 3,
                 skip_to: str = None,
                 export_geotiff: bool = True) -> None:

    safe_path = Path(safe_path)
    t_total = time.time()

    print(f"\n{'='*60}")
    print(f"  SAR PIPELINE — {cfg.AOI_NAME}")
    print(f"  SAFE   : {safe_path.name}")
    print(f"  SW/BU  : {subswath} / burst {burst_index}")
    print(f"  Output : {cfg.RESULTS}/")
    print(f"{'='*60}")

    SKIP_ORDER = ["range", "rcmc", "azimuth", "detect"]
    skip_idx = SKIP_ORDER.index(skip_to) if skip_to in SKIP_ORDER else -1

    # ---- Stage 0: Read burst ----
    from reader import read_burst
    burst = None

    if skip_idx < 0:
        with Timer("Read burst"):
            burst = read_burst(safe_path, subswath=subswath, burst_index=burst_index)

    # ---- Stage 1: Range compression ----
    rc = None
    if skip_idx < SKIP_ORDER.index("range"):
        with Timer("Range compression"):
            from range_compress import range_compress
            rc = range_compress(burst)
            np.save(cfg.NP_RANGE_COMPRESSED, rc)
            print(f"  Saved → {cfg.NP_RANGE_COMPRESSED}")
    elif skip_idx >= SKIP_ORDER.index("range"):
        print(f"\n[Skip] Loading range-compressed from {cfg.NP_RANGE_COMPRESSED}")
        rc = np.load(cfg.NP_RANGE_COMPRESSED)
        if burst is None:
            burst = read_burst(safe_path, subswath=subswath, burst_index=burst_index)

    # ---- Stage 2: RCMC ----
    rcmc_data = None
    if skip_idx < SKIP_ORDER.index("rcmc"):
        with Timer("RCMC"):
            from rcmc import rcmc
            rcmc_data = rcmc(rc, burst)
            np.save(cfg.NP_RCMC, rcmc_data)
            print(f"  Saved → {cfg.NP_RCMC}")
    elif skip_idx >= SKIP_ORDER.index("rcmc"):
        print(f"\n[Skip] Loading RCMC from {cfg.NP_RCMC}")
        rcmc_data = np.load(cfg.NP_RCMC)
        if burst is None:
            burst = read_burst(safe_path, subswath=subswath, burst_index=burst_index)

    # ---- Stage 3: Azimuth compression ----
    focused = None
    if skip_idx < SKIP_ORDER.index("azimuth"):
        with Timer("Azimuth compression"):
            from azimuth_compress import azimuth_compress
            focused = azimuth_compress(rcmc_data, burst)
            np.save(cfg.NP_FOCUSED, focused)
            print(f"  Saved → {cfg.NP_FOCUSED}")
    elif skip_idx >= SKIP_ORDER.index("azimuth"):
        print(f"\n[Skip] Loading focused from {cfg.NP_FOCUSED}")
        focused = np.load(cfg.NP_FOCUSED)
        if burst is None:
            burst = read_burst(safe_path, subswath=subswath, burst_index=burst_index)

    # ---- Stage 4: Detect + PNG ----
    with Timer("Detection + visualisation"):
        from detector import detect_and_save
        detect_and_save(focused, looks_az=looks_az)

    # ---- Stage 5: GeoTIFF ----
    if export_geotiff:
        with Timer("Geocoding → GeoTIFF"):
            from geocode import geocode_to_geotiff
            try:
                geocode_to_geotiff(focused, burst, looks_az=looks_az)
            except Exception as e:
                print(f"  [Warning] Geocoding failed: {e}")
                print("  Install rasterio to enable GeoTIFF export.")
    else:
        print("\n[Skip] GeoTIFF export disabled (--no-geotiff)")

    # ---- Stage 6 (optional): Change detection ----
    if safe_path2:
        print(f"\n{'='*60}")
        print(f"  CHANGE DETECTION: second acquisition")
        print(f"  SAFE2: {Path(safe_path2).name}")
        print(f"{'='*60}")

        with Timer("Process second acquisition"):
            from reader import read_burst as rb
            from range_compress import range_compress as rcomp
            from rcmc import rcmc as do_rcmc
            from azimuth_compress import azimuth_compress as az_comp

            b2       = rb(safe_path2, subswath=subswath, burst_index=burst_index)
            rc2      = rcomp(b2)
            rcmc2    = do_rcmc(rc2, b2)
            focused2 = az_comp(rcmc2, b2)
            np.save(cfg.DATA_PROCESSED / "focused2.npy", focused2)

        with Timer("Coherence change detection"):
            from change_detect import run_change_detection
            run_change_detection(
                str(cfg.NP_FOCUSED),
                str(cfg.DATA_PROCESSED / "focused2.npy")
            )

    # ---- Summary ----
    total_time = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE  ({total_time/60:.1f} min total)")
    print(f"{'='*60}")
    print(f"  Results:")
    for out in [cfg.OUT_DETECTED_PNG, cfg.OUT_PHASE_PNG,
                cfg.OUT_GEOTIFF, cfg.OUT_CHANGE_PNG]:
        if out.exists():
            size_kb = out.stat().st_size / 1024
            print(f"    {out.name:35s}  {size_kb:7.0f} KB")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end SAR pipeline for BBSR Odisha"
    )
    parser.add_argument("--safe",       required=True, help="Primary .SAFE path")
    parser.add_argument("--safe2",      default=None,  help="Second .SAFE for change detection")
    parser.add_argument("--subswath",   default=cfg.SUBSWATH)
    parser.add_argument("--burst",      type=int, default=cfg.BURST_INDEX)
    parser.add_argument("--looks-az",   type=int, default=3)
    parser.add_argument("--no-geotiff", action="store_true")
    parser.add_argument("--skip-to",    default=None,
                        choices=["range", "rcmc", "azimuth", "detect"],
                        help="Resume from this stage using saved intermediates")
    args = parser.parse_args()

    run_pipeline(
        safe_path      = args.safe,
        safe_path2     = args.safe2,
        subswath       = args.subswath,
        burst_index    = args.burst,
        looks_az       = args.looks_az,
        skip_to        = args.skip_to,
        export_geotiff = not args.no_geotiff,
    )
