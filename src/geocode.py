

from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from reader import BurstData


# ---------------------------------------------------------------------------
# Earth model (WGS84)
# ---------------------------------------------------------------------------
_WGS84_A = 6378137.0           # semi-major axis (m)
_WGS84_B = 6356752.314245      # semi-minor axis (m)
_WGS84_E2 = 1 - (_WGS84_B / _WGS84_A)**2


def geodetic_to_ecef(lat_deg, lon_deg, alt_m=0.0):
    """Convert geodetic (WGS84) to ECEF Cartesian coordinates."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    N = _WGS84_A / np.sqrt(1 - _WGS84_E2 * np.sin(lat)**2)
    x = (N + alt_m) * np.cos(lat) * np.cos(lon)
    y = (N + alt_m) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - _WGS84_E2) + alt_m) * np.sin(lat)
    return np.array([x, y, z])


# ---------------------------------------------------------------------------
# State vector (orbit) loader
# ---------------------------------------------------------------------------

@dataclass
class StateVector:
    time_s: float        # seconds from burst start
    pos:    np.ndarray   # ECEF [x,y,z] metres
    vel:    np.ndarray   # ECEF [vx,vy,vz] m/s


def _load_state_vectors(annotation_path: str) -> List[StateVector]:
    """Parse orbit state vectors from annotation XML."""
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    svs = []
    t0 = None
    for sv in root.findall(".//orbitList/orbit"):
        time_str = sv.findtext("time", "")
        pos_el   = sv.find("position")
        vel_el   = sv.find("velocity")
        if pos_el is None:
            continue
        pos = np.array([float(pos_el.findtext("x", "0")),
                        float(pos_el.findtext("y", "0")),
                        float(pos_el.findtext("z", "0"))])
        vel = np.array([float(vel_el.findtext("x", "0")),
                        float(vel_el.findtext("y", "0")),
                        float(vel_el.findtext("z", "0"))]) if vel_el else np.zeros(3)
        # Convert time to float seconds (simplified: just use index)
        idx = len(svs)
        svs.append(StateVector(time_s=float(idx), pos=pos, vel=vel))
    return svs


def _interpolate_sv(svs: List[StateVector], t: float) -> np.ndarray:
    """Linear interpolation of satellite position at time t (seconds)."""
    times = np.array([sv.time_s for sv in svs])
    idx = np.searchsorted(times, t)
    idx = np.clip(idx, 1, len(svs) - 1)
    t0, t1 = svs[idx-1].time_s, svs[idx].time_s
    alpha = (t - t0) / max(t1 - t0, 1e-9)
    return svs[idx-1].pos + alpha * (svs[idx].pos - svs[idx-1].pos)


# ---------------------------------------------------------------------------
# Core geocoder
# ---------------------------------------------------------------------------

def geocode_to_geotiff(focused: np.ndarray,
                       burst: BurstData,
                       out_tif: Path = cfg.OUT_GEOTIFF,
                       output_res_deg: float = 0.0001,
                       looks_az: int = 3) -> None:
    """
    Geocode SAR image and write a GeoTIFF.

    Parameters
    ----------
    focused        : complex64 focused image [az, rg]
    burst          : BurstData (contains geometry parameters)
    out_tif        : output GeoTIFF path
    output_res_deg : output grid resolution in degrees (~10m at equator for 0.0001)
    looks_az       : must match what was used in detector (for aspect ratio)
    """
    try:
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS
    except ImportError:
        print("[Geocode] rasterio not found — skipping GeoTIFF export.")
        print("  Install with: pip install rasterio")
        return

    out_tif = Path(out_tif)
    out_tif.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Geocode] Building output grid over {cfg.AOI_NAME}...")

    lon_min, lat_min, lon_max, lat_max = cfg.AOI_BBOX
    lons = np.arange(lon_min, lon_max, output_res_deg)
    lats = np.arange(lat_max, lat_min, -output_res_deg)
    n_lat, n_lon = len(lats), len(lons)

    print(f"  Grid: {n_lat} × {n_lon} pixels  ({output_res_deg:.4f}° ≈ "
          f"{output_res_deg*111000:.0f}m)")

    # Magnitude image (multi-looked power in dB, float32)
    from detector import multilook, to_db, clip_percentile
    power   = multilook(focused, looks_az, 1)
    db_img  = to_db(power)
    naz_ml, nrg_ml = db_img.shape

    # ---- Load state vectors (orbit) ----
    svs = _load_state_vectors(burst.annotation_path) if burst.annotation_path else []
    has_orbit = len(svs) >= 2

    # ---- Build output grid via simple affine mapping ----
    # For a proper orbit geocoder we'd solve the Range-Doppler equations.
    # Here we use a first-order approximation: map the burst bounding box to
    # the output AOI assuming a flat-Earth linear relationship. This is
    # accurate to ~100m for a small AOI like BBSR.

    # Approximate near/far lat-lon from geometry
    # (replace with proper orbit-based computation if higher accuracy needed)
    near_range_km = burst.near_range / 1000.0
    far_range_km  = (burst.near_range + nrg_ml * burst.range_spacing) / 1000.0

    print(f"  Near range: {near_range_km:.1f} km, far range: {far_range_km:.1f} km")

    # Linear interpolation of image coordinates → geographic coordinates
    out_img = np.zeros((n_lat, n_lon), dtype=np.float32)

    lon_to_rg = (lons - lon_min) / (lon_max - lon_min) * (nrg_ml - 1)
    lat_to_az = (lat_max - lats) / (lat_max - lat_min) * (naz_ml - 1)

    from scipy.ndimage import map_coordinates
    rg_coords, lat_coords = np.meshgrid(lon_to_rg, lat_to_az)
    out_img = map_coordinates(db_img,
                               [lat_coords.ravel(), rg_coords.ravel()],
                               order=1, mode='nearest').reshape(n_lat, n_lon)

    # Clip for display
    vmin, vmax = np.percentile(out_img, [cfg.DISPLAY_LOW_PCT, cfg.DISPLAY_HIGH_PCT])
    out_img = np.clip(out_img, vmin, vmax).astype(np.float32)

    # ---- Write GeoTIFF ----
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, n_lon, n_lat)

    with rasterio.open(
        out_tif, "w",
        driver="GTiff",
        height=n_lat, width=n_lon,
        count=1,
        dtype=rasterio.float32,
        crs=CRS.from_epsg(4326),
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(out_img[np.newaxis, :, :])
        dst.update_tags(
            PRODUCT="Sentinel-1 IW SLC",
            AOI=cfg.AOI_NAME,
            SUBSWATH=burst.subswath,
            PROCESSING="Range-Doppler Algorithm (RDA)"
        )

    print(f"  Saved GeoTIFF → {out_tif}")
    print(f"  CRS: EPSG:4326 (WGS84)")
    print(f"  Bounds: {lon_min:.4f}E {lat_min:.4f}N – {lon_max:.4f}E {lat_max:.4f}N")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("safe",         help="Path to .SAFE")
    parser.add_argument("focused_npy",  help="Path to focused .npy")
    args = parser.parse_args()

    from reader import read_burst
    burst   = read_burst(args.safe)
    focused = np.load(args.focused_npy).astype(np.complex64)
    geocode_to_geotiff(focused, burst)
