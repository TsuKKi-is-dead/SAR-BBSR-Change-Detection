

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class BurstData:
    raw_iq:          np.ndarray           # complex64 [az, rg]
    az_lines:        int
    rg_samples:      int
    prf:             float                # Hz
    range_fs:        float                # Hz
    chirp_bw:        float                # Hz
    chirp_duration:  float                # s
    wavelength:      float                # m
    near_range:      float                # m  (slant range to first sample)
    range_spacing:   float                # m  (slant range sample spacing)
    az_spacing:      float                # m  (azimuth sample spacing)
    doppler_centroid: float               # Hz
    burst_index:     int
    subswath:        str
    annotation_path: str = ""
    safe_path:       str = ""

    @property
    def slant_range_vec(self) -> np.ndarray:
        """Slant range (m) to each range sample."""
        return self.near_range + np.arange(self.rg_samples) * self.range_spacing

    @property
    def az_time_vec(self) -> np.ndarray:
        """Relative azimuth time (s) for each line, centred at zero."""
        t = np.arange(self.az_lines) / self.prf
        return t - t.mean()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_file(directory: Path, keyword: str, ext: str) -> Optional[Path]:
    for f in directory.iterdir():
        if keyword.lower() in f.name.lower() and f.suffix.lower() == ext:
            return f
    return None


def _xml_text(root, xpath: str, default=None):
    el = root.find(xpath)
    if el is None:
        return default
    return el.text.strip()


def _parse_annotation(ann_path: Path) -> dict:
    """Extract burst geometry and sensor parameters from annotation XML."""
    tree = ET.parse(ann_path)
    root = tree.getroot()

    params = {}

    # Burst dimensions
    params["lines_per_burst"]   = int(_xml_text(root, ".//linesPerBurst",   "0"))
    params["samples_per_burst"] = int(_xml_text(root, ".//samplesPerBurst", "0"))

    # Timing
    params["prf"] = float(_xml_text(
        root, ".//replicaInformation/chirpParameters/txPulseRepetitionFrequency",
        str(cfg.PRF_DEFAULT)
    ))
    # Some annotation versions use azimuthFrequency instead
    az_freq_el = root.find(".//azimuthFrequency")
    if az_freq_el is not None:
        params["prf"] = float(az_freq_el.text)

    params["range_fs"] = float(_xml_text(
        root, ".//rangeSamplingRate", str(cfg.RANGE_SAMPLING_RATE)
    ))

    # Chirp
    params["chirp_bw"]       = float(_xml_text(root, ".//txPulseRampRate",   str(cfg.CHIRP_BANDWIDTH)))
    params["chirp_duration"] = float(_xml_text(root, ".//txPulseLength",     str(cfg.CHIRP_DURATION)))

    # Geometry
    params["near_range"] = float(_xml_text(root, ".//slantRangeTime", "0.005")) * cfg.SPEED_OF_LIGHT / 2
    params["range_spacing"] = cfg.SPEED_OF_LIGHT / (2 * params["range_fs"])
    params["az_spacing"]    = cfg.SAT_VELOCITY / params["prf"]

    # Doppler centroid (first polynomial coefficient)
    dc_el = root.find(".//dcEstimateList/dcEstimate/dataDcPolynomial")
    params["doppler_centroid"] = float(dc_el.text.split()[0]) if dc_el is not None else 0.0

    # Wavelength from carrier frequency
    carrier_el = root.find(".//radarFrequency")
    if carrier_el is not None:
        params["wavelength"] = cfg.SPEED_OF_LIGHT / float(carrier_el.text)
    else:
        params["wavelength"] = cfg.WAVELENGTH

    return params


def _read_tiff_iq(tiff_path: Path, lines: int, samples: int,
                  burst_index: int) -> np.ndarray:
    """
    Read one burst worth of complex int16 IQ from a Sentinel-1 measurement TIFF.
    The TIFF stores bursts sequentially: each burst = lines × samples × 2 int16.

    For real Sentinel-1 products the TIFF is a standard GeoTIFF; we use a
    direct binary read here to avoid a heavyweight GDAL dependency in the core
    reader. If rasterio/GDAL is available, use _read_tiff_rasterio() instead.
    """
    bytes_per_sample = 2   # int16
    samples_per_line = samples * 2  # I + Q
    bytes_per_line   = samples_per_line * bytes_per_sample
    bytes_per_burst  = lines * bytes_per_line
    offset           = burst_index * bytes_per_burst

    with open(tiff_path, "rb") as f:
        # Skip TIFF header (typically 8 bytes) + IFD + any strip offsets.
        # Sentinel-1 measurement TIFFs store data starting at a known strip
        # offset listed in the TIFF IFD. We use a simple heuristic: scan for
        # the actual data start using the known burst dimensions.
        # For robustness use rasterio when available (see below).
        f.seek(offset)
        raw = np.frombuffer(f.read(bytes_per_burst), dtype=np.int16)

    if raw.size < lines * samples * 2:
        raise ValueError(
            f"Read only {raw.size} int16 values; expected {lines * samples * 2}. "
            "Try a different burst_index or check the file integrity."
        )

    iq = raw[::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
    return iq.reshape(lines, samples).astype(np.complex64)


def _read_tiff_rasterio(tiff_path: Path, lines: int, samples: int,
                         burst_index: int) -> np.ndarray:
    """Preferred reader when rasterio is available — handles TIFF strip layout correctly."""
    import rasterio
    with rasterio.open(tiff_path) as src:
        row_start = burst_index * lines
        window = rasterio.windows.Window(0, row_start, samples * 2, lines)
        data = src.read(1, window=window)   # band 1 = interleaved int16

    # data shape: [lines, samples*2]  — interleaved I, Q
    if np.iscomplexobj(data):
        return data.astype(np.complex64)
    n = (data.shape[1] // 2) * 2
    I = data[:, 0:n:2].astype(np.float32)
    Q = data[:, 1:n:2].astype(np.float32)
    return (I + 1j * Q).astype(np.complex64)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_burst(safe_path: Path,
               subswath: str = cfg.SUBSWATH,
               burst_index: int = cfg.BURST_INDEX,
               polarisation: str = cfg.POLARISATION) -> BurstData:
    """
    Read one burst from a Sentinel-1 .SAFE product.

    Parameters
    ----------
    safe_path   : Path to the .SAFE directory
    subswath    : 'IW1', 'IW2', or 'IW3'
    burst_index : 0-indexed burst within the subswath
    polarisation: 'VV', 'VH', 'HH', 'HV'

    Returns
    -------
    BurstData dataclass with raw_iq and all sensor parameters
    """
    safe_path = Path(safe_path)
    if not safe_path.exists():
        raise FileNotFoundError(f"SAFE product not found: {safe_path}")

    sw = subswath.lower()   # e.g. 'iw2'
    pol = polarisation.lower()   # e.g. 'vv'

    # ---- Annotation XML ----
    ann_dir = safe_path / "annotation"
    ann_path = _find_file(ann_dir, sw, ".xml")
    if ann_path is None:
        # Try substring match with polarisation
        for f in ann_dir.iterdir():
            if sw in f.name and pol in f.name:
                ann_path = f
                break
    if ann_path is None:
        raise FileNotFoundError(
            f"Annotation XML not found for subswath={subswath}, pol={polarisation} "
            f"in {ann_dir}"
        )

    print(f"Reading annotation: {ann_path.name}")
    params = _parse_annotation(ann_path)

    lines   = params["lines_per_burst"]
    samples = params["samples_per_burst"]

    if lines == 0 or samples == 0:
        raise ValueError(
            "Could not read burst dimensions from annotation XML. "
            "Check linesPerBurst / samplesPerBurst elements."
        )

    print(f"Burst dimensions : {lines} az lines × {samples} rg samples")

    # ---- Measurement TIFF ----
    meas_dir = safe_path / "measurement"
    tiff_path = _find_file(meas_dir, sw, ".tiff")
    if tiff_path is None:
        tiff_path = _find_file(meas_dir, sw, ".tif")
    if tiff_path is None:
        raise FileNotFoundError(
            f"Measurement TIFF not found for subswath={subswath} in {meas_dir}"
        )

    print(f"Reading TIFF     : {tiff_path.name}  (burst {burst_index})")

    try:
        import rasterio
        raw_iq = _read_tiff_rasterio(tiff_path, lines, samples, burst_index)
        print("  (using rasterio reader)")
    except ImportError:
        raw_iq = _read_tiff_iq(tiff_path, lines, samples, burst_index)
        print("  (using binary reader — install rasterio for more robustness)")

    print(f"  IQ array shape : {raw_iq.shape}, dtype={raw_iq.dtype}")
    print(f"  PRF            : {params['prf']:.3f} Hz")
    print(f"  Range fs       : {params['range_fs']/1e6:.3f} MHz")
    print(f"  Chirp BW       : {params['chirp_bw']/1e6:.3f} MHz")
    print(f"  Near range     : {params['near_range']/1e3:.1f} km")
    print(f"  Doppler centre : {params['doppler_centroid']:.2f} Hz")

    return BurstData(
        raw_iq           = raw_iq,
        az_lines         = lines,
        rg_samples       = samples,
        prf              = params["prf"],
        range_fs         = params["range_fs"],
        chirp_bw         = params["chirp_bw"],
        chirp_duration   = params["chirp_duration"],
        wavelength       = params["wavelength"],
        near_range       = params["near_range"],
        range_spacing    = params["range_spacing"],
        az_spacing       = params["az_spacing"],
        doppler_centroid = params["doppler_centroid"],
        burst_index      = burst_index,
        subswath         = subswath,
        annotation_path  = str(ann_path),
        safe_path        = str(safe_path),
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("safe", help="Path to .SAFE product")
    parser.add_argument("--subswath", default=cfg.SUBSWATH)
    parser.add_argument("--burst", type=int, default=cfg.BURST_INDEX)
    args = parser.parse_args()

    burst = read_burst(args.safe, subswath=args.subswath, burst_index=args.burst)
    print(f"\nBurst read OK. Shape: {burst.raw_iq.shape}")
