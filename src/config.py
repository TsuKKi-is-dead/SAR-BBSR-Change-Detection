from pathlib import Path

ROOT           = Path(__file__).resolve().parent.parent

# Directories
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_OUTPUT    = ROOT / "data" / "output"
RESULTS        = ROOT / "results"

for _d in (DATA_RAW, DATA_PROCESSED, DATA_OUTPUT, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)

# CDSE endpoints
CDSE_AUTH_URL      = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CDSE_CATALOG_URL   = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
CDSE_DOWNLOAD_BASE = "https://zipper.dataspace.copernicus.eu/odata/v1/Products"

# AOI — Bhubaneswar, Odisha
AOI_NAME = "Bhubaneswar_Odisha"
AOI_BBOX = (85.75, 20.20, 85.95, 20.35)
AOI_WKT  = "POLYGON((85.75 20.20,85.95 20.20,85.95 20.35,85.75 20.35,85.75 20.20))"

# Sentinel-1 product parameters
PLATFORM     = "Sentinel-1"
PRODUCT_TYPE = "SLC"
SENSOR_MODE  = "IW"
SUBSWATH     = "IW2"
POLARISATION = "VV"
BURST_INDEX  = 0

# Radar constants (C-band)
SPEED_OF_LIGHT      = 3.0e8
WAVELENGTH          = 0.05546576
SAT_VELOCITY        = 7200.0
RANGE_SAMPLING_RATE = 64.345e6
PRF_DEFAULT         = 486.486
CHIRP_BANDWIDTH     = 56.5e6
CHIRP_DURATION      = 52.0e-6

# Processing
RANGE_WINDOW     = "hamming"
AZIMUTH_WINDOW   = "hamming"
RCMC_INTERP      = "sinc"
SINC_HALF_WIDTH  = 8
DISPLAY_LOW_PCT  = 2
DISPLAY_HIGH_PCT = 98

# Output files
OUT_DETECTED_PNG     = RESULTS / "bbsr_sar_detected.png"
OUT_GEOTIFF          = RESULTS / "bbsr_sar.tif"
OUT_CHANGE_PNG       = RESULTS / "bbsr_change_map.png"
OUT_PHASE_PNG        = RESULTS / "bbsr_phase.png"
NP_RANGE_COMPRESSED  = DATA_PROCESSED / "range_compressed.npy"
NP_RCMC              = DATA_PROCESSED / "rcmc.npy"
NP_FOCUSED           = DATA_PROCESSED / "focused.npy"

# Date range for download (YYYYMMDD) — change these to target different passes
DATE_START = "20240201"
DATE_END   = "20240229"
