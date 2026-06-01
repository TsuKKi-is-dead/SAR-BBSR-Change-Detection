# SAR Image Reconstruction — Bhubaneswar, Odisha

Sentinel-1 SAR processing pipeline using the Range-Doppler Algorithm (RDA).
Area of interest: Bhubaneswar (BBSR), Odisha, India.

## Project Structure

```
sar_bbsr/
├── data/
│   ├── raw/          ← Downloaded Sentinel-1 .SAFE products go here
│   ├── processed/    ← Intermediate numpy arrays (range-compressed, RCMC etc.)
│   └── output/       ← Final detected images, GeoTIFFs, change maps
├── src/
│   ├── config.py          ← AOI, paths, processing parameters
│   ├── downloader.py      ← Sentinel-1 data download via sentinelsat
│   ├── reader.py          ← Read raw IQ bursts from .SAFE format
│   ├── range_compress.py  ← Stage 1: Range matched filter
│   ├── rcmc.py            ← Stage 2: Range Cell Migration Correction
│   ├── azimuth_compress.py← Stage 3: Azimuth matched filter
│   ├── detector.py        ← Stage 4: Magnitude detection + log scale
│   ├── geocode.py         ← Stage 5: Map to WGS84 / GeoTIFF export
│   ├── change_detect.py   ← Bonus: Coherence-based change detection
│   └── pipeline.py        ← End-to-end runner
├── notebooks/
│   └── explore.ipynb      ← Interactive exploration / visualisation
├── results/               ← PNGs and GeoTIFFs written here
└── docs/
    └── theory.md          ← RDA theory notes
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your ESA Copernicus credentials
cp .env.example .env
# edit .env with your username/password

# 3. Download data for BBSR
python src/downloader.py

# 4. Run full pipeline
python src/pipeline.py --safe data/raw/<product>.SAFE

# 5. View results in results/
```

## AOI
Bhubaneswar, Odisha: 85.75°E – 85.95°E, 20.20°N – 20.35°N

## Output
- `results/bbsr_sar_detected.png`  — greyscale backscatter image
- `results/bbsr_sar.tif`           — georeferenced GeoTIFF
- `results/bbsr_change_map.png`    — coherence change detection (if 2 acquisitions)
