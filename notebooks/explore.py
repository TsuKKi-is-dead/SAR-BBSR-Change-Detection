# %% [markdown]
# # SAR Image Reconstruction — BBSR Odisha
# **Interactive exploration notebook**
# Run cells top-to-bottom after running the pipeline at least once.

# %%
import sys
sys.path.insert(0, '../src')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
%matplotlib inline
plt.rcParams['figure.facecolor'] = '#111'
plt.rcParams['axes.facecolor']   = '#111'
plt.rcParams['text.color']       = 'white'
plt.rcParams['axes.labelcolor']  = 'white'
plt.rcParams['xtick.color']      = 'white'
plt.rcParams['ytick.color']      = 'white'

import config as cfg

# %% [markdown]
# ## 1. Inspect raw IQ data

# %%
# Change this to your actual .SAFE path
SAFE_PATH = list(cfg.DATA_RAW.glob("*.SAFE"))
print("Found SAFE products:")
for p in SAFE_PATH:
    print(f"  {p.name}")

# %%
from reader import read_burst

burst = read_burst(SAFE_PATH[0])
raw = burst.raw_iq

print(f"Shape    : {raw.shape}")
print(f"Dtype    : {raw.dtype}")
print(f"Max amp  : {np.abs(raw).max():.1f}")
print(f"Mean amp : {np.abs(raw).mean():.3f}")

# %%
# Plot a small section of raw IQ (before any processing)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
section = np.abs(raw[:500, :1000])

axes[0].imshow(section, cmap='inferno', aspect='auto', interpolation='nearest')
axes[0].set_title('Raw amplitude (first 500 az lines, 1000 rg samples)')
axes[0].set_xlabel('Range (samples)')
axes[0].set_ylabel('Azimuth (samples)')

axes[1].plot(20*np.log10(section[250, :] + 1e-3), color='#4DA6FF', linewidth=0.8)
axes[1].set_title('Range profile — azimuth line 250 (raw)')
axes[1].set_xlabel('Range sample')
axes[1].set_ylabel('Amplitude (dB)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Range compression effect

# %%
from range_compress import range_compress

rc = range_compress(burst)

# Compare one range line before and after
line = 250
raw_line = np.abs(burst.raw_iq[line, :])
rc_line  = np.abs(rc[line, :])

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(20*np.log10(raw_line + 1e-3), color='#FF6B6B', linewidth=0.7, label='Raw')
axes[0].plot(20*np.log10(rc_line  + 1e-3), color='#4DA6FF', linewidth=0.7, label='After RC', alpha=0.8)
axes[0].set_title('Range profile: raw vs range-compressed')
axes[0].set_xlabel('Range sample')
axes[0].set_ylabel('Amplitude (dB)')
axes[0].legend()

axes[1].imshow(20*np.log10(np.abs(rc[:500, :1000]) + 1e-3),
               cmap='gray', aspect='auto', vmin=-20, vmax=40)
axes[1].set_title('Range-compressed amplitude')
axes[1].set_xlabel('Range'); axes[1].set_ylabel('Azimuth')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. RCMC and azimuth compression

# %%
from rcmc import rcmc
from azimuth_compress import azimuth_compress

rcmc_data = rcmc(rc, burst, interp='linear')
focused   = azimuth_compress(rcmc_data, burst)

np.save(cfg.NP_FOCUSED, focused)
print("Focused image saved.")

# %%
# Full focused image
from detector import multilook, to_db, clip_percentile

power   = multilook(focused, 3, 1)
db_img  = to_db(power)
display = clip_percentile(db_img)

fig, ax = plt.subplots(figsize=(14, 11))
ax.imshow(display, cmap='gray', aspect='auto', interpolation='nearest')
ax.set_title(f'Focused SAR — {cfg.AOI_NAME}  (3×1 looks, backscatter dB)')
ax.set_xlabel('Range (samples)')
ax.set_ylabel('Azimuth (samples)')
plt.colorbar(plt.cm.ScalarMappable(cmap='gray',
    norm=mcolors.Normalize(vmin=db_img.min(), vmax=db_img.max())),
    ax=ax, label='dB')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Zoom into Bhubaneswar area

# %%
# After geocoding, load the GeoTIFF and show with lat/lon axes
try:
    import rasterio
    from rasterio.plot import show

    with rasterio.open(cfg.OUT_GEOTIFF) as src:
        data  = src.read(1)
        bounds = src.bounds

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(data, cmap='gray', extent=[bounds.left, bounds.right,
                                          bounds.bottom, bounds.top],
              aspect='auto', interpolation='nearest')
    ax.set_title(f'SAR Backscatter — {cfg.AOI_NAME} (WGS84)')
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    plt.colorbar(plt.cm.ScalarMappable(cmap='gray',
        norm=mcolors.Normalize(data.min(), data.max())),
        ax=ax, label='dB (clipped)')
    plt.tight_layout()
    plt.show()

except ImportError:
    print("rasterio not installed. Run: pip install rasterio")
    print("Showing slant-range image instead.")

# %% [markdown]
# ## 5. Phase image (interferometry preparation)

# %%
phase = np.angle(focused)

fig, ax = plt.subplots(figsize=(12, 9))
ax.imshow(phase, cmap='hsv', aspect='auto', interpolation='nearest', vmin=-np.pi, vmax=np.pi)
ax.set_title('Focused SAR — phase (rad)')
ax.set_xlabel('Range'); ax.set_ylabel('Azimuth')
plt.colorbar(plt.cm.ScalarMappable(cmap='hsv',
    norm=mcolors.Normalize(-np.pi, np.pi)),
    ax=ax, label='Phase (rad)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Coherence change detection (requires two acquisitions)

# %%
FOCUSED2_PATH = cfg.DATA_PROCESSED / "focused2.npy"

if FOCUSED2_PATH.exists():
    from change_detect import estimate_coherence, make_change_map

    img1 = np.load(cfg.NP_FOCUSED).astype(np.complex64)
    img2 = np.load(FOCUSED2_PATH).astype(np.complex64)

    # Match shapes
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1, img2 = img1[:h,:w], img2[:h,:w]

    coh    = estimate_coherence(img1, img2, window_az=5, window_rg=20)
    change = make_change_map(coh, threshold=0.3)

    print(f"\nCoherence stats: mean={coh.mean():.3f}  std={coh.std():.3f}")
    print(f"Changed pixels: {100*change.mean():.1f}%")
else:
    print("No second acquisition found.")
    print("Download a second pass and process it to enable change detection.")
    print(f"Expected: {FOCUSED2_PATH}")
