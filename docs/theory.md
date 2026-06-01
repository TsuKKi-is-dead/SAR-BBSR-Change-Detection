# SAR Processing Theory — Range-Doppler Algorithm

## Why SAR needs processing

A real-aperture radar gets azimuth resolution limited by the antenna beam-width:
`ρ_az = λ * R / D` — at 800 km range, a 6m antenna gives ~7.4 km resolution.

SAR (Synthetic Aperture Radar) exploits the satellite's forward motion to
synthesise a much longer virtual aperture, achieving:
`ρ_az ≈ D/2` — **independent of range and wavelength**.

For Sentinel-1 with D=12m, this gives ~6m azimuth resolution globally.

---

## Signal model

The received signal from a point target at slant range R₀ is:

```
s(τ, η) = A₀ · wr(τ - 2R(η)/c) · wa(η) · exp(-j4πR(η)/λ) · exp(jπKr(τ - 2R(η)/c)²)
```

where:
- τ = range (fast) time
- η = azimuth (slow) time  
- R(η) = √(R₀² + v²η²)  — hyperbolic range history
- Kr = chirp rate (Hz/s)
- wa, wr = azimuth and range envelope functions

---

## Range-Doppler Algorithm (RDA)

### Step 1: Range compression
Transform to range-frequency domain, apply matched filter:

```
H_rc(f_τ) = exp(jπ f_τ² / Kr)   [conjugate of chirp spectrum]
```

After IFFT: targets are focused in range with resolution `ρ_r = c/(2B)`.

### Step 2: Azimuth FFT → Range-Doppler domain
Columns become Doppler frequency f_η. The range migration curve becomes:

```
R(f_η) = R₀ / √(1 - (f_η·λ/(2v))²)
```

### Step 3: Range Cell Migration Correction (RCMC)
Shift each Doppler frequency line by `ΔR(f_η) = R(f_η) - R₀`:

```
ΔR(f_η, R₀) = R₀ · [1/√(1 - (f_ηλ/2v)²) - 1]
```

This is a range-dependent interpolation problem (sinc or spline).

### Step 4: Azimuth compression
Apply range-dependent azimuth matched filter:

```
H_az(f_η; R₀) = exp(jπ f_η² / Ka(R₀))

Ka(R₀) = 2v² / (λR₀)   — azimuth FM rate
```

After azimuth IFFT: fully focused 2D complex image.

---

## Sentinel-1 IW specifics

Sentinel-1 uses **TOPS (Terrain Observation with Progressive Scans)** mode,
which scans the antenna in azimuth during each burst, resulting in:

- 3 subswaths (IW1, IW2, IW3), each ~80km wide
- ~9 bursts per subswath
- Burst overlap used for burst stitching (EAP correction needed)
- Doppler centroid ≠ 0 (squinted acquisition)

For BBSR (20°N, 85.8°E), **IW2** typically has the best coverage.

---

## Key parameters — Sentinel-1 C-band IW

| Parameter           | Value                |
|---------------------|----------------------|
| Carrier frequency   | 5.405 GHz            |
| Wavelength (λ)      | 5.547 cm             |
| Range bandwidth (B) | 56.5 MHz             |
| Range resolution    | ~2.3 m (slant)       |
| Azimuth resolution  | ~14 m (IW, 3 looks)  |
| PRF                 | ~486 Hz              |
| Orbit altitude      | ~693 km              |
| Orbital velocity    | ~7.2 km/s            |
| Revisit time        | 12 days (same track) |

---

## Output interpretation

- **Backscatter (σ°)**: ratio of reflected to transmitted power per unit area
  - High (bright): buildings, metallic surfaces, corner reflectors
  - Low (dark): calm water, smooth surfaces, low-moisture desert

- **BBSR signatures**:
  - Bhubaneswar urban area → bright, heterogeneous
  - Mahanadi delta / wetlands → variable with season
  - Chilika Lake → dark (smooth water surface)
  - Agricultural areas → moderate, seasonal variation

---

## Further reading

1. Cumming & Wong, *Digital Processing of SAR Data* (2005) — definitive RDA reference
2. ESA Sentinel-1 Product Definition: https://sentinel.esa.int/documents/247904/1877131/Sentinel-1-Product-Definition
3. SAR Handbook (SERVIR): https://www.servirglobal.net/Portals/0/Documents/Articles/2019_SAR_Handbook.pdf
