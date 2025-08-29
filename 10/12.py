#!/usr/bin/env python3
"""
PWI + Microbubble (MB) Simulation from a 2D CT Slice using k-Wave (Python)
-----------------------------------------------------------------------------
- Plane-Wave Imaging (PWI) through a heterogeneous medium derived from a 2D CT slice
- MBs spawned randomly (no vessel map), move with constant-velocity + bounce
- MB re-radiation uses a BODEGA-like linear damped-oscillator response
- Two-stage simulation per angle (recommended):
    1) TX pass: propagate plane wave, record local TX pressure at MB locations
    2) RX pass: drive MBs as small secondary sources with s_mb(t) computed from Stage 1,
       then measure at the array to get RF data through the same skull medium
- Optional fast (Born-style) synthesis path without the RX k-Wave pass
- Simple DAS beamforming + coherent compounding demo

Notes
-----
* Designed for the `kwave` Python package you already used (kspaceFirstOrder2D[ _gpu ]).
* Falls back to CPU if CUDA binary fails. 
* CT [-1,1] normalization → HU → (c, rho, alpha) mapping is heuristic and easy to tune.
* This is a template focusing on correctness and extensibility rather than speed.

Author: ChatGPT (GPT-5 Thinking)
Date: 2025-08-28 (Asia/Seoul)
"""
from __future__ import annotations
import os
import sys
import json
import math
import time
import argparse
import h5py
import numpy as np
from dataclasses import dataclass

# --- k-Wave imports (Python wrapper) ---
try:
    from kwave.kspaceFirstOrder2D import kspace_first_order_2d, kspace_first_order_2d_gpu
    from kwave.utils import cart2grid
    HAVE_KWAVE = True
except Exception as e:
    HAVE_KWAVE = False
    print("[WARN] k-Wave Python package not importable:", e, file=sys.stderr)

# ---- Optional SciPy for IIR filter design (for MB oscillator) ----
try:
    from scipy.signal import bilinear, lfilter
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# ----------------------------- Utility ---------------------------------

def hu_from_norm(x_norm: np.ndarray, hu_min: float = -1024.0, hu_max: float = 3000.0) -> np.ndarray:
    """Convert normalized [-1,1] array to HU linearly.
    Assumes input has already been clipped to [-1,1].
    """
    x = np.clip(x_norm, -1.0, 1.0)
    return ((x + 1.0) * 0.5) * (hu_max - hu_min) + hu_min

@dataclass
class MediumMaps:
    c: np.ndarray      # m/s
    rho: np.ndarray    # kg/m^3
    alpha: np.ndarray  # dB / (MHz^y * cm)
    alpha_power: float # tissue power-law exponent y


def props_from_hu(hu: np.ndarray, f0_mhz: float = 7.0) -> MediumMaps:
    """Heuristic mapping HU → (c, rho, alpha) for ultrasound.

    * Soft tissue baseline: c≈1540 m/s, rho≈1060 kg/m^3, alpha≈0.5 dB/(MHz·cm)
    * Cortical bone: c≈3000–3500 m/s, rho≈1900–2200 kg/m^3, alpha≈15–25 dB/(MHz·cm)

    We map piecewise linearly with gentle clamping. Adjust if you have better priors.
    """
    hu = np.nan_to_num(hu)

    # Speed of sound c(HU)
    c = 1540.0 + 0.50 * np.clip(hu, 0.0, 2000.0)  # +1 HU → +0.5 m/s up to ~+1000 m/s
    # c = np.where(hu < 0, 1540.0 + 0.10 * np.clip(hu, -200.0, 0.0), c)  # fat/air side: small decrease            NO HU < 0
    c = np.clip(c, 1400.0, 3500.0)

    # Density rho(HU)
    rho = 1060.0 + 0.40 * np.clip(hu, 0.0, 2500.0)  # ~ +1000 HU → +400 kg/m^3
    # rho = np.where(hu < 0, 1000.0 + 0.10 * np.clip(hu, -500.0, 0.0), rho)                                        NO HU < 0
    rho = np.clip(rho, 900.0, 2200.0)

    # Attenuation alpha(HU) in dB/(MHz^y·cm)
    # Tissue ~0.5, bone up to ~20+. Keep conservative to avoid numerical issues.
    alpha = 0.5 + 0.010 * np.clip(hu, 0.0, 2000.0)  # 2000 HU → +20 dB/(MHz·cm)
    # alpha = np.where(hu < 0, 0.35, alpha)                                                                        NO HU < 0
    alpha = np.clip(alpha, 0.1, 25.0)

    alpha_power = 1.01  # near-tissue like power law

    return MediumMaps(c=c.astype(np.float32),
                      rho=rho.astype(np.float32),
                      alpha=alpha.astype(np.float32),
                      alpha_power=float(alpha_power))

# ---------------------- Array & PWI waveform ----------------------------

def tone_burst(fs: float, f0: float, n_cycles: float, envelope: str = 'hann') -> np.ndarray:
    """Create a tone burst of n_cycles at f0 with optional taper. Returns pressure waveform.
    fs: sample rate [Hz]
    f0: center frequency [Hz]
    n_cycles: number of cycles of the carrier
    envelope: 'hann' | 'rect' | 'tukey0.5'
    """
    n = int(np.round(n_cycles * fs / f0))
    t = np.arange(n) / fs
    carrier = np.sin(2 * np.pi * f0 * t)
    if envelope == 'hann':
        win = np.hanning(n)
    elif envelope.startswith('tukey'):
        try:
            alpha = float(envelope.replace('tukey', ''))
        except Exception:
            alpha = 0.5
        # simple Tukey
        from numpy import hanning
        # quick polyfill: blend of flat and hann
        L = n
        win = np.ones(L)
        r = int(alpha * (L - 1) / 2)
        if r > 0:
            han = np.hanning(2 * r)
            win[:r] = han[:r]
            win[-r:] = han[-r:]
    else:
        win = np.ones(n)
    return (carrier * win).astype(np.float32)


def fractional_delay(sig: np.ndarray, fs: float, delay_s: float) -> np.ndarray:
    """Apply a (possibly fractional) delay by resampling with linear interpolation.
    Pads with zeros on the left; trims to original length.
    """
    n = sig.size
    t = np.arange(n) / fs
    t_shift = t - delay_s
    # interpolate; values outside become 0
    y = np.interp(t, t_shift, sig, left=0.0, right=0.0)    # *선형 보간: 나중에 라그랑지안 등으로 교체 가능*
    return y.astype(sig.dtype)

@dataclass
class ArrayGeom:
    x: np.ndarray  # x-position [m] per element center (shape: [Ne])
    y: np.ndarray  # y-position [m] per element center (all equal at top)
    pitch: float   # [m]
    ne: int
    elem_mask: np.ndarray  # boolean mask on grid for each element (Ny,Nx,Ne) or (Ne,Ny,Nx) depending on usage


def build_linear_array_mask(nx: int, ny: int, dx: float, ne: int, active_frac: float = 1.0) -> ArrayGeom:
    """Place a linear array along the top row spanning the full width.
    Each element occupies an equal horizontal segment of the top row.
    """
    # Grid physical extents
    Lx = nx * dx
    pitch = Lx / ne
    x_centers = (np.arange(ne) + 0.5) * pitch - Lx / 2.0
    y_top = - (ny * dx) / 2.0 + dx * 0.5  # top row y

    elem_mask = np.zeros((ne, ny, nx), dtype=bool)
    n_per_elem = nx // ne
    for e in range(ne):
        x0 = e * n_per_elem
        x1 = (e + 1) * n_per_elem if e < ne - 1 else nx
        elem_mask[e, 0, x0:x1] = True  # top row only

    # Active aperture (centered)
    if active_frac < 1.0:
        active = int(round(ne * active_frac))
        start = (ne - active) // 2
        active_idx = np.arange(ne)
        active_idx = active_idx[start:start + active]
        keep = np.zeros(ne, dtype=bool)
        keep[active_idx] = True
        elem_mask[~keep] = False

    X = x_centers.astype(np.float32)
    Y = np.full(ne, y_top, dtype=np.float32)
    return ArrayGeom(x=X, y=Y, pitch=pitch, ne=ne, elem_mask=elem_mask)

# --------------------------- MB model -----------------------------------
@dataclass
class MBParams:
    R_um: float      # bubble radius [micrometers]
    f0_hz: float     # Minnaert-like resonance [Hz]
    zeta: float      # damping ratio (~0.05-0.3)
    kappa: float     # drive gain (Pa→arb source units)


def minnaert_f0(radius_m: float, gamma: float = 1.4, P0: float = 101325.0, rho: float = 1000.0) -> float:
    """Minnaert resonance frequency (no shell), rough.
    f0 = (1/(2πR)) * sqrt(3γP0/ρ)
    """
    return (1.0 / (2.0 * math.pi * radius_m)) * math.sqrt(3.0 * gamma * P0 / rho)


def design_mb_params(n_mb: int, rng: np.random.Generator, R_um_range=(1.0, 4.0)) -> list[MBParams]:
    R_um = rng.uniform(R_um_range[0], R_um_range[1], size=n_mb)
    params = []
    for R in R_um:
        R_m = R * 1e-6
        f0 = minnaert_f0(R_m)
        # Clamp to reasonable imaging band (say 0.5–10 MHz)
        f0 = float(np.clip(f0, 0.5e6, 10e6))
        zeta = float(rng.uniform(0.08, 0.20))
        kappa = float(rng.uniform(5e-11, 2e-10))  # tuned small; adjust to match echo SNR
        params.append(MBParams(R_um=float(R), f0_hz=f0, zeta=zeta, kappa=kappa))
    return params


def mb_oscillator_filter(fs: float, f0: float, zeta: float):
    """Create a discrete-time IIR for x'' + 2ζω0 x' + ω0^2 x = κ u.
    Returns (b, a) such that y = lfilter(b, a, u) ≈ x (bubble radial response proxy).
    Uses bilinear transform. Requires SciPy; falls back to FIR if not present.
    """
    w0 = 2.0 * math.pi * f0
    # Continuous-time transfer X(s)/U(s) = 1 / (s^2 + 2ζω0 s + ω0^2)
    if HAVE_SCIPY:
        b_ct = [1.0]
        a_ct = [1.0, 2.0 * zeta * w0, w0 * w0]
        b_d, a_d = bilinear(b_ct, a_ct, fs=fs)
        return np.asarray(b_d, dtype=np.float64), np.asarray(a_d, dtype=np.float64)
    else:
        # Crude FIR fallback: second-order resonant IIR approximated by a short bandpass
        # (not ideal; strongly recommend installing SciPy)
        n = int(max(64, fs / f0 * 6))
        t = np.arange(n) / fs
        env = np.exp(-zeta * w0 * t)
        h = env * np.sin(w0 * t)
        h /= (np.linalg.norm(h) + 1e-9)
        return h.astype(np.float64), np.array([1.0], dtype=np.float64)


def mb_reradiated_series(p_tx: np.ndarray, fs: float, params: MBParams) -> np.ndarray:
    """Given local TX pressure p_tx(t) at the bubble, generate s_mb(t) (arbitrary units).
    We compute x(t) via second-order LTI, then scale by κ to get source pressure.
    """
    b, a = mb_oscillator_filter(fs, params.f0_hz, params.zeta)
    if HAVE_SCIPY:
        x = lfilter(b, a, p_tx.astype(np.float64))
    else:
        # FIR conv fallback
        x = np.convolve(p_tx.astype(np.float64), b, mode='full')[:p_tx.size]
    return (params.kappa * x).astype(np.float32)

# ----------------------- MB swarm & motion -------------------------------
@dataclass
class MBSwarm:
    xy: np.ndarray       # shape (N,2) in grid indices (ix, iy)
    vel: np.ndarray      # shape (N,2) in grid cells per frame (Δix, Δiy)
    params: list[MBParams]


def spawn_mb_random(mask_soft: np.ndarray, n_mb: int, rng: np.random.Generator,
                    speed_pix_per_frame=(0.1, 0.8)) -> MBSwarm:
    """Spawn MBs randomly inside mask_soft (True=allowed). Random velocity magnitudes.
    """
    iy, ix = np.where(mask_soft)
    if ix.size < n_mb:
        raise ValueError("Not enough soft-tissue pixels to place MBs. Relax mask.")
    sel = rng.choice(ix.size, size=n_mb, replace=False)
    xy = np.stack([ix[sel], iy[sel]], axis=1).astype(np.float32)

    # random velocities
    angles = rng.uniform(-math.pi, math.pi, size=n_mb)
    speeds = rng.uniform(speed_pix_per_frame[0], speed_pix_per_frame[1], size=n_mb)
    vel = np.stack([speeds * np.cos(angles), speeds * np.sin(angles)], axis=1).astype(np.float32)

    params = design_mb_params(n_mb, rng)
    return MBSwarm(xy=xy, vel=vel, params=params)


def step_mb(sw: MBSwarm, soft_mask: np.ndarray, rng: np.random.Generator) -> None:
    """Advance MBs one frame with bounce against non-soft regions and borders.
    Attempts to keep MBs inside allowed region.
    """
    H, W = soft_mask.shape
    xy_new = sw.xy + sw.vel

    # Reflect on borders
    for i in range(sw.xy.shape[0]):
        xi, yi = float(xy_new[i, 0]), float(xy_new[i, 1])
        # keep within image bounds
        if xi < 1 or xi >= W - 1:
            sw.vel[i, 0] *= -1.0
        if yi < 1 or yi >= H - 1:
            sw.vel[i, 1] *= -1.0
        xi = np.clip(xi, 1, W - 2)
        yi = np.clip(yi, 1, H - 2)
        # if new location hits hard tissue, randomize direction a bit
        if not soft_mask[int(round(yi)), int(round(xi))]:
            ang = rng.uniform(-math.pi, math.pi)
            spd = np.linalg.norm(sw.vel[i])
            sw.vel[i] = np.array([spd * np.cos(ang), spd * np.sin(ang)], dtype=np.float32)
        sw.xy[i] = np.array([xi, yi], dtype=np.float32)

# -------------------------- k-Wave wrappers ------------------------------
class KWaveRunner:
    def __init__(self, dx: float, dt: float, Nt: int, c_map: np.ndarray, rho_map: np.ndarray,
                 alpha_map: np.ndarray, alpha_power: float, pml_size: int = 20, pml_alpha: float = 1.5,
                 use_gpu: bool = True, device_index: int | None = None, binary_path: str | None = None):
        self.dx = dx
        self.dt = dt
        self.Nt = Nt
        self.c_map = c_map
        self.rho_map = rho_map
        self.alpha_map = alpha_map
        self.alpha_power = alpha_power
        self.pml_size = pml_size
        self.pml_alpha = pml_alpha
        self.use_gpu = use_gpu
        self.device_index = device_index
        self.binary_path = binary_path  # path to kspaceFirstOrder-CUDA if needed
        if not HAVE_KWAVE:
            raise RuntimeError("kwave package not available")

    def run(self, source_p_mask: np.ndarray, source_p: np.ndarray,
            sensor_mask: np.ndarray, record_p: bool = True) -> dict:
        """Run k-Wave forward simulation and return sensor data dict.
        - source_p_mask: boolean mask (Ny,Nx) for source points
        - source_p: array (Nsources, Nt)
        - sensor_mask: boolean mask (Ny,Nx) for sensor points
        """
        Ny, Nx = self.c_map.shape
        assert source_p_mask.shape == (Ny, Nx)
        assert sensor_mask.shape == (Ny, Nx)
        assert source_p.shape[1] == self.Nt
        n_src = int(source_p_mask.sum())
        assert source_p.shape[0] == n_src, f"source_p shape {source_p.shape} != n_src {n_src}"

        # Build k-Wave structures
        medium = {
            'sound_speed': self.c_map,
            'density': self.rho_map,
            'alpha_coeff': self.alpha_map,
            'alpha_power': self.alpha_power,
            'BonA': 0.0,  # linear acoustics
        }
        kgrid = {
            'Nx': Nx, 'Ny': Ny, 'dx': self.dx, 'dy': self.dx, 'dt': self.dt, 'Nt': self.Nt,
        }
        source = {'p_mask': source_p_mask.astype(bool), 'p': source_p.astype(np.float32)}
        sensor = {'mask': sensor_mask.astype(bool)}
        sim_opts = {
            'PMLInside': False,
            'PMLSize': self.pml_size,
            'PMLAlpha': self.pml_alpha,
            'DataCast': 'single',
            'RecordMovie': False,
        }
        exec_opts = {
            'is_gpu_simulation': self.use_gpu,
            'device_index': self.device_index,
            'binary_path': self.binary_path,
            'save_to_disk': False,
        }
        try:
            if self.use_gpu:
                out = kspace_first_order_2d_gpu(kgrid, medium, source, sensor, sim_opts, exec_opts)
            else:
                out = kspace_first_order_2d(kgrid, medium, source, sensor, sim_opts, exec_opts)
        except Exception as e:
            print(f"[WARN] GPU path failed ({e}); falling back to CPU...")
            out = kspace_first_order_2d(kgrid, medium, source, sensor, sim_opts, exec_opts)
        return out

# ---------------------- PWI TX/RX orchestration -------------------------

def build_tx_series_for_angle(angle_deg: float, f0: float, fs: float, n_cycles: float,
                              arr: ArrayGeom, dt: float, Nt: int, c_tx: float = 1540.0,
                              apod: str = 'hann') -> tuple[np.ndarray, np.ndarray]:
    """Create per-element delayed tone-bursts to steer a plane wave by angle.
    Returns (src_mask, src_p) where:
      - src_mask: (Ny,Nx) boolean, source at top row for each element span
      - src_p: (Nsources, Nt) float32 time series for each source-grid point
    """
    # One waveform per *grid-point* that belongs to the element mask
    # Build the base tone burst
    base = tone_burst(fs=fs, f0=f0, n_cycles=n_cycles, envelope='hann' if apod == 'hann' else 'rect')
    # pad waveform to Nt
    wave = np.zeros(Nt, dtype=np.float32)
    wave[:min(Nt, base.size)] = base[:min(Nt, base.size)]

    # Delays per element center (small-angle approx for plane-wave steering)
    theta = math.radians(angle_deg)
    delays = (arr.x * math.sin(theta)) / c_tx  # seconds; arr.x in meters

    # Apodization per element
    if apod == 'hann':
        apo = np.hanning(arr.ne).astype(np.float32)
    else:
        apo = np.ones(arr.ne, dtype=np.float32)

    # Build source mask & per-point series
    ne, Ny, Nx = arr.elem_mask.shape
    assert Ny * Nx == arr.elem_mask.shape[1] * arr.elem_mask.shape[2]
    src_mask = np.zeros((Ny, Nx), dtype=bool)
    src_rows = []
    for e in range(ne):
        mask_e = arr.elem_mask[e]
        idx = np.argwhere(mask_e)
        if idx.size == 0:
            continue
        # Each grid-point in element e gets the same delayed & apodized waveform
        delayed = fractional_delay(wave, fs, delays[e]) * apo[e]
        for (iy, ix) in idx:
            src_rows.append(delayed)
            src_mask[iy, ix] = True
    if len(src_rows) == 0:
        raise RuntimeError("No active source points")
    src_p = np.stack(src_rows, axis=0).astype(np.float32)
    return src_mask, src_p


def build_sensor_mask_from_array(arr: ArrayGeom) -> np.ndarray:
    """Use the same element line as receiver points (top row)."""
    # Union of element masks
    sensor_mask = np.any(arr.elem_mask, axis=0)
    return sensor_mask

# ---------------------------- Beamforming --------------------------------

def das_pwi_beamform(rf: np.ndarray, fs: float, arr: ArrayGeom, angles_deg: list[float],
                     c_bf: float, nx: int, ny: int, dx: float, z0_m: float = 0.0) -> np.ndarray:
    """Simple DAS beamformer for PWI (receive-only focusing, transmit is plane-wave at angles).
    rf shape: [n_angles, Nt, Ne]
    Returns B-mode image (ny,nx).
    """
    n_angles, Nt, Ne = rf.shape
    assert Ne == arr.ne
    x_axis = (np.arange(nx) - nx/2 + 0.5) * dx
    z_axis = (np.arange(ny) - ny/2 + 0.5) * dx  # assume square pixels

    img = np.zeros((ny, nx), dtype=np.float32)
    t_axis = np.arange(Nt) / fs

    for ai, ang in enumerate(angles_deg):
        th = math.radians(ang)
        # plane-wave launch from z=z0 (top). For each pixel, compute time-of-flight to elements.
        # TX path is approx: x*cos th + z*sin th / c_bf (plane-wave front hit time)
        # RX path: distance from pixel to each element center
        for iy, z in enumerate(z_axis):
            for ix, x in enumerate(x_axis):
                # TX delay (plane wave reaching pixel)
                t_tx = (x * math.sin(th) + (z - z0_m) * math.cos(th)) / c_bf
                # RX delays per element
                rx_d = np.sqrt((x - arr.x)**2 + (z - arr.y)**2)
                t_rx = rx_d / c_bf
                t_total = t_tx + t_rx
                # Sample & sum
                it = (t_total * fs).astype(np.int32)
                valid = (it >= 0) & (it < Nt)
                s = rf[ai, it[valid], np.where(valid)[0]].sum()
                img[iy, ix] += s
    # Envelope detection (magnitude) and log compression
    env = np.abs(img)
    env /= (env.max() + 1e-9)
    bmode = 20.0 * np.log10(env + 1e-6)
    bmode = np.clip((bmode + 60.0) / 60.0, 0.0, 1.0)  # map [-60,0] dB → [0,1]
    return bmode.astype(np.float32)

# ------------------------------- Main -----------------------------------

def main():
    ap = argparse.ArgumentParser(description="PWI + MB k-Wave simulator from 2D CT slice")
    ap.add_argument('--ct_npy', type=str, required=True, default='/home/user/2/00/01/8/dataset_x_0230.npy', help='Path to normalized [-1,1] CT .npy')
    ap.add_argument('--dx_mm', type=float, default=0.02, help='Pixel size [mm] (e.g., 0.02)')
    ap.add_argument('--f0_mhz', type=float, default=7.0, help='Center frequency [MHz]')
    ap.add_argument('--cycles', type=float, default=2.0, help='Tone-burst cycles')
    ap.add_argument('--angles', type=str, default='-12,-6,0,6,12', help='Comma-separated steering angles [deg]')
    ap.add_argument('--fs_oversample', type=float, default=6.0, help='Sample rate multiple of f0 (fs=oversample*f0)')
    ap.add_argument('--Nt', type=int, default=4096, help='#time samples')
    ap.add_argument('--ne', type=int, default=128, help='# array elements')
    ap.add_argument('--active_frac', type=float, default=1.0, help='Fraction of active aperture (0-1)') 
    ap.add_argument('--n_frames', type=int, default=16, help='# frames (MB motion)')
    ap.add_argument('--n_mb', type=int, default=500, help='# microbubbles')
    ap.add_argument('--mb_speed_pix', type=str, default='0.2,0.6', help='MB speed range [pix/frame], e.g. 0.2,0.6')
    ap.add_argument('--gpu', action='store_true', help='Try CUDA binary if available')
    ap.add_argument('--device', type=int, default=None, help='CUDA device index')
    ap.add_argument('--binary', type=str, default='/home/user/binaries/kspaceFirstOrder-CUDA', help='Path to kspaceFirstOrder-CUDA binary')
    ap.add_argument('--save_h5', type=str, default='out_pwi_mb.h5', help='Output HDF5 path')
    ap.add_argument('--seed', type=int, default=123, help='RNG seed')
    ap.add_argument('--born_only', action='store_true', help='Skip RX k-Wave; synthesize echoes fast (approx)')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # 1) Load CT
    ct_norm = np.load(args.ct_npy).astype(np.float32)
    if ct_norm.ndim != 2:
        raise ValueError("ct_npy must be 2D array")
    Ny, Nx = ct_norm.shape

    # 2) HU → medium maps
    hu = ct_norm # hu_from_norm(ct_norm)
    maps = props_from_hu(hu, f0_mhz=args.f0_mhz)

    # 3) Discretization
    dx = args.dx_mm * 1e-3  # m
    f0 = args.f0_mhz * 1e6
    fs = args.fs_oversample * f0
    dt = 1.0 / fs
    Nt = args.Nt

    # Stability sanity-check (very basic); reduce dt if needed
    cmax = float(np.max(maps.c))
    cfl = cmax * dt / dx
    if cfl > 0.3:
        print(f"[WARN] CFL={cfl:.3f} too high; increasing fs")
        dt = (0.3 * dx) / cmax
        fs = 1.0 / dt
        Nt = int(max(Nt, 4096))

    # 4) Array geometry & masks
    arr = build_linear_array_mask(nx=Nx, ny=Ny, dx=dx, ne=args.ne, active_frac=args.active_frac)
    sensor_mask = build_sensor_mask_from_array(arr)

    # Soft mask for MB spawn: avoid high-HU regions (bone)
    soft_mask = hu < 200.0

    # 5) MB swarm init
    spd_lo, spd_hi = [float(x) for x in args.mb_speed_pix.split(',')]
    swarm = spawn_mb_random(soft_mask=soft_mask, n_mb=args.n_mb, rng=rng,
                            speed_pix_per_frame=(spd_lo, spd_hi))

    # 6) Prepare runner
    runner = KWaveRunner(dx=dx, dt=dt, Nt=Nt, c_map=maps.c, rho_map=maps.rho,
                         alpha_map=maps.alpha, alpha_power=maps.alpha_power,
                         pml_size=20, pml_alpha=1.5, use_gpu=args.gpu,
                         device_index=args.device, binary_path=args.binary)

    # 7) Outputs
    angles = [float(s) for s in args.angles.split(',') if s]
    Ne = arr.ne
    RF = np.zeros((args.n_frames, len(angles), Nt, Ne), dtype=np.float32)
    MB_POS = np.zeros((args.n_frames, swarm.xy.shape[0], 2), dtype=np.float32)

    # Helper to build source index → coordinate map for later (MB stage)
    def mask_to_indices(mask: np.ndarray) -> np.ndarray:
        return np.argwhere(mask)

    elem_indices = mask_to_indices(sensor_mask)  # receivers

    # 8) Main frames loop
    for t_idx in range(args.n_frames):
        print(f"[Frame {t_idx+1}/{args.n_frames}] MB count={swarm.xy.shape[0]}")
        MB_POS[t_idx] = swarm.xy

        # Build MB params list for this frame (static per bubble)
        # Pre-compute per-MB oscillator filters only once per run (cache by f0,zeta)

        # --- per-angle loop ---
        for ai, ang in enumerate(angles):
            print(f"  Angle {ang:.1f} deg: TX pass…")
            # TX source at top line
            src_mask_tx, src_p_tx = build_tx_series_for_angle(
                angle_deg=ang, f0=f0, fs=fs, n_cycles=args.cycles, arr=arr, dt=dt, Nt=Nt,
                c_tx=1540.0, apod='hann')

            # Sensor at MB positions to record local TX pressure
            mb_mask = np.zeros((Ny, Nx), dtype=bool)
            mb_idx_int = np.round(swarm.xy).astype(int)
            mb_idx_int[:, 0] = np.clip(mb_idx_int[:, 0], 0, Nx - 1)
            mb_idx_int[:, 1] = np.clip(mb_idx_int[:, 1], 0, Ny - 1)
            mb_mask[mb_idx_int[:, 1], mb_idx_int[:, 0]] = True

            out_tx = runner.run(source_p_mask=src_mask_tx, source_p=src_p_tx,
                                sensor_mask=mb_mask, record_p=True)
            # out_tx['p'] expected shape: (Nt, Nmb) or (Nmb, Nt) depending on wrapper — normalize
            p_mb = out_tx.get('p')
            if p_mb is None:
                raise RuntimeError("TX run did not return 'p' at MB locations")
            # Convert to shape (Nmb, Nt)
            if p_mb.shape[0] == Nt:
                p_mb = np.transpose(p_mb, (1, 0))
            elif p_mb.shape[1] == Nt:
                pass
            else:
                raise RuntimeError(f"Unexpected TX sensor shape {p_mb.shape}")

            # Generate reradiated series for each MB
            print("    Synthesizing MB re-radiation…")
            smb_rows = []
            for bi in range(swarm.xy.shape[0]):
                smb = mb_reradiated_series(p_tx=p_mb[bi], fs=fs, params=swarm.params[bi])
                smb_rows.append(smb)
            smb_arr = np.stack(smb_rows, axis=0)  # (Nmb, Nt)

            if args.born_only:
                # Fast synthesis: propagate to each element with 1/r delay & sum (homogeneous c)
                rf = np.zeros((Nt, Ne), dtype=np.float32)
                px = (swarm.xy[:, 0] - Nx/2 + 0.5) * dx
                pz = (swarm.xy[:, 1] - Ny/2 + 0.5) * dx
                for e in range(Ne):
                    rx = arr.x[e]
                    rz = arr.y[e]
                    r = np.sqrt((px - rx)**2 + (pz - rz)**2)
                    tprop = r / 1540.0
                    it = (tprop * fs).astype(np.int32)
                    for bi in range(swarm.xy.shape[0]):
                        valid = (it[bi] >= 0) & (it[bi] < Nt)
                        if valid:
                            rf[it[bi], e] += smb_arr[bi, 0]  # very rough; for speed/demo only
                RF[t_idx, ai] = rf
            else:
                # RX k-Wave pass: set MBs as pressure point sources
                print("    RX pass via k-Wave…")
                src_mask_rx = mb_mask.copy()
                src_p_rx = smb_arr.astype(np.float32)  # (Nmb, Nt)
                out_rx = runner.run(source_p_mask=src_mask_rx, source_p=src_p_rx,
                                    sensor_mask=sensor_mask, record_p=True)
                p_rx = out_rx.get('p')
                if p_rx is None:
                    raise RuntimeError("RX run did not return 'p' at array")
                # Normalize to (Nt, Ne)
                if p_rx.ndim == 2 and p_rx.shape[0] == Nt:
                    RF[t_idx, ai] = p_rx
                elif p_rx.ndim == 2 and p_rx.shape[1] == Nt:
                    RF[t_idx, ai] = p_rx.T
                else:
                    raise RuntimeError(f"Unexpected RX shape {p_rx.shape}")

        # Advance MBs for next frame
        step_mb(swarm, soft_mask, rng)

    # 9) Optional beamforming demo on last frame
    try:
        bmode = das_pwi_beamform(rf=RF[-1], fs=fs, arr=arr, angles_deg=angles,
                                 c_bf=1540.0, nx=Nx, ny=Ny, dx=dx, z0_m=arr.y[0])
    except Exception as e:
        print(f"[WARN] Beamforming failed: {e}")
        bmode = None

    # 10) Save HDF5
    with h5py.File(args.save_h5, 'w') as f:
        f.create_dataset('RF', data=RF, compression='gzip')  # [T, A, Nt, Ne]
        f.create_dataset('MB_POS', data=MB_POS, compression='gzip')  # [T, Nmb, 2] (ix,iy)
        f.create_dataset('medium/c', data=maps.c, compression='gzip')
        f.create_dataset('medium/rho', data=maps.rho, compression='gzip')
        f.create_dataset('medium/alpha', data=maps.alpha, compression='gzip')
        f.create_dataset('bmode_last', data=bmode if bmode is not None else np.array([]), compression='gzip')
        meta = {
            'dx_m': dx,
            'fs_hz': fs,
            'dt_s': dt,
            'Nt': Nt,
            'angles_deg': angles,
            'f0_hz': f0,
            'cycles': args.cycles,
            'ne': args.ne,
            'active_frac': args.active_frac,
            'alpha_power': maps.alpha_power,
        }
        f.attrs['meta'] = json.dumps(meta)
    print(f"Saved → {args.save_h5}")


if __name__ == '__main__':
    main()
