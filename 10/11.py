# sim.py — PWI-ULM (k-Wave-python, 128ch, skull HU→물성, per-pixel delays)
# deps: numpy, scipy, h5py, k-wave-python
import os
import numpy as np
from scipy.io import savemat

# ---- k-Wave-python API (정식 네임스페이스) ----
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.kspaceFirstOrder2D import (
    kspace_first_order_2d_gpu,   # GPU (CUDA)
    kspaceFirstOrder2DC          # CPU (OMP)
)
from kwave.utils.signals import tone_burst
from kwave.utils.conversion import hounsfield2density

# -------------------------- 파라미터 --------------------------
device = 'single'             # 내부 캐스팅용(데이터형)
Lx_cm = 6.0
Ly_cm = 6.0
gs_mm = 0.1171875             # -> 512
freq = 0.5e6
cycles = 5
angles = [0]                  # 먼저 0도 테스트(잘되면 range(-5,6)로..)
T = 16
n_e = 128
td_loc = (0.3, Lx_cm/2)       # (x[cm], y[cm])
pitch_mm, elem_w_mm = 0.30, 0.25
p_amp = 1e5                   # [Pa]

# 파일·폴더
INPUTS_NPY_DIR = "./00"
OUT_RF_DIR = "./ouputs/out_rf"
os.makedirs(OUT_RF_DIR, exist_ok=True)


#
# --- Physical microbubble (air-like) properties ---
BUBBLE_C   = 340.0   # m/s, effective sound speed for gas core
BUBBLE_RHO = 1.2     # kg/m^3, effective density for gas core
# Default geometric params in pixels (grid ~0.117 mm/px)
MB_RADIUS_PIX_DEFAULT     = 1        # ≈117 µm (matches large UCA microbubbles cluster)
MB_EDGE_SOFTNESS_PIX      = 0.75     # soft-edge thickness for smooth impedance transition

# -------------------------- 유틸 --------------------------
def makeLinearArrayMask(Nx, Ny, td_x, y_center, n_e, elem_w_pix, pitch_pix):
    mask = np.zeros((Nx, Ny), dtype=bool)
    col0 = int(np.round(y_center - (n_e*pitch_pix - (pitch_pix-elem_w_pix))/2))
    elem_cols = np.zeros((n_e, 2), dtype=int)
    for e in range(n_e):
        y0 = col0 + e*pitch_pix
        y1 = y0 + elem_w_pix - 1
        y0 = max(1, y0); y1 = min(Ny, y1)
        mask[td_x-1, (y0-1):y1] = True   # Python 0-index
        elem_cols[e,:] = [y0, y1]
    yc = np.rint(np.mean(elem_cols, axis=1)).astype(int)
    return mask, elem_cols, yc

def plane_wave_delays(yc_pix, theta_deg, dy_m, c_ref, dt_guess):      # tau = y sin(theta) / c 계산
    y = (yc_pix.astype(float) - 1.0) * dy_m  # 1-based to 0-based
    tau = (y * np.sin(np.deg2rad(theta_deg))) / c_ref
    d_samp = np.round(tau / dt_guess).astype(np.int64)
    dmin = d_samp.min()
    if dmin < 0:
        d_samp = d_samp - dmin
    return d_samp.astype(np.int64)    # [yref를 0으로 두고 -dmin 평행이동] == [yref = min(y)로 둔 것과 동치]

def randsample_fast(vec_idx, K, replace=True, rng=None):
    if rng is None: rng = np.random.default_rng()
    if replace:
        idx = rng.integers(low=0, high=vec_idx.size, size=K)
        return vec_idx[idx]
    else:
        if K > vec_idx.size: raise ValueError("K>n without replacement")
        idx = rng.choice(vec_idx.size, size=K, replace=False)
        return vec_idx[idx]

# Simplified MB tracks generator: only uses mask_water, no ROI/vessel mask, uses mixture speed
def make_mb_tracks_rand_avoid_bone(mask_water, T, K, rng=None):
    """
    Generate K microbubble tracks for T frames **only inside mask_water** (True=allowed).
    - No boundary bounce.
    - If next step would leave mask_water or the domain, re-sample direction (same speed)
      up to MAX_TRIES; if still invalid, stall for that frame.
    """
    if rng is None:
        rng = np.random.default_rng()
    Nx, Ny = mask_water.shape

    # Allowed region = mask_water
    allowed = mask_water.astype(bool)

    # Initial positions uniformly over allowed pixels (Fortran order for consistency)
    allowed_idx = np.flatnonzero(allowed.ravel(order='F'))
    if allowed_idx.size == 0:
        raise ValueError("No allowed pixels (mask_water all False)")
    sel = randsample_fast(allowed_idx, K, True, rng)
    x, y = np.unravel_index(sel, (Nx, Ny), order='F')
    x = x.astype(float) + 1.0  # work in 1-based-like float coords
    y = y.astype(float) + 1.0

    # Speeds: mixture distribution in pixels/frame
    v  = sample_speed_pix_per_frame(rng, K)
    th = 2*np.pi*rng.random(K)
    vx = v*np.cos(th)
    vy = v*np.sin(th)

    tracks = []
    for t in range(T):
        tracks.append(np.column_stack([np.rint(x).astype(int), np.rint(y).astype(int)]))

        # Propose next step
        x_new = x + vx
        y_new = y + vy

        # Invalid if out of bounds or into disallowed pixel (no bounce)
        invalid = (
            (x_new < 2) | (x_new > Nx - 1) |
            (y_new < 2) | (y_new > Ny - 1) |
            (~allowed[(np.rint(x_new).astype(int) - 1, np.rint(y_new).astype(int) - 1)])
        )

        if np.any(invalid):
            MAX_TRIES = 8
            for _ in range(MAX_TRIES):
                idx = np.flatnonzero(invalid)
                if idx.size == 0:
                    break
                th = 2 * np.pi * rng.random(idx.size)
                vx[idx] = v[idx] * np.cos(th)
                vy[idx] = v[idx] * np.sin(th)
                x_new[idx] = x[idx] + vx[idx]
                y_new[idx] = y[idx] + vy[idx]
                invalid[idx] = (
                    (x_new[idx] < 2) | (x_new[idx] > Nx - 1) |
                    (y_new[idx] < 2) | (y_new[idx] > Ny - 1) |
                    (~allowed[(np.rint(x_new[idx]).astype(int) - 1, np.rint(y_new[idx]).astype(int) - 1)])
                )

            # Still invalid → stall this step
            if np.any(invalid):
                idx = np.flatnonzero(invalid)
                x_new[idx] = x[idx]
                y_new[idx] = y[idx]
                # velocities unchanged

        # Final update (no bounce)
        x = x_new
        y = y_new

    return tracks

def sample_speed_pix_per_frame(rng, K):
    bins = rng.choice([0,1,2], size=K, p=[0.70,0.25,0.05])
    v = np.empty(K, dtype=float)
    v[bins==0] = 0.02 + (0.20-0.02)*rng.random(np.sum(bins==0))
    v[bins==1] = 0.20 + (0.60-0.20)*rng.random(np.sum(bins==1))
    v[bins==2] = 0.60 + (1.50-0.60)*rng.random(np.sum(bins==2))
    return v

def apply_mb_disks_physical(c, rho, centers, r_pix=MB_RADIUS_PIX_DEFAULT, edge_soft=MB_EDGE_SOFTNESS_PIX):
    """Stamp gas-like microbubbles with soft edges.
    Inside each disk (radius r_pix), set c≈340 m/s, ρ≈1.2 kg/m^3 with a smooth radial blend
    over `edge_soft` pixels to reduce staircasing and numerical ringing.
    """
    c2 = c.copy(); rho2 = rho.copy()
    Nx, Ny = c.shape
    X, Y = np.mgrid[1:Nx+1, 1:Ny+1]
    r_pix = float(max(0.5, r_pix))
    edge  = float(max(0.0, edge_soft))

    for cx, cy in centers:
        # radial distance field [pixels]
        R = np.sqrt((X - cx)**2 + (Y - cy)**2)
        # Soft mask w in [0,1]: 1=inside bubble core, 0=background
        if edge > 1e-6:
            # cosine ramp from r to r+edge
            w = 0.5 * (1.0 + np.cos(np.clip((R - r_pix) / edge, 0.0, 1.0) * np.pi))
            w[R <= r_pix] = 1.0
            w[R >= r_pix + edge] = 0.0
        else:
            w = (R <= r_pix).astype(float)

        # Blend material parameters: new = w*BUBBLE + (1-w)*old
        c2 = w * BUBBLE_C   + (1.0 - w) * c2
        rho2 = w * BUBBLE_RHO + (1.0 - w) * rho2

    return c2, rho2

def run2d(kgrid, source, sensor, medium, sim_opts, exec_opts):
    """ GPU / CPU """
    try:
        exec_opts.is_gpu_simulation = True
        return kspace_first_order_2d_gpu(kgrid, source, sensor, medium, sim_opts, exec_opts)
    except Exception:
        exec_opts.is_gpu_simulation = False
        return kspaceFirstOrder2DC(kgrid, source, sensor, medium, sim_opts, exec_opts)


# -------------------------- main --------------------------
def main():
    # Recursively collect all .npy files under INPUTS_NPY_DIR
    npy_files = []
    for root, _, files in os.walk(INPUTS_NPY_DIR):
        for f in files:
            if f.endswith(".npy"):
                npy_files.append(os.path.join(root, f))
    npy_files = sorted(npy_files)
    assert npy_files, f"입력 폴더 {INPUTS_NPY_DIR} 하위에 .npy 파일이 없습니다."

    for npy_path in npy_files:
        skull = np.load(npy_path).astype(np.float32)

        Nx = int(round(Lx_cm / (gs_mm/10.0)))
        Ny = int(round(Ly_cm / (gs_mm/10.0)))
        assert skull.shape == (Nx, Ny), f"skull 크기 {skull.shape} != ({Nx},{Ny})"
        print(f"Loaded {npy_path}  size={skull.shape}")

        dx = dy = gs_mm * 1e-3

        # === Grid ===
        kgrid = kWaveGrid((Nx, Ny), (dx, dy))

        # === Medium ===
        skull_clipped = skull.copy()
        skull_clipped[skull_clipped < 25] = 0.0
        density_map = hounsfield2density(skull_clipped)
        density_map[density_map < 997.0] = 997.0
        velocity_map = 1.33 * density_map + 167.0

        attenuation_map = np.full((Nx, Ny), 0.53, dtype=np.float32)
        attenuation_map[skull_clipped > 100.0] = 13.3

        medium = kWaveMedium(
            sound_speed=velocity_map.astype(np.float32),
            density=density_map.astype(np.float32),
            alpha_coeff=attenuation_map.astype(np.float32),
            alpha_power=1.5
        )

        # === 배열 마스크 ===
        pitch_pix  = max(1, int(round(pitch_mm/gs_mm)))
        elem_w_pix = max(1, int(round(elem_w_mm/gs_mm)))
        td_x = int(round(td_loc[0] / (gs_mm/10.0)))
        y_center = int(round(td_loc[1] / (gs_mm/10.0)))
        array_mask, elem_cols, yc = makeLinearArrayMask(Nx, Ny, td_x, y_center, n_e, elem_w_pix, pitch_pix)

        # === 시간 옵션 ===
        c0 = float(np.min(velocity_map))
        CFL = 0.10
        depth_m = Lx_cm * 1e-2
        t_end = 2*depth_m/c0 * 1.2

        # --- Define time array using kgrid ---
        kgrid.makeTime(c0, CFL, t_end)
        dt_actual = float(kgrid.dt)
        Nt_actual = int(kgrid.Nt)

        sim_opts = SimulationOptions(
            pml_inside=False,
            pml_size=14,
            pml_alpha=5.0,
            data_cast=device,
            save_to_disk=True
        )
        exec_opts = SimulationExecutionOptions(is_gpu_simulation=True)

        # === MB 트랙 ===
        # Spawn & move **only** where HU==0 (after clipping, this is water/soft).
        mask_water_spawn = (skull_clipped == 0.0)
        K = 100  # adjust as needed
        tracks = make_mb_tracks_rand_avoid_bone(
            mask_water_spawn, T, K, rng=np.random.default_rng(1234)
        )

        # (dt_actual, Nt_actual are now defined from kgrid.makeTime)

        # === 프레임/각도 루프 ===
        for t in range(1, T+1):
            # Use physical gas-like bubbles with soft edge (defaults ~1 px radius, 0.75 px ramp)
            c_t, rho_t = apply_mb_disks_physical(velocity_map, density_map, tracks[t-1])

            # Note: Very large K or r_pix may cause strong reflections at simulation boundaries.
            medium_t = kWaveMedium(
                sound_speed=c_t.astype(np.float32),
                density=rho_t.astype(np.float32),
                alpha_coeff=attenuation_map.astype(np.float32),
                alpha_power=1.5
            )

            for th in angles:
                # --- Build fresh source/sensor each run (k-Wave mutates masks) ---
                source = kSource()
                # ensure mask has correct shape and Fortran layout; dtype must be boolean
                amask = array_mask
                if amask.shape != (kgrid.Nx, kgrid.Ny):
                    if amask.shape == (kgrid.Ny, kgrid.Nx):
                        amask = amask.T
                    else:
                        raise ValueError(f"array_mask shape {amask.shape} incompatible with grid {(kgrid.Nx, kgrid.Ny)}")
                # Cast to bool and force Fortran order (column-major)
                amask = np.asfortranarray(amask.astype(np.bool_))
                source.p_mask = amask

                sensor = kSensor()
                sensor.mask = np.asfortranarray(amask.copy(order='F'))
                sensor.record = ['p']

                # 요소별 지연 (dt는 위에서 얻은 dt_actual 사용)
                d_pw = plane_wave_delays(yc, th, dy, 1500.0, dt_actual)

                # per-pixel 지연 (mask의 column-major 순서)
                idx_mask = np.flatnonzero(source.p_mask.ravel(order='F'))
                Ny_tmp = source.p_mask.shape[1]
                delay_by_col = np.zeros((Ny_tmp,), dtype=np.int64)
                for e in range(n_e):
                    y0, y1 = elem_cols[e,0], elem_cols[e,1]
                    delay_by_col[(y0-1):y1] = d_pw[e]
                _, cols_m = np.unravel_index(idx_mask, (Nx, Ny_tmp), order='F')
                offs_pix = delay_by_col[cols_m]    # [M]
                if offs_pix.size != np.count_nonzero(source.p_mask):
                    print(f"[dbg] offs_pix={offs_pix.size}, nnz(mask)={np.count_nonzero(source.p_mask)}")

                # 송신 파형(sig) — Nt 길이에 맞춤 (항상 1D 보장)
                sig = p_amp * tone_burst(1.0/dt_actual, freq, cycles)
                sig = np.asarray(sig, dtype=np.float32)
                sig = np.squeeze(sig)
                if sig.ndim != 1:
                    sig = sig.ravel()
                Lsig = sig.shape[0]
                if Lsig < Nt_actual:
                    sig = np.pad(sig, (0, Nt_actual - Lsig))
                else:
                    sig = sig[:Nt_actual]
                # sig: shape (Nt_actual,)

                # source.p initially built as [Nt × M] for convenience
                M = idx_mask.size
                P = np.zeros((Nt_actual, M), dtype=np.float32)
                for m in range(M):
                    d = int(max(0, offs_pix[m]))
                    ncopy = min(sig.size, Nt_actual - d)
                    if ncopy > 0:
                        P[d:d+ncopy, m] = sig[:ncopy]

                # --- Prepare for k-Wave: expects [num_series x Nt] = [M_mask x Nt] ---
                M_mask = int(np.count_nonzero(source.p_mask))
                Nt_k = int(kgrid.Nt)
                if P.shape[1] != M_mask:
                    raise ValueError(f"columns(P)={P.shape[1]} != nnz(mask)={M_mask}")
                if P.shape[0] != Nt_k:
                    # adjust to simulation Nt if needed
                    if P.shape[0] > Nt_k:
                        P = P[:Nt_k, :]
                    else:
                        P = np.pad(P, ((0, Nt_k - P.shape[0]), (0, 0)))

                # Transpose to [M_mask x Nt] and ensure float32 contiguous
                P_kw = np.asarray(P.T, dtype=np.float32, order='C')
                assert P_kw.shape == (M_mask, Nt_k), f"final source.p shape {P_kw.shape} != ({M_mask}, {Nt_k})"
                source.p = P_kw

                # 실행
                out = run2d(kgrid, source, sensor, medium_t, sim_opts, exec_opts)
                rf_pix = out['p']                 # [Nt, M]
                dt_out = float(out['dt'])

                # 픽셀→128채널 평균
                Nt_samp = rf_pix.shape[0]
                RF128 = np.zeros((Nt_samp, n_e), dtype=np.float32)
                pix_counts = (elem_cols[:,1] - elem_cols[:,0] + 1).astype(int)
                ptr = 0
                for e in range(n_e):
                    w = pix_counts[e]
                    RF128[:, e] = rf_pix[:, ptr:ptr+w].mean(axis=1)
                    ptr += w

                skull_id = os.path.splitext(os.path.basename(npy_path))[0]
                savemat(os.path.join(OUT_RF_DIR, f"{skull_id}_t{t:02d}_a{th:+03d}.mat"),
                        {'RF128': RF128, 'dt': dt_out, 'freq': freq, 'angles': np.array([th])},
                        do_compression=True)
            print(f"Frame {t}/{T} done.")


if __name__ == "__main__":
    main()
