# sim.py — PWI-ULM (k-Wave-python, 128ch, skull HU→물성, random-MB, CC/H5 export)
# deps: numpy, scipy, h5py, k-wave-python
import os, json, argparse
import numpy as np
import h5py
from scipy.io import savemat

# ---- k-Wave-python API ----
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.kspaceFirstOrder2D import kspace_first_order_2d_gpu, kspaceFirstOrder2DC
from kwave.utils.signals import tone_burst
from kwave.utils.conversion import hounsfield2density

# -------------------------- 기본 파라미터 (기본값, CLI로 변경 가능) --------------------------
Lx_cm = 1.024
Ly_cm = 1.024
gs_mm = 0.02                 # 512 × 512 at 0.02 mm/px
freq_default = 0.5e6
cycles_default = 5
angles_default = [0]         # 예: [-10,-5,0,5,10]
T_default = 16
n_e = 128
td_loc = (0.30, Lx_cm/2)     # (x[cm], y[cm]) 선형 어레이 x위치, y중앙
pitch_mm, elem_w_mm = 0.30, 0.25
p_amp = 1e5                  # [Pa]
seed_default = 1234

# MB(가스) 물성
BUBBLE_C   = 340.0           # m/s
BUBBLE_RHO = 1.2             # kg/m^3
MB_RADIUS_PIX_DEFAULT = 1.0
MB_EDGE_SOFTNESS_PIX  = 0.75

HU_MIN = -1024.0
HU_MAX = 3500.0

# -------------------------- 유틸 --------------------------
def makeLinearArrayMask(Nx, Ny, td_x, y_center, n_e, elem_w_pix, pitch_pix):
    mask = np.zeros((Nx, Ny), dtype=bool)
    col0 = int(round(y_center - (n_e*pitch_pix - (pitch_pix-elem_w_pix))/2))
    elem_cols = np.zeros((n_e, 2), dtype=int)
    for e in range(n_e):
        y0 = col0 + e*pitch_pix
        y1 = y0 + elem_w_pix - 1
        y0 = max(1, y0); y1 = min(Ny, y1)
        mask[td_x-1, (y0-1):y1] = True
        elem_cols[e,:] = [y0, y1]
    yc = np.rint(np.mean(elem_cols, axis=1)).astype(int)
    return mask, elem_cols, yc

def plane_wave_delays(yc_pix, theta_deg, dy_m, c_ref, dt):
    y = (yc_pix.astype(float) - 1.0) * dy_m
    tau = (y * np.sin(np.deg2rad(theta_deg))) / c_ref
    d_samp = np.round(tau / dt).astype(np.int64)
    dmin = d_samp.min()
    if dmin < 0:
        d_samp = d_samp - dmin
    return d_samp.astype(np.int64)

def randsample_fast(vec_idx, K, replace=True, rng=None):
    if rng is None: rng = np.random.default_rng()
    if replace:
        idx = rng.integers(low=0, high=vec_idx.size, size=K)
        return vec_idx[idx]
    else:
        if K > vec_idx.size: raise ValueError("K>n without replacement")
        idx = rng.choice(vec_idx.size, size=K, replace=False)
        return vec_idx[idx]

def sample_speed_pix_per_frame(rng, K):
    # 혼합분포(느림/보통/빠름) in 픽셀/프레임
    bins = rng.choice([0,1,2], size=K, p=[0.70,0.25,0.05])
    v = np.empty(K, dtype=float)
    v[bins==0] = 0.02 + (0.20-0.02)*rng.random(np.sum(bins==0))
    v[bins==1] = 0.20 + (0.60-0.20)*rng.random(np.sum(bins==1))
    v[bins==2] = 0.60 + (1.50-0.60)*rng.random(np.sum(bins==2))
    return v

def make_mb_tracks_rand_avoid_bone(mask_water, T, K, rng=None):
    """
    랜덤 MB 트랙(T 프레임, K 버블). 연조직 픽셀에서만 이동.
    경계/해골 진입 시 방향 재샘플, 끝까지 실패하면 해당 프레임 정지.
    반환: (tracks_xy, tracks_amp)
      - tracks_xy: 길이 T 리스트, 각 원소 shape=(K,2) [1-based 픽셀 좌표(int)]
      - tracks_amp: 길이 T 리스트, 각 원소 shape=(K,)  [상대 강도(float)]
    """
    import numpy as np
    if rng is None:
        rng = np.random.default_rng()
    Nx, Ny = mask_water.shape

    # 가장자리 1픽셀 금지(경계/인덱싱 안전)
    allowed = mask_water.astype(bool).copy()
    allowed[0,:] = allowed[-1,:] = False
    allowed[:,0] = allowed[:,-1] = False

    allowed_idx = np.flatnonzero(allowed.ravel(order='F'))
    if allowed_idx.size == 0:
        raise ValueError("No allowed pixels (mask_water all False)")

    # 초기 위치 샘플링(1-based float 좌표 유지)
    sel = randsample_fast(allowed_idx, K, True, rng)
    x, y = np.unravel_index(sel, (Nx, Ny), order='F')
    x = x.astype(float) + 1.0
    y = y.astype(float) + 1.0

    # 속도/방향
    v  = sample_speed_pix_per_frame(rng, K)
    th = 2*np.pi*rng.random(K)
    vx = v*np.cos(th); vy = v*np.sin(th)

    # 버블 강도(상대값, 프레임마다 동일)
    amp0 = 0.8 + 0.4*rng.random(K)

    tracks_xy, tracks_amp = [], []
    for _ in range(T):
        # 기록(반올림해 1-based int 좌표 저장)
        tracks_xy.append(np.column_stack([np.rint(x).astype(int), np.rint(y).astype(int)]))
        tracks_amp.append(amp0.copy())

        # 다음 위치 제안
        x_new = x + vx
        y_new = y + vy

        # 1) 먼저 경계 체크 → in-bounds만 allowed 조회 (floor 사용)
        ix = np.floor(x_new).astype(int) - 1
        iy = np.floor(y_new).astype(int) - 1
        inb = (ix >= 0) & (ix < Nx) & (iy >= 0) & (iy < Ny)

        invalid = ~inb.copy()
        if np.any(inb):
            idx_inb = np.flatnonzero(inb)
            bad = ~allowed[ix[idx_inb], iy[idx_inb]]
            invalid[idx_inb[bad]] = True

        if np.any(invalid):
            MAX_TRIES = 8
            for _ in range(MAX_TRIES):
                idx = np.flatnonzero(invalid)
                if idx.size == 0:
                    break
                th = 2*np.pi*rng.random(idx.size)
                vx[idx] = v[idx]*np.cos(th); vy[idx] = v[idx]*np.sin(th)
                x_new[idx] = x[idx] + vx[idx]
                y_new[idx] = y[idx] + vy[idx]

                ix = np.floor(x_new[idx]).astype(int) - 1
                iy = np.floor(y_new[idx]).astype(int) - 1
                inb = (ix >= 0) & (ix < Nx) & (iy >= 0) & (iy < Ny)

                invalid[idx] = ~inb
                if np.any(inb):
                    sub = np.flatnonzero(inb)
                    bad = ~allowed[ix[sub], iy[sub]]
                    invalid[idx[sub[bad]]] = True

            # 그래도 invalid → 해당 프레임은 정지
            if np.any(invalid):
                stuck = np.flatnonzero(invalid)
                x_new[stuck] = x[stuck]
                y_new[stuck] = y[stuck]

        x, y = x_new, y_new

    return tracks_xy, tracks_amp

def apply_mb_disks_physical(c, rho, centers, amp=None, r_pix=MB_RADIUS_PIX_DEFAULT, edge_soft=MB_EDGE_SOFTNESS_PIX):
    """
    MB 디스크를 부드럽게 주입하여 c, rho 수정.
    amp가 주어지면 중심에서 BUBBLE 물성으로 더 강하게 blend (선형 가중).
    """
    c2 = c.copy(); rho2 = rho.copy()
    Nx, Ny = c.shape
    X, Y = np.mgrid[1:Nx+1, 1:Ny+1]
    r_pix = float(max(0.5, r_pix)); edge = float(max(0.0, edge_soft))
    if amp is None:
        amp = np.ones(len(centers), dtype=float)
    for (cx, cy), a in zip(centers, amp):
        R = np.sqrt((X - cx)**2 + (Y - cy)**2)
        if edge > 1e-6:
            w = 0.5*(1.0 + np.cos(np.clip((R - r_pix)/edge, 0.0, 1.0)*np.pi))
            w[R <= r_pix] = 1.0; w[R >= r_pix + edge] = 0.0
        else:
            w = (R <= r_pix).astype(float)
        w = np.clip(w*a, 0.0, 1.0)
        c2  = w*BUBBLE_C   + (1.0 - w)*c2
        rho2= w*BUBBLE_RHO + (1.0 - w)*rho2
    return c2, rho2

def run2d(kgrid, source, sensor, medium, sim_opts, exec_opts):
    """
    Prefer GPU via CUDA binary; if it fails and KSPACE_GPU_ONLY!=1, fallback to CPU.
    To force GPU-only (no fallback), set env KSPACE_GPU_ONLY=1 or run with --gpu_only.
    """
    try:
        exec_opts.is_gpu_simulation = True
        return kspace_first_order_2d_gpu(kgrid, source, sensor, medium, sim_opts, exec_opts)
    except Exception as e:
        # GPU path failed
        if os.getenv("KSPACE_GPU_ONLY", "0") == "1":
            raise
        exec_opts.is_gpu_simulation = False
        return kspaceFirstOrder2DC(kgrid, source, sensor, medium, sim_opts, exec_opts)

# -------------------------- HU -> 물성 매핑 --------------------------
def hu_to_acoustics(hu, mode='piecewise'):
    """
    hu -> (rho, c, alpha_db_cm_mhz)
    mode='piecewise' : 권장 시작점 (연조직/해면골/피질골)
    """
    hu_clip = np.clip(hu, 0, None)
    # 밀도: k-Wave 기본 hounsfield2density 사용(물 ~ 997 kg/m3)
    rho = hounsfield2density(hu_clip).astype(np.float32)
    # 속도: 구간형(안정)
    c = np.where(hu_clip<50, 1540.0,
         np.where(hu_clip<300, 2200.0, 2900.0)).astype(np.float32)
    # 감쇠: dB/cm/MHz
    alpha = np.where(hu_clip<50, 0.5,
            np.where(hu_clip<300, 5.0, 15.0)).astype(np.float32)
    # 물/연조직 최소값 보정
    rho[rho < 997.0] = 997.0
    return rho, c, alpha

# -------------------------- CC/SIMUS 연계용 H5 내보내기 --------------------------
def save_tracks_h5_for_cc(h5_path, tracks_xy, tracks_amp, dx, dy, fps):
    """
    CC(또는 다른 엔진)에서 산란원으로 쉽게 읽도록 H5로 저장.
    좌표는 [m]로 저장. 그룹:
      /tracks/frame_tXX : [K,2] (x_pix, y_pix)
      /tracks/amp_tXX   : [K,]
      /cc_scatterers/frame_tXX : [K,3] (x[m], y[m], amp)
    """
    T = len(tracks_xy)
    with h5py.File(h5_path, 'w') as f:
        g_tr = f.create_group('tracks')
        g_cc = f.create_group('cc_scatterers')
        for t in range(T):
            xy = tracks_xy[t].astype(np.int32)
            amp = np.asarray(tracks_amp[t], dtype=np.float32)
            g_tr.create_dataset(f'frame_t{t+1:02d}', data=xy, compression="gzip")
            g_tr.create_dataset(f'amp_t{t+1:02d}', data=amp, compression="gzip")
            # pixel -> meters (센터 기준 1-based를 0-base로 교정)
            x_m = (xy[:,0].astype(float)-1.0)*dx
            y_m = (xy[:,1].astype(float)-1.0)*dy
            cc_arr = np.column_stack([x_m, y_m, amp]).astype(np.float32)
            g_cc.create_dataset(f'frame_t{t+1:02d}', data=cc_arr, compression="gzip")
        f.attrs['dx'] = dx; f.attrs['dy'] = dy; f.attrs['fps'] = fps

# -------------------------- main --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_dir', default='./00/02')
    ap.add_argument('--out_dir', default='./outputs/out_rf')
    ap.add_argument('--angles', default='0', help='comma sep, e.g. "-10,-5,0,5,10"')
    ap.add_argument('--freq', type=float, default=freq_default)
    ap.add_argument('--cycles', type=int, default=cycles_default)
    ap.add_argument('--T', type=int, default=T_default)
    ap.add_argument('--K', type=int, default=100, help='num microbubbles')
    ap.add_argument('--seed', type=int, default=seed_default)
    ap.add_argument('--input_is_hu', action='store_true', help='treat .npy as HU already')
    ap.add_argument('--mb_radius_pix', type=float, default=MB_RADIUS_PIX_DEFAULT)
    ap.add_argument('--mb_edge_soft', type=float, default=MB_EDGE_SOFTNESS_PIX)
    ap.add_argument('--gpu_only', action='store_true', help='GPU만 사용(실패 시 중단, CPU 폴백 금지)')
    ap.add_argument('--gpu_id', type=str, default='0', help='CUDA_VISIBLE_DEVICES 설정값 (예: "0" 또는 "0,1")')
    args = ap.parse_args()

    # GPU 환경 설정
    if args.gpu_only:
        os.environ["KSPACE_GPU_ONLY"] = "1"
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # 수집
    npy_files = []
    for root, _, files in os.walk(args.in_dir):
        for f in files:
            if f.endswith('.npy'):
                npy_files.append(os.path.join(root, f))
    npy_files = sorted(npy_files)
    assert npy_files, f"입력 폴더 {args.in_dir} 하위에 .npy 파일이 없습니다."

    # 각도 파싱
    try:
        angles = [int(s) for s in args.angles.split(',')]
    except Exception:
        raise ValueError(f"--angles 파싱 실패: {args.angles}")

    # 메타 저장
    meta = {
        'freq': args.freq, 'cycles': args.cycles, 'angles': angles,
        'T': args.T, 'K': args.K, 'seed': args.seed,
        'pitch_mm': pitch_mm, 'elem_w_mm': elem_w_mm,
        'Lx_cm': Lx_cm, 'Ly_cm': Ly_cm, 'gs_mm': gs_mm
    }
    with open(os.path.join(args.out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # 메인 루프
    for npy_path in npy_files:
        arr = np.load(npy_path).astype(np.float32)

        Nx = int(round(Lx_cm / (gs_mm/10.0)))
        Ny = int(round(Ly_cm / (gs_mm/10.0)))
        assert arr.shape == (Nx, Ny), f"skull 크기 {arr.shape} != ({Nx},{Ny})"
        print(f"[load] {npy_path} size={arr.shape}")

        dx = dy = gs_mm * 1e-3

        # HU 판별/매핑
        if args.input_is_hu:
            hu = arr
        else:
            vmin, vmax = float(arr.min()), float(arr.max())
            if (vmin >= -1.5 and vmax <= 1.5):
                # [-1,1] → HU 역변환
                hu = ((np.clip(arr, -1.0, 1.0) + 1.0)*0.5)*(HU_MAX - HU_MIN) + HU_MIN
            else:
                # 이미 HU로 판단
                hu = arr

        # === Grid ===
        kgrid = kWaveGrid((Nx, Ny), (dx, dy))

        # === Medium (HU -> rho, c, alpha) ===
        rho_map, c_map, alpha_db = hu_to_acoustics(hu, mode='piecewise')
        medium_base = kWaveMedium(
            sound_speed=c_map.astype(np.float32),
            density=rho_map.astype(np.float32),
            alpha_coeff=alpha_db.astype(np.float32),
            alpha_power=1.5
        )

        # === 배열 마스크 ===
        pitch_pix  = max(1, int(round(pitch_mm/gs_mm)))
        elem_w_pix = max(1, int(round(elem_w_mm/gs_mm)))
        td_x = int(round(td_loc[0] / (gs_mm/10.0)))
        y_center = int(round(td_loc[1] / (gs_mm/10.0)))
        # Precompute (reference only); actual mask will be rebuilt per-iteration using kgrid sizes
        array_mask_pre, elem_cols_pre, yc_pre = makeLinearArrayMask(Nx, Ny, td_x, y_center, n_e, elem_w_pix, pitch_pix)

        # === 시간 ===
        c0 = float(np.minimum(c_map.min(), 1500.0))
        CFL = 0.07
        depth_m = Lx_cm * 1e-2
        t_end = 2*depth_m/c0 * 1.2
        kgrid.makeTime(c0, CFL, t_end)
        dt_actual = float(kgrid.dt)
        Nt_actual = int(kgrid.Nt)

        sim_opts = SimulationOptions(
            pml_inside=False, pml_size=20, pml_alpha=2.0,
            data_cast='single', save_to_disk=True
        )
        exec_opts = SimulationExecutionOptions(is_gpu_simulation=True)

        # === MB 트랙(연조직에서만 스폰) ===
        mask_soft = (hu < 50.0)  # 연조직/물 영역
        K = args.K
        tracks_xy, tracks_amp = make_mb_tracks_rand_avoid_bone(mask_soft, args.T, K, rng=rng)

        # === CC/SIMUS 연계용 트랙 H5 저장(프레임별 [x(m),y(m),amp]) ===
        h5_out = os.path.join(args.out_dir, f"tracks_cc_{os.path.splitext(os.path.basename(npy_path))[0]}.h5")
        fps = 1000.0  # 가정(원하면 계산/인자로 넘길 수 있음)
        save_tracks_h5_for_cc(h5_out, tracks_xy, tracks_amp, dx, dy, fps)

        skull_id = os.path.splitext(os.path.basename(npy_path))[0]

        # === 프레임/각도 루프 ===
        for t in range(1, args.T+1):
            centers = tracks_xy[t-1]
            amps_t  = tracks_amp[t-1]
            c_t, rho_t = apply_mb_disks_physical(c_map, rho_map, centers,
                                                 amp=amps_t,
                                                 r_pix=args.mb_radius_pix,
                                                 edge_soft=args.mb_edge_soft)
            medium_t = kWaveMedium(
                sound_speed=c_t.astype(np.float32),
                density=rho_t.astype(np.float32),
                alpha_coeff=alpha_db.astype(np.float32),
                alpha_power=1.5
            )

            for th in angles:
                # --- source/sensor fresh (k-Wave가 마스크를 변형하므로) ---
                # Rebuild array mask using the final kgrid dimensions (including PML)
                amask, elem_cols_curr, yc_curr = makeLinearArrayMask(
                    kgrid.Nx, kgrid.Ny, td_x, y_center, n_e, elem_w_pix, pitch_pix
                )

                # Create source and sensor objects
                source = kSource()
                sensor = kSensor()

                # Assign the mask. k-Wave requires Fortran-contiguous arrays for masks.
                # The mask from makeLinearArrayMask should already have the correct shape (kgrid.Nx, kgrid.Ny).
                source.p_mask = np.asfortranarray(amask)
                sensor.mask = np.asfortranarray(amask.copy()) # Use a copy for the sensor
                sensor.record = ['p']

                # Safety assertions (debug)
                assert source.p_mask.shape == (kgrid.Nx, kgrid.Ny), f"{source.p_mask.shape} vs {(kgrid.Nx, kgrid.Ny)}"
                assert source.p_mask.dtype == np.bool_, source.p_mask.dtype

                # 요소 지연
                d_pw = plane_wave_delays(yc_curr, th, dy, 1500.0, dt_actual)

                # per-pixel 지연
                idx_mask = np.flatnonzero(source.p_mask.ravel(order='F'))
                Ny_tmp = source.p_mask.shape[1]
                delay_by_col = np.zeros((Ny_tmp,), dtype=np.int64)
                for e in range(n_e):
                    y0, y1 = elem_cols_curr[e,0], elem_cols_curr[e,1]
                    delay_by_col[(y0-1):y1] = d_pw[e]
                _, cols_m = np.unravel_index(idx_mask, (kgrid.Nx, Ny_tmp), order='F')
                offs_pix = delay_by_col[cols_m]

                # 송신 파형
                sig = p_amp * tone_burst(1.0/dt_actual, args.freq, args.cycles)
                sig = np.asarray(np.squeeze(sig), dtype=np.float32)
                if sig.ndim != 1: sig = sig.ravel()
                if sig.size < Nt_actual: sig = np.pad(sig, (0, Nt_actual - sig.size))
                else: sig = sig[:Nt_actual]

                # [Nt x M]
                M = idx_mask.size
                P = np.zeros((Nt_actual, M), dtype=np.float32)
                for m in range(M):
                    d = int(max(0, offs_pix[m]))
                    ncopy = min(sig.size, Nt_actual - d)
                    if ncopy > 0:
                        P[d:d+ncopy, m] = sig[:ncopy]

                # k-Wave 요구 [M x Nt]
                P_kw = np.asarray(P.T, dtype=np.float32, order='C')
                assert P_kw.shape == (int(np.count_nonzero(source.p_mask)), int(kgrid.Nt))
                source.p = P_kw

                # 실행
                out = run2d(kgrid, source, sensor, medium_t, sim_opts, exec_opts)
                rf_pix = out['p']                 # [Nt, M]
                dt_out = float(out['dt'])

                # 픽셀→128채널 평균
                Nt_samp = rf_pix.shape[0]
                RF128 = np.zeros((Nt_samp, n_e), dtype=np.float32)
                pix_counts = (elem_cols_curr[:,1] - elem_cols_curr[:,0] + 1).astype(int)
                ptr = 0
                for e in range(n_e):
                    w = pix_counts[e]
                    RF128[:, e] = rf_pix[:, ptr:ptr+w].mean(axis=1)
                    ptr += w

                savemat(
                    os.path.join(args.out_dir, f"{skull_id}_t{t:02d}_a{th:+03d}.mat"),
                    {'RF128': RF128, 'dt': dt_out, 'freq': args.freq, 'angles': np.array([th])},
                    do_compression=True
                )
            print(f"[done] {skull_id} frame {t}/{args.T}")

if __name__ == "__main__":
    main()