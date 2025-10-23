
import numpy as np, math, json, time, matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from utils import (q_mul,q_conj,q_norm,q_from_omega,R_from_q,q_from_R,ang_error_deg,aitoff_xy,
                   build_angle_table, KDTree, lis_match, temporal_gating, quest, ransac_wahba, allan_deviation)
from filters import MEKFBias, UKFBias
from event_sim import synth_events, project as project_px

OUT = Path("outputs"); FR = OUT/"frames"
OUT.mkdir(parents=True, exist_ok=True); FR.mkdir(parents=True, exist_ok=True)

def rand_unit_sphere(n):
    u = np.random.rand(n); v = np.random.rand(n)
    th = 2*np.pi*u; ph = np.arccos(2*v-1)
    x = np.sin(ph)*np.cos(th); y = np.sin(ph)*np.sin(th); z = np.cos(ph)
    S = np.stack([x,y,z],axis=1); S = S/np.linalg.norm(S,axis=1,keepdims=True)
    return S, th, ph

def aitoff_plot(th, ph, save_to):
    theta = th - np.pi
    phi = (np.pi/2) - ph
    x,y = aitoff_xy(theta, phi)
    fig = plt.figure(figsize=(7,3.5))
    plt.scatter(x,y,s=1,alpha=0.5)
    plt.title("Sky catalog (Aitoff)")
    plt.grid(True); fig.tight_layout(); fig.savefig(save_to, dpi=170); plt.close(fig)

def psd_plot_multi(signals, fs, labels, fname, title):
    fig = plt.figure(figsize=(6,3.1))
    for s in signals:
        n = len(s); win = np.hanning(n)
        X = np.fft.rfft((s - np.mean(s))*win)
        freqs = np.fft.rfftfreq(n, d=1/fs)
        Pxx = (np.abs(X)**2) / (np.sum(win**2)*fs)
        plt.semilogx(freqs[1:], 10*np.log10(Pxx[1:]))
    plt.xlabel("Frequency (Hz)"); plt.ylabel("PSD (dB/Hz)"); plt.title(title)
    plt.legend(labels); plt.grid(True); fig.tight_layout(); fig.savefig(OUT/fname, dpi=170); plt.close(fig)

def cdf_plot_multi(data_list, labels, fname, xlabel="Error (deg)", title="CDF"):
    fig = plt.figure(figsize=(5.6,3.2))
    for d in data_list:
        s = np.sort(d); y = np.linspace(0,1,len(s))
        plt.plot(s,y)
    plt.xlabel(xlabel); plt.ylabel("CDF"); plt.title(title)
    plt.legend(labels); plt.grid(True); fig.tight_layout(); fig.savefig(OUT/fname, dpi=170); plt.close(fig)

def run_sim():
    np.random.seed(2)
    # Sky & catalog
    stars, th, ph = rand_unit_sphere(4000)
    ang_tab, tree = build_angle_table(stars, k=24)
    aitoff_plot(th, ph, OUT/"sky_aitoff.png")

    # Camera & trajectory
    W,H,focal = 1280, 960, 950
    fov = 18.0
    T, dt = 18.0, 0.02
    steps = int(T/dt)
    w_bias_true = np.deg2rad(np.array([0.08, -0.05, 0.07]))/60
    w_true = np.vstack([
        0.02*np.sin(np.linspace(0,6*np.pi,steps)),
        0.015*np.cos(np.linspace(0,4*np.pi,steps)),
        np.linspace(0.0, 0.03, steps)
    ]).T
    q = np.array([1,0,0,0],dtype=float)
    q_hist = []
    for k in range(steps):
        q = q_norm(q_mul(q, q_from_omega(w_true[k], dt)))
        q_hist.append(q.copy())
    q_hist = np.stack(q_hist)

    gyro_noise = np.deg2rad(0.02)
    w_meas = w_true + w_bias_true + gyro_noise*np.random.randn(steps,3)

    # APS pipeline (frame-based) with robust LIS + RANSAC-QUEST; then MEKF/UKF
    mekf = MEKFBias(); ukf = UKFBias()
    err_mekf=[]; err_ukf=[]; prev_ids=None

    for k in range(steps):
        Rtrue = R_from_q(q_hist[k])
        idx, uv, camv = project_px(Rtrue, stars, focal=focal, W=W, H=H, fov_deg=fov)
        if len(idx) < 12: continue
        center = np.array([W/2,H/2])
        sel = np.argsort(np.linalg.norm(uv-center,axis=1))[:30]
        obs = camv[sel]/np.linalg.norm(camv[sel],axis=1,keepdims=True)

        # Robust LIS via angle-table + KD-Tree + temporal gating
        cand = lis_match(obs, stars, ang_tab, tree, ang_tol=np.deg2rad(0.05))
        if cand is None: 
            cand = np.random.choice(stars.shape[0], size=12, replace=False).tolist()
        cand = temporal_gating(prev_ids, cand); prev_ids = cand
        ref = stars[cand[:min(len(cand), obs.shape[0])]]
        obs_use = obs[:ref.shape[0]]

        R_meas, inl = ransac_wahba(obs_use, ref, iters=200, thresh=np.deg2rad(0.04))

        # Filters
        mekf.step(w_meas[k], R_meas, dt)
        ukf.step(w_meas[k], R_meas, dt)
        err_mekf.append(ang_error_deg(R_from_q(mekf.q), Rtrue))
        err_ukf.append(ang_error_deg(R_from_q(ukf.q), Rtrue))

        # demo frames (sparser)
        if k % 6 == 0:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(5.0,3.8))
            plt.scatter(uv[:,0], uv[:,1], s=5, alpha=0.4)
            det = uv[sel]
            plt.scatter(det[:,0], det[:,1], s=12)
            plt.gca().invert_yaxis(); plt.title(f"APS Frame {k} (inliers~{len(inl)})")
            fig.tight_layout(); fig.savefig(OUT/f"frame_aps_{k:04d}.png", dpi=120); plt.close(fig)

    # Build GIF (APS)
    frames = sorted(OUT.glob("frame_aps_*.png"))
    imgs = [Image.open(p) for p in frames[:120]]
    if imgs: imgs[0].save(OUT/"demo_aps.gif", save_all=True, append_images=imgs[1:], duration=80, loop=0)

    err_mekf = np.array(err_mekf); err_ukf = np.array(err_ukf)

    # UKF vs MEKF comparison plots
    cdf_plot_multi([err_mekf, err_ukf], ["MEKF","UKF"], "cdf_mekf_vs_ukf.png", "Error (deg)", "CDF: MEKF vs UKF")
    fs = 1.0/dt
    psd_plot_multi([err_mekf, err_ukf], fs, ["MEKF","UKF"], "psd_mekf_vs_ukf.png", "PSD: MEKF vs UKF")

    # ---------- Event-based pipeline ----------
    # Higher-rate attitude with small windows
    dt_ev = 0.005
    steps_ev = int(T/dt_ev)
    # Interpolate true attitude for event timeline
    def slerp(q1,q2,t):
        dot = np.clip(np.dot(q1,q2),-1,1)
        if dot < 0: q2=-q2; dot=-dot
        if dot > 0.9995: return q_norm(q1 + t*(q2-q1))
        th = np.arccos(dot); s1 = np.sin((1-t)*th)/np.sin(th); s2 = np.sin(t*th)/np.sin(th)
        return q1*s1 + q2*s2
    q_hist_ev = []
    for k in range(steps_ev):
        s = k*dt_ev/T
        idxf = s*(steps-1)
        i0 = int(np.floor(idxf)); i1 = min(i0+1, steps-1)
        t = idxf - i0
        q_hist_ev.append(slerp(q_hist[i0], q_hist[i1], t))
    q_hist_ev = np.stack(q_hist_ev)

    events = synth_events(q_hist_ev, stars, dt_ev, focal=focal, W=W,H=H, fov=fov, C=0.25, leak=0.01)

    # Process events in short windows to produce attitude updates (EBS-EKF-like)
    eb_q = np.array([1,0,0,0],dtype=float)
    eb_err=[]; eb_times=[]; eb_frames=[]
    win = int(0.01/dt_ev)  # 10ms window
    from utils import quest
    prev_ids=None
    for i in range(0, len(events), max(1,win)):
        batch = events[i:i+win]
        if not batch: continue
        t_mid = batch[len(batch)//2][0]
        # cluster events by coarse grid -> centroids (toy)
        arr = np.array([[b[1],b[2]] for b in batch])
        if arr.size == 0: continue
        # simple grid cluster
        g = (arr/6).astype(int)
        uniq = np.unique(g, axis=0)
        cent = []
        for u in uniq:
            mask = np.all(g==u, axis=1)
            cent.append(arr[mask].mean(axis=0))
        cent = np.array(cent)
        # back-project to unit directions (assume z=1)
        # camera model: x = (u - W/2)/focal, y = (v - H/2)/focal, z=1 -> normalize
        dirs = np.stack([(cent[:,0]-W/2)/focal, (cent[:,1]-H/2)/focal, np.ones(len(cent))], axis=1)
        dirs = dirs/np.linalg.norm(dirs,axis=1,keepdims=True)

        # Predict catalog match using LIS robust matcher (reuse angle table)
        cand = lis_match(dirs, stars, ang_tab, tree, ang_tol=np.deg2rad(0.08))
        if cand is None: continue
        cand = temporal_gating(prev_ids, cand); prev_ids = cand
        ref = stars[cand[:dirs.shape[0]]]
        obs_use = dirs[:ref.shape[0]]

        R_meas, inl = ransac_wahba(obs_use, ref, iters=100, thresh=np.deg2rad(0.05))
        # "update" attitude directly to R_meas for visualization (could also run MEKF at dt_ev)
        from utils import q_from_R
        eb_q = q_from_R(R_meas)
        # evaluate against truth at t_mid
        kf = min(int(t_mid/dt * (steps-1)), steps-1)
        Rtrue = R_from_q(q_hist[kf])
        eb_err.append(ang_error_deg(R_meas, Rtrue))
        eb_times.append(t_mid)

        # light-weight frame for GIF
        if len(eb_frames) < 120:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(4.6,3.6))
            plt.scatter(arr[:,0], arr[:,1], s=4, alpha=0.5)
            if len(cent)>0: plt.scatter(cent[:,0], cent[:,1], s=18)
            plt.gca().invert_yaxis(); plt.title(f"Events window @ {t_mid:.2f}s, inliers~{len(inl)}")
            fig.tight_layout(); fn = OUT/f"frame_event_{len(eb_frames):04d}.png"
            fig.savefig(fn, dpi=120); plt.close(fig); eb_frames.append(fn)

    # Build GIF (Events)
    imgs = [Image.open(p) for p in eb_frames]
    if imgs: imgs[0].save(OUT/"demo_event.gif", save_all=True, append_images=imgs[1:], duration=60, loop=0)

    # Compare Event vs APS
    err_ev = np.array(eb_err)
    # For APS baseline, resample err_mekf to event times (nearest)
    aps_ts = np.linspace(0, T, len(err_mekf))
    ev_ts = np.array(eb_times)
    if len(ev_ts)>5 and len(aps_ts)>5:
        # simple nearest mapping
        aps_err_resamp = np.interp(ev_ts, aps_ts, err_mekf[:len(aps_ts)])
        cdf_plot_multi([aps_err_resamp, err_ev], ["APS(MEKF)","Events(EBS)"],
                       "cdf_event_vs_aps.png", "Error (deg)", "CDF: APS vs Events")
        # PSD (same length)
        nmin = min(len(aps_err_resamp), len(err_ev))
        from math import isfinite
        aps_cut = aps_err_resamp[:nmin]
        ev_cut  = err_ev[:nmin]
        fs_ev = 1.0/np.median(np.diff(ev_ts)) if len(ev_ts)>1 else 1.0/dt_ev
        psd_plot_multi([aps_cut, ev_cut], fs_ev, ["APS","Events"], "psd_event_vs_aps.png", "PSD: APS vs Events")
        # Latency estimate via xcorr (find lag at max corr)
        def lag_ms(a,b,fs):
            a = a - np.mean(a); b = b - np.mean(b)
            c = np.correlate(a, b, mode="full")
            lags = np.arange(-len(a)+1, len(a))
            L = lags[np.argmax(c)]
            return 1000.0*L/fs
        with open(OUT/"latency_event_vs_aps.txt","w") as f:
            f.write(f"Estimated latency (ms, Events relative to APS): {lag_ms(aps_cut, ev_cut, fs_ev):.2f}\n")

    # Dump metrics
    mets = {
        "MEKF_RMSE_deg": float(np.sqrt(np.mean(err_mekf**2))),
        "UKF_RMSE_deg":  float(np.sqrt(np.mean(err_ukf**2))),
        "MEKF_median_deg": float(np.median(err_mekf)),
        "UKF_median_deg":  float(np.median(err_ukf)),
        "Events_median_deg": float(np.median(err_ev)) if len(eb_err)>0 else None
    }
    (OUT/"metrics.json").write_text(json.dumps(mets, indent=2))

if __name__ == "__main__":
    run_sim()
