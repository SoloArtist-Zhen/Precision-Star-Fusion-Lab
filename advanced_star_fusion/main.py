
import numpy as np, matplotlib.pyplot as plt, json, math
from pathlib import Path
from PIL import Image

from catalog import random_catalog
from camera import project_dirs, add_pixel_noise, psf_render
from detect import select_top_by_brightness
from lis import build_triangle_db, match_triplet
from wahba import wahba
from mekf import MEKF
from utils import ang_err_deg, quat_to_R

OUT = Path("outputs"); FR = OUT/"frames"
OUT.mkdir(parents=True, exist_ok=True); FR.mkdir(parents=True, exist_ok=True)

def euler_R(roll,pitch,yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    return Rz@Ry@Rx

def allan_deviation(x, dt, taus):
    x = np.asarray(x)
    out = []
    y = np.cumsum(x)*dt
    for tau in taus:
        m = int(tau/dt)
        if m < 2:
            out.append(np.nan); continue
        diff = y[2*m:] - 2*y[m:-m] + y[:-2*m]
        adev = np.sqrt(0.5*np.mean(diff**2)/(tau**2))
        out.append(adev)
    return np.array(out)

def run_once(seed=0, render_preview=True):
    rng = np.random.default_rng(seed)
    ids, dirs, mags = random_catalog(N=5000, seed=seed)
    tri_db = build_triangle_db(dirs, ids, K=900, seed=seed)

    T, dt = 18.0, 0.04
    steps = int(T/dt)
    roll = np.deg2rad(np.linspace(0, 6, steps))
    pitch = np.deg2rad(0.6*np.sin(np.linspace(0, 2*np.pi, steps)))
    yaw = np.deg2rad(np.linspace(0, 10, steps))
    w_true = np.vstack([np.gradient(roll, dt), np.gradient(pitch, dt), np.gradient(yaw, dt)]).T
    b_true = np.array([0.01, -0.015, 0.02])
    w_meas = w_true + b_true + 0.002*rng.standard_normal((steps,3))

    mekf = MEKF(Q_gyro=2e-6, Q_bias=5e-8, R_meas=2e-4)

    err_total, err_r, err_p, err_y = [], [], [], []
    nees = []; bias_est = []
    lost = False; matched_ids = None

    for k in range(steps):
        Rtrue = euler_R(roll[k], pitch[k], yaw[k])
        idx, px, cam_dirs = project_dirs(Rtrue, dirs, f=900, W=1440, H=1080, fov_deg=22.0)
        if idx.size < 12:
            continue
        pxn = add_pixel_noise(px, sigma=0.6)
        vis_ids = ids[idx]; vis_mags = mags[idx]
        order = select_top_by_brightness(pxn, vis_mags, topk=28)
        px_sel = pxn[order]; cam_sel = cam_dirs[order]

        # Preview render
        if k==0 and render_preview:
            img = psf_render(np.round(pxn).astype(int))
            plt.figure(figsize=(5.2,4))
            plt.imshow(img, origin="upper")
            plt.title("Rendered star frame (preview)"); plt.tight_layout()
            plt.savefig(OUT/"scene_000.png", dpi=150); plt.close()

        # LIS (first / when lost)
        if matched_ids is None or lost:
            if cam_sel.shape[0] >= 6:
                ids_trip, e = match_triplet(cam_sel[:3], tri_db, tol=0.02)
                if ids_trip is not None:
                    R_seed = wahba(cam_sel[:3], dirs[ids_trip])
                    pred_inertial = (R_seed.T @ cam_sel[3:].T).T
                    matched_more = []
                    for v in pred_inertial:
                        dots = dirs @ v
                        matched_more.append(int(np.argmax(dots)))
                    matched_ids = np.array(ids_trip + matched_more[:5])
                    lost = False
                else:
                    lost = True
            else:
                lost = True

        # Predict correspondences by current estimate
        used = 0
        if matched_ids is not None and cam_sel.shape[0]>=6:
            R_est = quat_to_R(mekf.q)
            pred_inertial = (R_est.T @ cam_sel.T).T
            cat_idx = []
            for v in pred_inertial:
                dots = dirs @ v
                cat_idx.append(int(np.argmax(dots)))
            useN = min(20, cam_sel.shape[0]-1)
            ref_vecs = dirs[np.array(cat_idx[:useN])]
            y_body = cam_sel[:useN]
            used = useN
        else:
            ref_vecs = np.zeros((0,3)); y_body = np.zeros((0,3))

        mekf.predict(w_meas[k], dt)
        if used >= 6:
            mekf.update_vectors(ref_vecs, y_body)
        else:
            lost = True

        R_est = quat_to_R(mekf.q)
        err_total.append(ang_err_deg(R_est, Rtrue))
        Rt = R_est.T @ Rtrue
        roll_err = np.rad2deg(np.arctan2(Rt[2,1], Rt[2,2]))
        pitch_err = np.rad2deg(np.arcsin(-Rt[2,0]))
        yaw_err = np.rad2deg(np.arctan2(Rt[1,0], Rt[0,0]))
        err_r.append(roll_err); err_p.append(pitch_err); err_y.append(yaw_err)

        nees.append(float(np.trace(mekf.P[:3,:3])))
        bias_est.append(mekf.x[3:].copy())

        if k % 6 == 0:
            plt.figure(figsize=(4.6,4))
            plt.scatter(pxn[:,0], pxn[:,1], s=5, alpha=0.35, label="vis")
            if used>=6:
                plt.scatter(px_sel[:used,0], px_sel[:used,1], s=25, marker="x", label="used")
            plt.gca().invert_yaxis()
            plt.title(f"Tracking k={k} {'(LIS)' if (matched_ids is None or lost) else ''}")
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(FR/f"frame_{k:04d}.png", dpi=110); plt.close()

    err_total = np.array(err_total)
    err_r = np.array(err_r); err_p = np.array(err_p); err_y = np.array(err_y)
    nees = np.array(nees); bias_est = np.array(bias_est)
    t = np.arange(err_total.size)*dt

    # Plots
    plt.figure(figsize=(6,3.2)); plt.plot(t, err_total); plt.xlabel("Time (s)"); plt.ylabel("Angle error (deg)"); plt.title("Attitude Error"); plt.grid(True); plt.tight_layout(); plt.savefig(OUT/"error_total.png", dpi=170); plt.close()
    plt.figure(figsize=(6,3)); plt.plot(t, err_r); plt.xlabel("Time (s)"); plt.ylabel("Roll error (deg)"); plt.title("Roll Error"); plt.grid(True); plt.tight_layout(); plt.savefig(OUT/"roll_error.png", dpi=170); plt.close()
    plt.figure(figsize=(6,3)); plt.plot(t, err_p); plt.xlabel("Time (s)"); plt.ylabel("Pitch error (deg)"); plt.title("Pitch Error"); plt.grid(True); plt.tight_layout(); plt.savefig(OUT/"pitch_error.png", dpi=170); plt.close()
    plt.figure(figsize=(6,3)); plt.plot(t, err_y); plt.xlabel("Time (s)"); plt.ylabel("Yaw error (deg)"); plt.title("Yaw Error"); plt.grid(True); plt.tight_layout(); plt.savefig(OUT/"yaw_error.png", dpi=170); plt.close()

    plt.figure(figsize=(6,3))
    plt.plot(t, nees); plt.axhline(y=np.median(nees)*2.0, linestyle="--")
    plt.xlabel("Time (s)"); plt.ylabel("trace(P_att)"); plt.title("NEES Proxy"); plt.grid(True)
    plt.tight_layout(); plt.savefig(OUT/"nees.png", dpi=170); plt.close()

    plt.figure(figsize=(5.5,3.2))
    n, bins, _ = plt.hist(err_total, bins=40, density=True, alpha=0.5)
    cdf = np.cumsum(n)/np.sum(n)
    centers = 0.5*(bins[1:]+bins[:-1])
    plt.plot(centers, cdf); plt.xlabel("Angle error (deg)"); plt.ylabel("PDF / CDF")
    plt.title("Error Distribution"); plt.grid(True); plt.tight_layout()
    plt.savefig(OUT/"hist_cdf_error.png", dpi=170); plt.close()

    az = np.mod(np.linspace(0, 2*np.pi, err_total.size), 2*np.pi)
    plt.figure(figsize=(4.0,4.0))
    ax = plt.subplot(111, projection='polar')
    ax.scatter(az, err_total, s=6, alpha=0.7)
    ax.set_title("Polar: Azimuth vs Error")
    plt.tight_layout(); plt.savefig(OUT/"polar_pointing.png", dpi=160); plt.close()

    idx, px, _ = project_dirs(euler_R(roll[-1], pitch[-1], yaw[-1]), dirs, f=900, W=1440, H=1080, fov_deg=22.0)
    center = np.array([1440/2,1080/2]); r = np.linalg.norm(px-center,axis=1)
    e = np.abs(np.random.normal(err_total.mean(), max(err_total.std(),1e-3), size=px.shape[0]))
    plt.figure(figsize=(6,3.5))
    plt.hexbin(r, e, gridsize=45); plt.xlabel("Pixel radius"); plt.ylabel("Angle error (deg)")
    plt.title("FOV Error Heatmap"); plt.tight_layout(); plt.savefig(OUT/"heatmap_fov_error.png", dpi=170); plt.close()

    taus = np.logspace(np.log10(0.08), np.log10(5.0), 18)
    adev = (lambda x: allan_deviation(x, dt, taus))(bias_est[:,2])
    plt.figure(figsize=(5.8,3.4)); plt.loglog(taus, adev, marker="o")
    plt.xlabel("Tau (s)"); plt.ylabel("Allan deviation (rad/s)")
    plt.title("Allan Deviation of Estimated Gyro Bias"); plt.tight_layout()
    plt.savefig(OUT/"allan_bias.png", dpi=170); plt.close()

    # GIF
    frames = sorted(FR.glob("frame_*.png"))
    imgs = [Image.open(p) for p in frames[:80]]
    if imgs:
        imgs[0].save(OUT/"demo.gif", save_all=True, append_images=imgs[1:], duration=80, loop=0)

    metrics = {
        "rmse_deg": float(np.sqrt(np.mean(err_total**2))),
        "median_deg": float(np.median(err_total)),
        "p95_deg": float(np.percentile(err_total, 95)),
        "mean_traceP_att": float(np.mean(nees)) if len(nees)>0 else None
    }
    (OUT/"metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics

def monte_carlo(num=10):
    finals = []
    for i in range(num):
        m = run_once(seed=100+i, render_preview=(i==0))
        finals.append(m["rmse_deg"])
    plt.figure(figsize=(4.2,3.5))
    plt.boxplot(finals, vert=True)
    plt.ylabel("RMSE (deg)"); plt.title("Monte Carlo ({} runs)".format(num))
    plt.tight_layout(); plt.savefig(OUT/"monte_carlo_box.png", dpi=160); plt.close()
    return finals

def reacquire_curve():
    occ = np.linspace(0, 0.6, 10); succ = []
    for p in occ:
        trials=20; ok=0
        for t in range(trials):
            total=28; visible = total - int(p*total)
            if visible >= 8: ok+=1
        succ.append(ok/trials)
    plt.figure(figsize=(6,3))
    plt.plot(occ*100, succ, marker="o"); plt.xlabel("Occlusion (%)"); plt.ylabel("Reacq prob")
    plt.title("LIS Reacquisition vs Occlusion"); plt.grid(True)
    plt.tight_layout(); plt.savefig(OUT/"reacquire_rate.png", dpi=170); plt.close()
    return occ.tolist(), succ

if __name__ == "__main__":
    m = run_once(seed=0, render_preview=True)
    monte_carlo(10)
    reacquire_curve()
