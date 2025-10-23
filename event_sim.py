
import numpy as np
from utils import R_from_q

def project(R, stars, focal=900, W=1280, H=960, fov_deg=18):
    cam = (R @ stars.T).T
    cos_fov = np.cos(np.deg2rad(fov_deg))
    keep = np.logical_and(cam[:,2] > 0, cam[:,2]/np.linalg.norm(cam,axis=1) > cos_fov)
    idx = np.where(keep)[0]
    v = cam[idx]
    u = focal*(v[:,0]/v[:,2]) + W/2
    w = focal*(v[:,1]/v[:,2]) + H/2
    return idx, np.stack([u,w],axis=1), v

def synth_events(q_hist, stars, dt, focal, W,H,fov, C=0.3, leak=0.0):
    """
    Very simple event generator:
    - For each frame dt, compute projected positions; intensity ~ Gaussian around star center
    - Emit an event at pixel when log-intensity change dL exceeds +/- C
    - We approximate using star centers and keep a per-star accumulator for log change
    Returns list of events: (t, x, y, p) where p in {-1, +1}
    """
    acc = np.zeros(stars.shape[0])
    events = []
    for k,q in enumerate(q_hist):
        R = R_from_q(q)
        idx, px, camv = project(R, stars, focal=focal, W=W,H=H, fov_deg=fov)
        # approximate log-intensity change by pixel displacement magnitude
        # (toy model: larger motion => more change)
        if k == 0:
            prev = px.copy()
        else:
            # use only tracked subset for speed
            m = min(300, px.shape[0], prev.shape[0])
            order = np.argsort(np.linalg.norm(px - np.array([W/2,H/2]),axis=1))
            ii = order[:m]
            disp = np.linalg.norm(px[ii]-prev[:m], axis=1)/(np.sqrt(W*H)+1e-9)
            acc[idx[ii]] += disp - leak*acc[idx[ii]]
            fired_pos = ii[acc[idx[ii]] > C]
            for j in fired_pos:
                t = k*dt
                x,y = px[j]
                p = 1 if acc[idx[j]]>C else -1
                events.append((t, float(x), float(y), int(p)))
                acc[idx[j]] = 0.0
            prev = px.copy()
    return events
