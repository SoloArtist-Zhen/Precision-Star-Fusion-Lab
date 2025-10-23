
import numpy as np

def project_dirs(R, dirs_inertial, f=900, W=1440, H=1080, fov_deg=22.0):
    cam = (R @ dirs_inertial.T).T
    cos_fov = np.cos(np.deg2rad(fov_deg))
    vis = cam[:,2] > 0
    vis &= (cam[:,2]/np.linalg.norm(cam,axis=1)) > cos_fov
    idx = np.where(vis)[0]
    if idx.size == 0:
        return idx, np.zeros((0,2)), np.zeros((0,3))
    v = cam[idx]
    x = v[:,0]/v[:,2]; y = v[:,1]/v[:,2]
    r2 = x*x + y*y
    k1, k2 = -0.08, 0.012
    x_d = x*(1 + k1*r2 + k2*r2*r2)
    y_d = y*(1 + k1*r2 + k2*r2*r2)
    u = f*x_d + W/2; vv = f*y_d + H/2
    cam_dirs = v/np.linalg.norm(v,axis=1,keepdims=True)
    px = np.stack([u,vv],axis=1)
    return idx, px, cam_dirs

def add_pixel_noise(px, sigma=0.6):
    return px + np.random.randn(*px.shape)*sigma

def psf_render(px, W=1440, H=1080, sigma=1.2, amp=1.0):
    img = np.zeros((H,W), dtype=float)
    for (u,v) in px.astype(int):
        if 2<=u<W-2 and 2<=v<H-2:
            for dy in range(-2,3):
                for dx in range(-2,3):
                    img[v+dy, u+dx] += amp*np.exp(-(dx*dx+dy*dy)/(2*sigma*sigma))
    img += 0.02*np.random.randn(H,W)
    return img
