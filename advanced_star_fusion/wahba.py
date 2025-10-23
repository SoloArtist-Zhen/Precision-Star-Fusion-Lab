
import numpy as np

def wahba(obs_cam, ref_inertial, w=None):
    if w is None: w = np.ones(obs_cam.shape[0])
    W = np.diag(w)
    B = obs_cam.T @ W @ ref_inertial
    U,S,Vt = np.linalg.svd(B)
    M = np.eye(3)
    if np.linalg.det(U@Vt) < 0:
        M[2,2] = -1
    R = U@M@Vt
    return R
