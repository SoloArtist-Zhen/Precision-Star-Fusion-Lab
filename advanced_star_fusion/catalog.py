
import numpy as np

def random_catalog(N=5000, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.random(N); v = rng.random(N)
    theta = 2*np.pi*u
    phi = np.arccos(2*v-1)
    x = np.sin(phi)*np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(phi)
    dirs = np.stack([x,y,z], axis=1)
    dirs = dirs/np.linalg.norm(dirs,axis=1,keepdims=True)
    mags = 1.4 + 5.0*rng.random(N)**0.7
    ids = np.arange(N)
    return ids, dirs, mags
