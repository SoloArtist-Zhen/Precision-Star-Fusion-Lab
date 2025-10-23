
import numpy as np, math

# ---------- Quaternion / Rotation ----------
def q_mul(q1,q2):
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ],dtype=float)

def q_conj(q): return np.array([q[0],-q[1],-q[2],-q[3]],dtype=float)

def q_norm(q): return q/np.linalg.norm(q)

def q_from_omega(omega, dt):
    th = np.linalg.norm(omega)*dt
    if th < 1e-12: return np.array([1,0,0,0],dtype=float)
    axis = omega/np.linalg.norm(omega)
    s = math.sin(th/2.0)
    return np.array([math.cos(th/2.0), axis[0]*s, axis[1]*s, axis[2]*s])

def R_from_q(q):
    q = q_norm(q)
    w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ])

def q_from_R(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t+1.0)*2
        w = 0.25*s
        x = (R[2,1]-R[1,2])/s
        y = (R[0,2]-R[2,0])/s
        z = (R[1,0]-R[0,1])/s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = math.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2
            w = (R[2,1]-R[1,2])/s; x = 0.25*s; y = (R[0,1]+R[1,0])/s; z = (R[0,2]+R[2,0])/s
        elif i == 1:
            s = math.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2
            w = (R[0,2]-R[2,0])/s; x = (R[0,1]+R[1,0])/s; y = 0.25*s; z = (R[1,2]+R[2,1])/s
        else:
            s = math.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2
            w = (R[1,0]-R[0,1])/s; x = (R[0,2]+R[2,0])/s; y = (R[1,2]+R[2,1])/s; z = 0.25*s
    q = np.array([w,x,y,z],dtype=float)
    return q/np.linalg.norm(q)

def ang_error_deg(R_est, R_true):
    Rt = R_est @ R_true.T
    th = np.arccos(np.clip((np.trace(Rt)-1)/2, -1, 1))
    return np.rad2deg(th)

# ---------- Aitoff helper ----------
def aitoff_xy(theta, phi):
    # theta in [-pi, pi], phi in [-pi/2, pi/2]
    alpha = np.arccos(np.cos(phi)*np.cos(theta/2))
    with np.errstate(invalid='ignore', divide='ignore'):
        x = 2*np.cos(phi)*np.sin(theta/2)/np.sinc(alpha/np.pi)
        y = np.sin(phi)/np.sinc(alpha/np.pi)
    x[np.isnan(x)] = 0; y[np.isnan(y)] = 0
    return x, y

# ---------- KD-Tree (simple) ----------
class KDTree:
    def __init__(self, pts, leaf_size=16):
        self.pts = np.asarray(pts)
        self.idx = np.arange(len(pts))
        self.leaf_size = leaf_size
        self.tree = self._build(self.idx)

    def _build(self, idx, depth=0):
        if len(idx) <= self.leaf_size:
            return ("leaf", idx)
        axis = depth % 3
        pts = self.pts[idx]
        mid = np.argsort(pts[:,axis])[len(idx)//2]
        pivot = idx[mid]
        left = idx[self.pts[idx,axis] <= self.pts[pivot,axis]]
        right= idx[self.pts[idx,axis] >  self.pts[pivot,axis]]
        return ("node", axis, pivot, self._build(left, depth+1), self._build(right, depth+1))

    def _nn(self, node, q, best):
        kind = node[0]
        if kind == "leaf":
            idx = node[1]
            d = np.sum((self.pts[idx]-q)**2, axis=1)
            i = idx[np.argmin(d)]
            di = float(np.min(d))
            if di < best[0]:
                return (di, i)
            return best
        _, axis, pivot, L, R = node
        pv = self.pts[pivot]
        next_branch = L if q[axis] <= pv[axis] else R
        other_branch= R if q[axis] <= pv[axis] else L
        best = self._nn(next_branch, q, best)
        if (q[axis]-pv[axis])**2 < best[0]:
            best = self._nn(other_branch, q, best)
        # check pivot
        dp = float(np.sum((pv-q)**2))
        if dp < best[0]: best = (dp, pivot)
        return best

    def query(self, q):
        return self._nn(self.tree, q, (1e9, -1))

# ---------- LIS robust utilities ----------
def build_angle_table(stars, k=20):
    # For each star, keep k nearest neighbors (by Euclidean on unit sphere) and store angular distances
    tree = KDTree(stars)
    # naive kNN: query by grid (repeat calls)
    knn_idx = []
    for s in stars:
        # simple brute force for k neighbors (we avoid complex kNN to keep code compact)
        d2 = np.sum((stars - s)**2, axis=1)
        order = np.argsort(d2)[1:k+1]
        knn_idx.append(order)
    ang_tab = []
    for i, nbrs in enumerate(knn_idx):
        angs = np.arccos(np.clip(stars[nbrs] @ stars[i], -1, 1))
        ang_tab.append((i, nbrs, angs))
    return ang_tab, tree

def lis_match(obs_dirs, stars, ang_tab, tree, ang_tol=np.deg2rad(0.05)):
    # Use pair angles as signature: pick a seed observation and compare its pairwise angles to candidate catalog stars
    m = obs_dirs.shape[0]
    if m < 3: return None
    s0 = obs_dirs[0]
    # NN in catalog
    _, i0 = tree.query(s0)
    cand_idx = [i0]
    # expand using angle matches with neighbors
    obs_ang = np.arccos(np.clip(obs_dirs @ s0, -1, 1))
    i, nbrs, angs = ang_tab[i0]
    matches = [i0]
    for j, ang in enumerate(obs_ang[1:6]):
        # find closest angle in catalog neighbor list
        da = np.abs(angs - ang)
        t = np.argmin(da)
        if da[t] < ang_tol:
            matches.append(nbrs[t])
    if len(matches) < 3: return None
    return matches[:min(12,len(matches))]

def temporal_gating(prev_ids, curr_ids):
    # Keep intersection to maintain identity; fallback to current if empty
    if prev_ids is None: return curr_ids
    inter = [i for i in curr_ids if i in prev_ids]
    return inter if len(inter)>=3 else curr_ids

# ---------- Wahba/QUEST + RANSAC ----------
def quest(obs_cam, ref_inertial, w=None):
    assert obs_cam.shape == ref_inertial.shape
    n = obs_cam.shape[0]
    if w is None: w = np.ones(n)
    B = np.zeros((3,3))
    for i in range(n):
        B += w[i] * np.outer(obs_cam[i], ref_inertial[i])
    S = B + B.T
    sigma = np.trace(B)
    Z = np.array([B[1,2]-B[2,1], B[2,0]-B[0,2], B[0,1]-B[1,0]])
    K = np.zeros((4,4))
    K[:3,:3] = S - sigma*np.eye(3)
    K[:3,3]  = Z
    K[3,:3]  = Z
    K[3,3]   = sigma
    eigvals, eigvecs = np.linalg.eigh(K)
    q = eigvecs[:, np.argmax(eigvals)]; q = q/np.linalg.norm(q)
    w,x,y,z = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ])
    return R

def ransac_wahba(obs_cam, ref_inertial, iters=200, thresh=np.deg2rad(0.03)):
    N = obs_cam.shape[0]
    best_in = []
    best_R = np.eye(3)
    idx = np.arange(N)
    for _ in range(iters):
        if N < 3: break
        samp = np.random.choice(idx, size=3, replace=False)
        R = quest(obs_cam[samp], ref_inertial[samp])
        pred = (R @ ref_inertial.T).T
        ang = np.arccos(np.clip(np.sum(pred*obs_cam,axis=1), -1,1))
        inliers = np.where(ang < thresh)[0]
        if len(inliers) > len(best_in):
            best_in = inliers
            best_R = quest(obs_cam[inliers], ref_inertial[inliers])
    return best_R, best_in

# ---------- Allan deviation ----------
def allan_deviation(x, fs):
    x = np.asarray(x); N = len(x)
    taus = np.unique(np.logspace(0, np.log10(max(2, N//4)), num=20, dtype=int))
    adev = []
    for m in taus:
        if 2*m >= N: break
        y = np.cumsum(x)/fs
        z = y[2*m:] - 2*y[m:-m] + y[:-2*m]
        sigma2 = 0.5*(1/((N-2*m)))*(1/((m/fs)**2))*np.mean(z**2)
        adev.append(np.sqrt(max(sigma2,0)))
    taus = taus[:len(adev)]
    return (taus/fs), np.array(adev)
