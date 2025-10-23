
import numpy as np

def build_triangle_db(dirs, ids, K=900, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ids), size=K, replace=False)
    triplets = []
    for i in range(0, K-2, 3):
        a,b,c = idx[i], idx[i+1], idx[i+2]
        A = dirs[[a,b,c]]
        ang = []
        for p in range(3):
            for q in range(p+1,3):
                ang.append(np.arccos(np.clip(A[p]@A[q], -1,1)))
        ang = np.sort(np.array(ang))
        triplets.append({"ids":[int(ids[a]),int(ids[b]),int(ids[c])], "ang":ang})
    return triplets

def match_triplet(obs_dirs, tri_db, tol=0.02):
    ang = []
    for p in range(3):
        for q in range(p+1,3):
            ang.append(np.arccos(np.clip(obs_dirs[p]@obs_dirs[q], -1,1)))
    ang = np.sort(np.array(ang))
    best = None; best_err = 1e9
    for tri in tri_db:
        err = np.linalg.norm(ang - tri["ang"])
        if err < best_err:
            best_err = err; best = tri
    if best_err < tol:
        return best["ids"], best_err
    return None, best_err
