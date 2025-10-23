
import numpy as np

def skew(v):
    x,y,z = v
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]],dtype=float)

def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1; w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_inv(q):
    return np.array([q[0],-q[1],-q[2],-q[3]])/(np.dot(q,q)+1e-15)

def quat_from_small(delta):
    d = np.array(delta).reshape(3)
    n2 = np.dot(d,d)
    w = 1.0 - 0.125*n2
    xyz = 0.5*d
    return np.array([w, xyz[0], xyz[1], xyz[2]])

def quat_to_R(q):
    w,x,y,z = q / (np.linalg.norm(q)+1e-15)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ])

def R_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t+1.0)*2
        w = 0.25*s
        x = (R[2,1]-R[1,2])/s
        y = (R[0,2]-R[2,0])/s
        z = (R[1,0]-R[0,1])/s
    else:
        i = np.argmax(np.diag(R))
        if i == 0:
            s = np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2
            w = (R[2,1]-R[1,2])/s; x = 0.25*s; y = (R[0,1]+R[1,0])/s; z = (R[0,2]+R[2,0])/s
        elif i == 1:
            s = np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2
            w = (R[0,2]-R[2,0])/s; x = (R[0,1]+R[1,0])/s; y = 0.25*s; z = (R[1,2]+R[2,1])/s
        else:
            s = np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2
            w = (R[1,0]-R[0,1])/s; x = (R[0,2]+R[2,0])/s; y = (R[1,2]+R[2,1])/s; z = 0.25*s
    q = np.array([w,x,y,z])
    return q/(np.linalg.norm(q)+1e-15)

def ang_err_deg(R_est, R_true):
    Rt = R_est @ R_true.T
    ang = np.arccos(np.clip((np.trace(Rt)-1)/2, -1, 1))
    return np.rad2deg(ang)
