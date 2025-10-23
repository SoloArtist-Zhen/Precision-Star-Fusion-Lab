
import numpy as np
from utils import q_mul,q_conj,q_norm,q_from_omega,R_from_q,q_from_R

class MEKFBias:
    def __init__(self, Q=1e-6, R=1e-4):
        self.q = np.array([1,0,0,0],dtype=float)
        self.b = np.zeros(3)
        self.P = np.diag([1e-6]*6)
        self.Q = Q*np.eye(6)
        self.R = R*np.eye(3)
    def step(self, w_meas, R_meas, dt):
        self.q = q_norm(q_mul(self.q, q_from_omega(w_meas - self.b, dt)))
        F = np.eye(6); self.P = F@self.P@F.T + self.Q*dt
        q_err = q_mul(q_from_R(R_meas), q_conj(self.q))
        delta = 2*q_err[1:]
        H = np.eye(6)[:3,:6]
        S = H@self.P@H.T + self.R
        K = self.P@H.T@np.linalg.inv(S)
        upd = K@delta
        dth = upd[:3]; db = upd[3:]
        self.b += db
        self.q = q_norm(q_mul(np.array([1.0, *(0.5*dth)]), self.q))
        self.P = (np.eye(6)-K@H)@self.P
        return delta, db

class UKFBias:
    # Unscented filter on error-state [dtheta, b]; measurement z ~ dtheta_meas
    def __init__(self, Q=1e-6, R=1e-4):
        self.q = np.array([1,0,0,0],dtype=float)
        self.x = np.zeros(6)   # [dtheta(3), bias(3)]
        self.P = np.diag([1e-6]*6)
        self.Q = Q*np.eye(6)
        self.R = R*np.eye(3)

    def _sigma_points(self, x, P, kappa=0):
        n = len(x)
        lam = 1e-3 - n
        U = np.linalg.cholesky((n+lam)*P + 1e-12*np.eye(n))
        pts = [x]
        for i in range(n):
            pts.append(x + U[:,i])
            pts.append(x - U[:,i])
        Wm = np.full(2*n+1, 1/(2*(n+lam))); Wc = Wm.copy()
        Wm[0] = lam/(n+lam); Wc[0] = Wm[0] + (1-1e-3**2+2)
        return np.array(pts), Wm, Wc, lam

    def step(self, w_meas, R_meas, dt):
        # Predict: propagate quaternion with gyro minus bias; bias random walk
        # Use mean state bias for propagation; error dtheta used only in correction
        self.q = q_norm(q_mul(self.q, q_from_omega(w_meas - self.x[3:], dt)))
        # UKF predict on x
        X, Wm, Wc, lam = self._sigma_points(self.x, self.P)
        Xp = X.copy()
        # process: dtheta -> 0, bias -> bias (random walk)
        Xp[:,:3] = 0
        Xp[:,3:] = X[:,3:]
        x_pred = np.sum(Wm[:,None]*Xp, axis=0)
        P_pred = self.Q*dt
        for i in range(Xp.shape[0]):
            d = (Xp[i]-x_pred).reshape(-1,1); P_pred += Wc[i]*(d@d.T)
        self.x, self.P = x_pred, P_pred

        # Measurement: from attitude measurement R_meas vs current q -> dtheta_meas
        q_err = q_mul(q_from_R(R_meas), q_conj(self.q))
        z_meas = 2*q_err[1:]

        # UKF update: h(x)=dtheta (first 3 components)
        Z = Xp[:,:3]
        z_pred = np.sum(Wm[:,None]*Z, axis=0)
        Pzz = self.R.copy()
        Pxz = np.zeros((6,3))
        for i in range(Z.shape[0]):
            dz = (Z[i]-z_pred).reshape(3,1)
            dx = (Xp[i]-self.x).reshape(6,1)
            Pzz += Wc[i]*(dz@dz.T)
            Pxz += Wc[i]*(dx@dz.T)
        K = Pxz @ np.linalg.inv(Pzz)
        self.x = self.x + (K @ (z_meas - z_pred))
        self.P = self.P - K @ Pzz @ K.T
        # correct quaternion with estimated dtheta (first 3)
        dth = self.x[:3]
        self.q = q_norm(q_mul(np.array([1.0, *(0.5*dth)]), self.q))
        # zero the error part (to keep small-angle assumption)
        self.x[:3] = 0
        return z_meas - z_pred, self.x[3:]
