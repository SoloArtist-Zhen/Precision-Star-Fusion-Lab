
import numpy as np
from utils import quat_mul, quat_from_small, quat_to_R, skew

class MEKF:
    def __init__(self, Q_gyro=2e-6, Q_bias=5e-8, R_meas=2e-4):
        self.q = np.array([1,0,0,0], dtype=float)
        self.x = np.zeros(6)  # [dtheta(3), bias(3)]
        self.P = np.eye(6)*1e-4
        self.Q = np.diag([Q_gyro,Q_gyro,Q_gyro, Q_bias,Q_bias,Q_bias])
        self.Rm = R_meas

    def predict(self, w_m, dt):
        dtheta = (w_m - self.x[3:6])*dt
        dq = quat_from_small(dtheta)
        self.q = quat_mul(self.q, dq); self.q = self.q/np.linalg.norm(self.q)
        F = np.eye(6)
        F[0:3,3:6] = -np.eye(3)*dt
        G = np.eye(6)
        self.x = F@self.x
        self.P = F@self.P@F.T + G@self.Q@G.T*dt

    def update_vectors(self, v_inertial, y_body):
        R = quat_to_R(self.q)
        Hs = []; rs = []
        for i in range(y_body.shape[0]):
            h = (R.T @ v_inertial[i])
            r = (y_body[i] - h)
            H = np.zeros((3,6))
            H[0:3,0:3] = -skew(h)
            Hs.append(H); rs.append(r)
        H = np.vstack(Hs); r = np.hstack(rs)
        S = H@self.P@H.T + self.Rm*np.eye(H.shape[0])
        K = self.P@H.T@np.linalg.inv(S)
        dx = K@r
        self.x += dx
        self.P = (np.eye(6) - K@H)@self.P
        dq = quat_from_small(self.x[0:3])
        self.q = quat_mul(dq, self.q); self.q = self.q/np.linalg.norm(self.q)
        self.x[0:3] = 0.0
        return dx
