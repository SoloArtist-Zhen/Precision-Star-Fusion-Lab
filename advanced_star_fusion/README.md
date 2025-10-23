
# Advanced Project: High-Accuracy Star Tracker–Inertial Fusion Playground

**Purpose**
- Realistic star catalog simulator, camera with radial distortion, PSF blur, and pixel noise.
- Lost-in-Space (LIS) identification via triangle invariants; coarse-to-fine expansion.
- Per-frame Wahba attitude + **full MEKF** (attitude error 3 + gyro bias 3).
- Tracking, occlusion re-acquisition.
- **Advanced plots**: errors (total/roll/pitch/yaw), NEES, histogram+CDF, polar scatter,
  FOV error heatmap, Allan deviation of bias, Monte Carlo boxplot, reacquisition curve, GIF.

**Run**
```bash
python main.py
```
