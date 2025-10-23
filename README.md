# Precision Star-Fusion Lab — PRO
Upgrades on top of the LAB version:

1) **UKF(误差状态 sigma 点) vs MEKF**：同一测量模型下的时间序列、**CDF/PSD 对比**。
2) **事件相机(Events) 合成器 + EBS-EKF 管线**：毫秒级事件窗口聚类→星点→EKF 高频姿态；与 APS 帧式估计对比**延迟/频响**。
3) **更稳的 Lost-in-Space (LIS)**：**角距离表 + KD-Tree 最近邻门限** + **时间连贯 gating**（保持星迹的一致性）。

## 运行
```bash
pip install numpy matplotlib pillow
python main.py
```
生成在 `outputs/`：
- `cdf_mekf_vs_ukf.png`, `psd_mekf_vs_ukf.png`
- `cdf_event_vs_aps.png`, `psd_event_vs_aps.png`, `latency_event_vs_aps.txt`
- `sky_aitoff.png`, `demo_aps.gif`, `demo_event.gif`
- 其他图与 `metrics.json`
