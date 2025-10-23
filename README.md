# Precision Star-Fusion Lab
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
真实感模拟：随机星表（单位向量＋星等分布）、相机径向畸变、PSF 模糊、像素噪声

LIS 识别：三角不变量建库（几何哈希），第一帧/Lost 时触发，再做近邻扩展匹配

姿态估计：Wahba（SVD）单帧测量＋MEKF 融合陀螺（含零偏随机游走），保持姿态误差3维状态＋零偏3维，6×6 P 矩阵

跟踪与再捕获：遮挡/丢星后自动重新识别
<img width="1020" height="595" alt="heatmap_fov_error" src="https://github.com/user-attachments/assets/9788257a-6785-495e-9082-0ae6b6eaf2ba" />
<img width="780" height="600" alt="scene_000" src="https://github.com/user-attachments/assets/a5717238-bc31-4e4a-89e9-b16c545b49c6" />
<img width="672" height="560" alt="monte_carlo_box" src="https://github.com/user-attachments/assets/51bd0fe8-3972-42d5-b31a-88b80cda7b16" />
<img width="640" height="640" alt="polar_pointing" src="https://github.com/user-attachments/assets/b11378c5-cbca-446a-b9bf-1753f5ff9ca7" />
<img width="1190" height="595" alt="sky_aitoff" src="https://github.com/user-attachments/assets/27ae36c1-f56d-4973-addc-1f56dcf72ad1" />
<img width="1020" height="527" alt="psd_mekf_vs_ukf" src="https://github.com/user-attachments/assets/6b0dcb15-4f6f-4dea-ae41-4446155b8624" />
<img width="951" height="544" alt="cdf_mekf_vs_ukf" src="https://github.com/user-attachments/assets/da0a941c-6aa0-43da-b613-ce8c190a2f17" />
