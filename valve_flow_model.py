# -*- coding: utf-8 -*-
"""
阀门-流量模型（内环副对象 Gp2）+ 阀门环节 Gv（限幅）

- Gv：阀门指令限幅，将 PID 输出限制在 [u_min, u_max]。
- Gp2：4 路 SISO 一阶惯性（阀门指令 → 流量），方案 A。
"""

import numpy as np


def valve_limit(u, u_min, u_max):
    """
    阀门环节 Gv：最简单限幅。
    输入 u (4,) 或标量，输出限幅后的阀门指令 (4,)。

    u_min, u_max: 标量或 shape (4,) 的上下限
    """
    u = np.atleast_1d(np.asarray(u, dtype=float))
    if u.size != 4:
        u = np.broadcast_to(u.ravel()[0], 4)
    u_min = np.broadcast_to(np.atleast_1d(u_min), 4)
    u_max = np.broadcast_to(np.atleast_1d(u_max), 4)
    return np.clip(u, u_min, u_max)


class ValveFlowModel:
    """
    阀门-流量模型 Gp2（方案 A）：4 路 SISO 一阶惯性。

    动力学（每路 i）：
        dFcv_i/dt = (u_i - Fcv_i) / tau_i
    离散（欧拉）：
        Fcv_i[k+1] = Fcv_i[k] + (u_i[k] - Fcv_i[k]) * (dt / tau_i)

    输入 u：阀门指令 (4,)（通常为 Gv 限幅后的 PID 输出），单位与流量一致 (Nm³/h)。
    输出 Fcv：4 路实际流量 (Nm³/h)。
    """

    def __init__(self, dt=0.2, tau=8.0, flow_min=0.0, flow_max=60000.0, F0=None):
        """
        dt: 步长 (s)，与内环 PID 采样周期一致（如 SAMPLE_TIME）。
        tau: 一阶时间常数 (s)，标量（4 路同）或 shape (4,)。
        flow_min, flow_max: 流量输出上下限 (Nm³/h)。
        F0: 初始流量 (4,)，默认 [12500]*4。
        """
        self.dt = float(dt)
        self.tau = np.broadcast_to(np.atleast_1d(tau), 4).astype(float)
        self.flow_min = float(flow_min)
        self.flow_max = float(flow_max)
        if F0 is None:
            F0 = np.array([12500.0, 12500.0, 12500.0, 12500.0], dtype=float)
        self.F0 = np.array(F0, dtype=float).reshape(4,)
        self.Fcv = self.F0.copy()

    def reset(self, F0=None):
        """重置内部状态；若提供 F0 则输出置为 F0，否则置为 self.F0。"""
        if F0 is not None:
            self.Fcv = np.array(F0, dtype=float).reshape(4,).copy()
        else:
            self.Fcv = self.F0.copy()

    def step(self, u):
        """
        单步推进：输入阀门指令 u (4,)，输出本步流量 Fcv (4,)。

        u 应在限幅范围内（可由 valve_limit 得到）。
        """
        u = np.array(u, dtype=float).reshape(4,)
        # 一阶离散：Fcv += (u - Fcv) * (dt / tau)
        alpha = self.dt / self.tau
        alpha = np.clip(alpha, 0.0, 1.0)  # 避免 dt > tau 时不稳定
        self.Fcv = self.Fcv + (u - self.Fcv) * alpha
        self.Fcv = np.clip(self.Fcv, self.flow_min, self.flow_max)
        return self.Fcv.copy()

    def get_output(self):
        """当前流量输出 (4,)"""
        return self.Fcv.copy()
