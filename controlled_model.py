import numpy as np
from collections import deque

#被控对象模型（传递函数模型）
class MIMOFOPDTPlant:
    """
    4x4 MIMO Plant:
        ΔT_i(s) = Σ_j G_ij(s) ΔF_j(s)
        G_ij(s) = K_ij / (τ_ij s + 1) * e^{-L_ij s}

    Discretization:
        Δx_ij[k+1] = a_ij * Δx_ij[k] + (1-a_ij) * K_ij * ΔF_j_delayed[k]
        ΔT_i[k] = Σ_j Δx_ij[k]
    """

    def __init__(self, dt=60.0):
        """
        dt: sampling time in seconds (e.g. 5 for 5s update, 60 for 1 min).
        tau, L 为分钟，内部 dt_min = dt/60 用于离散化与纯滞后步数。
        """
        self.dt = float(dt)
        self.dt_min = self.dt / 60.0

        self.T0 = np.array([500.0, 500.0, 500.0, 500.0])
        self.F0 = np.array([12500.0, 12500.0, 12500.0, 12500.0])

        # 【修改点1】K 改为正数，且增大 10 倍，模拟“燃料增加->温度快速升高”
        # 单位: ℃ / (Nm3/h)
        self.K = 1.0e-2 * np.array([
            [5.0, 1.2, 0.6, 0.4],
            [2.5, 5.0, 1.2, 0.6],
            [1.2, 2.5, 5.0, 1.2],
            [0.8, 1.2, 2.5, 5.0]
        ], dtype=float)

        # 【修改点2】时间常数 tau 保持不变，代表物理惯性
        self.tau = np.array([
            [12, 18, 22, 25],
            [16, 12, 18, 22],
            [20, 16, 12, 18],
            [24, 20, 16, 12]
        ], dtype=float)

        # 纯滞后 L
        self.L = np.array([
            [0.5, 1.5, 2.5, 3.0],
            [1.0, 0.5, 1.5, 2.5],
            [2.0, 1.0, 0.5, 1.5],
            [3.0, 2.0, 1.0, 0.5]
        ], dtype=float)

        assert self.K.shape == (4, 4)
        assert self.tau.shape == (4, 4)
        assert self.L.shape == (4, 4)

        # -------- Discrete coefficients --------
        # a = exp(-dt/tau), b = (1-a)*K
        self.a = np.exp(-self.dt_min / self.tau)
        self.b = (1.0 - self.a) * self.K

        # -------- Internal states Δx_ij --------
        self.x = np.zeros((4, 4), dtype=float)

        # -------- Dead-time buffers per input per output (L_ij) --------
        # We implement delay on each channel ΔF_j -> (i,j) independently.
        self.delay_steps = np.maximum(
            (self.L / self.dt_min).round().astype(int),
            0
        )

        self.buffers = [[None for _ in range(4)] for __ in range(4)]
        for i in range(4):
            for j in range(4):
                n = int(self.delay_steps[i, j])
                # deque length n+1 so that "current" delayed value is popped from left
                self.buffers[i][j] = deque([0.0] * (n + 1), maxlen=(n + 1))

        # Output
        self.T = self.T0.copy()

    def reset(self, T_init=None, F0=None):
        """Reset plant state."""
        self.x[:] = 0.0
        if T_init is None:
            self.T = self.T0.copy()
        else:
            self.T = np.array(T_init, dtype=float).reshape(4,)
        if F0 is not None:
            self.F0 = np.array(F0, dtype=float).reshape(4,)

        # reset buffers
        for i in range(4):
            for j in range(4):
                n = int(self.delay_steps[i, j])
                self.buffers[i][j].clear()
                self.buffers[i][j].extend([0.0] * (n + 1))

    def step(self, F, noise_std=0.0, enforce_total_flow=None):
        """
        One simulation step.

        F: array-like, shape (4,), absolute flows (Nm3/h)
        noise_std: output measurement noise (℃), Gaussian
        enforce_total_flow: None or tuple(min_total, max_total)
            If set, will clip total flow into range by scaling all 4 flows proportionally.
        """
        F = np.array(F, dtype=float).reshape(4,)

        # Optional total flow constraint (simple scaling)
        if enforce_total_flow is not None:
            min_total, max_total = enforce_total_flow
            total = float(F.sum())
            if total < min_total and total > 1e-9:
                F *= (min_total / total)
            elif total > max_total and total > 1e-9:
                F *= (max_total / total)

        dF = F - self.F0  # deviation variable

        # Update each channel (i,j)
        for i in range(4):
            for j in range(4):
                # push current dF_j into delay buffer
                self.buffers[i][j].append(float(dF[j]))
                dF_delayed = self.buffers[i][j][0]

                # first-order discrete update
                self.x[i, j] = self.a[i, j] * self.x[i, j] + self.b[i, j] * dF_delayed

        dT = self.x.sum(axis=1)  # sum over j
        T = self.T0 + dT

        # add measurement noise
        if noise_std > 0:
            T = T + np.random.randn(4) * noise_std

        # clamp physical range
        T = np.clip(T, 0.0, 600.0)

        self.T = T
        return T.copy()

    def get_output(self):
        return self.T.copy()


# ---------------- 交互式：手动输入四路流量，计算并打印温度 ----------------
def run_interactive():
    """运行后循环读入四路流量，计算温度并打印，直到用户输入 q 或 Ctrl+C。"""
    plant = MIMOFOPDTPlant(dt=60.0)
    plant.reset()

    print("流量-温度模型 交互计算")
    print("工作点 F0 =", plant.F0.tolist(), "Nm3/h,  T0 =", plant.T0.tolist(), "℃")
    print("请输入四个阀门流量 (Nm3/h)，用空格或逗号分隔；输入 q 或 quit 退出。")
    print("-" * 50)

    while True:
        try:
            line = input("四路流量 F1 F2 F3 F4 > ").strip()
        except EOFError:
            print("\n已退出。")
            break
        if not line:
            continue
        if line.lower() in ("q", "quit", "exit"):
            print("已退出。")
            break

        parts = line.replace(",", " ").split()
        if len(parts) != 4:
            print("  需要 4 个数值，当前输入:", len(parts), "个，请重新输入。")
            continue
        try:
            F = [float(x) for x in parts]
        except ValueError as e:
            print("  输入无效，请输入 4 个数字:", e)
            continue

        F = np.array(F)
        T = plant.step(F, noise_std=0.5, enforce_total_flow=(10000.0, 80000.0))
        print("  流量 (Nm3/h):", [round(f, 1) for f in F], "  总流量:", round(float(F.sum()), 1))
        print("  温度 (℃):   ", [round(t, 2) for t in T])
        print()


# ---------------- Example usage (batch) ----------------
def run_example_batch():
    """示例：批量仿真 120 分钟，步进 F1。"""
    plant = MIMOFOPDTPlant(dt=60.0)
    plant.reset()
    N = 120
    F = plant.F0.copy()
    for k in range(N):
        if k == 10:
            F[0] += 2000.0
        T = plant.step(F, noise_std=0.2, enforce_total_flow=(40000.0, 60000.0))
        if k % 10 == 0:
            print(f"min={k:3d}  F={F.sum():8.1f}  T={T}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        run_example_batch()
    else:
        run_interactive()
