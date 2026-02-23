import numpy as np

#控制器模型（状态空间模型）
def build_mimo_fopdt_ss(dt=60.0):
    """
    将 4×4 的 FOPDT 传递函数矩阵模型：
        G_ij(s) = K_ij / (tau_ij*s + 1) * exp(-L_ij*s)
    转换为离散状态空间模型：
        x_{k+1} = A x_k + B u_k
        y_k     = C x_k + D u_k

    其中：
    - u_k 为 4 路输入（流量偏差 ΔF）
    - y_k 为 4 路输出（温度偏差 ΔT）
    - 纯滞后 exp(-L*s) 通过“延迟链 shift register”实现

    参数
    ----
    dt : float
        离散采样周期（秒），例如 60s

    返回
    ----
    A, B, C, D : ndarray
        离散状态空间矩阵
    meta : dict
        一些辅助信息（状态维度、每个通道延迟步数等）
    """

    # ========== 1) 定义模型参数（与之前一致） ==========
    dt = float(dt)
    dt_min = dt / 60.0  # 秒 -> 分钟（因为 tau 和 L 用的是分钟）

    # K: ℃/(Nm3/h)
    K = 1.0e-2 * np.array([
        [5.0, 1.2, 0.6, 0.4],
        [2.5, 5.0, 1.2, 0.6],
        [1.2, 2.5, 5.0, 1.2],
        [0.8, 1.2, 2.5, 5.0]
    ], dtype=float)

    bias_factor = 0.9  # 或 1.1，表示降低或增加10%
    K = K * bias_factor

    # tau: min
    tau = np.array([
        [12, 18, 22, 25],
        [16, 12, 18, 22],
        [20, 16, 12, 18],
        [24, 20, 16, 12]
    ], dtype=float)

    # L: min (纯滞后)
    L = np.array([
        [0.5, 1.5, 2.5, 3.0],
        [1.0, 0.5, 1.5, 2.5],
        [2.0, 1.0, 0.5, 1.5],
        [3.0, 2.0, 1.0, 0.5]
    ], dtype=float)

    # ========== 2) 将每个通道离散化：一阶惯性 + 延迟 ==========
    # 一阶惯性精确离散：
    #   x[k+1] = a x[k] + (1-a)K * u_delayed[k]
    a = np.exp(-dt_min / tau)           # 4x4
    b = (1.0 - a) * K                   # 4x4

    # 纯滞后离散为整数步延迟（四舍五入）
    # delay_steps[i,j] 表示 ΔF_j 延迟多少步后才作用于输出 i 的通道
    delay_steps = np.maximum(np.round(L / dt_min).astype(int), 0)

    # ========== 3) 构造扩维状态空间 ==========
    # 状态由两部分组成：
    # (A) 动态一阶惯性状态 x_dyn[i,j]  共 16 个
    # (B) 延迟链状态 x_delay[i,j,1..n]  每个通道一个 shift register
    #
    # 延迟链解释：
    # 若某通道延迟 n 步，则：
    #   d0 = u[k]
    #   d1[k+1] = d0[k]
    #   d2[k+1] = d1[k]
    #   ...
    #   dn[k+1] = d(n-1)[k]
    # 并且一阶惯性使用 u_delayed = dn[k]

    # 计算总延迟状态数
    total_delay_states = int(delay_steps.sum())
    n_dyn = 16
    n_x = n_dyn + total_delay_states

    # A, B, C, D 初始化
    A = np.zeros((n_x, n_x), dtype=float)
    B = np.zeros((n_x, 4), dtype=float)
    C = np.zeros((4, n_x), dtype=float)
    D = np.zeros((4, 4), dtype=float)  # 本模型严格真延迟，因此 D=0

    # 用于记录每个通道的状态索引，方便调试/后续扩展
    # dyn_idx[i,j] 给出动态状态 x_dyn(i,j) 在 x 向量中的位置
    dyn_idx = np.zeros((4, 4), dtype=int)

    # delay_chain_idx[i,j] 是一个 list，存该通道延迟链每个状态在 x 中的位置
    delay_chain_idx = [[[] for _ in range(4)] for __ in range(4)]

    # ========== 4) 给状态编号 ==========
    # 先编号动态状态 16 个
    idx = 0
    for i in range(4):
        for j in range(4):
            dyn_idx[i, j] = idx
            idx += 1

    # 再编号延迟链状态
    for i in range(4):
        for j in range(4):
            n_delay = int(delay_steps[i, j])
            # 为该通道创建 n_delay 个延迟状态
            for _ in range(n_delay):
                delay_chain_idx[i][j].append(idx)
                idx += 1

    assert idx == n_x, "状态编号错误：最终状态维度不一致"

    # ========== 5) 构造状态方程 x[k+1] = A x[k] + B u[k] ==========
    # 5.1 延迟链的状态更新
    # 对每个通道 (i,j)：
    #   若 n_delay = 0：u_delayed = u_j[k]（直接使用输入）
    #   若 n_delay > 0：延迟链第一个状态接收 u_j[k]，其余 shift
    for i in range(4):
        for j in range(4):
            chain = delay_chain_idx[i][j]
            n_delay = len(chain)

            if n_delay == 0:
                # 无延迟链状态，后面动态方程直接用 u_j[k]
                pass
            else:
                # 第一个延迟状态：d1[k+1] = u_j[k]
                A[chain[0], chain[0]] = 0.0
                B[chain[0], j] = 1.0

                # 后续延迟状态 shift：
                # d(m)[k+1] = d(m-1)[k]
                for m in range(1, n_delay):
                    A[chain[m], chain[m-1]] = 1.0

    # 5.2 动态一阶惯性状态更新
    # x_dyn_ij[k+1] = a_ij * x_dyn_ij[k] + b_ij * u_delayed
    #
    # u_delayed 的来源：
    #   - 若该通道延迟为 0：u_delayed = u_j[k]
    #   - 若延迟为 n：u_delayed = 延迟链最后一个状态 dn[k]
    for i in range(4):
        for j in range(4):
            xij = dyn_idx[i, j]
            A[xij, xij] = a[i, j]

            chain = delay_chain_idx[i][j]
            if len(chain) == 0:
                # 无延迟：直接从输入 u_j 进入
                B[xij, j] += b[i, j]
            else:
                # 有延迟：从延迟链最后一个状态进入
                delayed_state = chain[-1]
                A[xij, delayed_state] += b[i, j]

    # ========== 6) 构造输出方程 y[k] = C x[k] + D u[k] ==========
    # 输出 ΔT_i = Σ_j x_dyn(i,j)
    for i in range(4):
        for j in range(4):
            C[i, dyn_idx[i, j]] = 1.0

    # D 保持 0（纯滞后系统严格无直接通道）

    meta = {
        "dt": dt,
        "dt_min": dt_min,
        "K": K,
        "tau": tau,
        "L": L,
        "a": a,
        "b": b,
        "delay_steps": delay_steps,
        "n_x": n_x,
        "n_dyn": n_dyn,
        "n_delay": total_delay_states,
        "dyn_idx": dyn_idx,
        "delay_chain_idx": delay_chain_idx
    }

    return A, B, C, D, meta


class DiscreteStateSpacePlant:
    """
    一个通用的离散状态空间仿真器：
        x[k+1] = A x[k] + B u[k]
        y[k]   = C x[k] + D u[k]
    """

    def __init__(self, A, B, C, D, x0=None):
        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float)
        self.C = np.array(C, dtype=float)
        self.D = np.array(D, dtype=float)

        n = self.A.shape[0]
        if x0 is None:
            self.x = np.zeros(n, dtype=float)
        else:
            self.x = np.array(x0, dtype=float).reshape(n,)

    def reset(self, x0=None):
        if x0 is None:
            self.x[:] = 0.0
        else:
            self.x = np.array(x0, dtype=float).reshape(self.x.shape)

    def step(self, u):
        """
        输入 u (4,) 输出 y (4,)
        """
        u = np.array(u, dtype=float).reshape(4,)
        y = self.C @ self.x + self.D @ u
        self.x = self.A @ self.x + self.B @ u
        return y


# ---------------- 测试运行 ----------------
if __name__ == "__main__":
    # 生成状态空间矩阵
    A, B, C, D, meta = build_mimo_fopdt_ss(dt=3.0)

    print("离散状态空间维度：")
    print("A:", A.shape, "B:", B.shape, "C:", C.shape, "D:", D.shape)
    print("动态状态数:", meta["n_dyn"], "延迟链状态数:", meta["n_delay"], "总状态数:", meta["n_x"])
    print("延迟步数矩阵 delay_steps=\n", meta["delay_steps"])

    # 构造状态空间 plant
    plant_ss = DiscreteStateSpacePlant(A, B, C, D)

    # 工作点（用于从绝对量转为偏差量）
    T0 = np.array([500, 500, 500, 500], dtype=float)
    F0 = np.array([12500, 12500, 12500, 12500], dtype=float)

    # 仿真 120 min，每步 60s
    N = 120
    F = F0.copy()

    for k in range(N):
        # 第 10 分钟：对 F1 施加 +2000 的阶跃
        if k == 10:
            F[0] += 2000.0

        # 输入必须用偏差变量 ΔF
        dF = F - F0

        # 状态空间输出为 ΔT
        dT = plant_ss.step(dF)

        # 转为绝对温度
        T = T0 + dT

        # 限幅（模拟物理范围）
        T = np.clip(T, 0.0, 600.0)

        if k % 10 == 0:
            print(f"min={k:3d}  Fsum={F.sum():8.1f}  T={T}")
