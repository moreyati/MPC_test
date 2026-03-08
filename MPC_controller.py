import numpy as np
import cvxpy as cp

# 引入你给出的状态空间模型
from controller_model import build_mimo_fopdt_ss


class PriorityMPCController:
    """
    4x4 MIMO MPC 控制器
    使用你提供的离散状态空间模型
    支持：
        - 总流量约束
        - 单路流量约束
        - 优先级权重
        - 偏差变量结构
    """

    def __init__(
        self,
        dt=3.0,
        Np=80,                      # 预测步长，拉长可以预测更长的时间
        Q_diag=[10, 10, 10, 2],        # 输出温度误差权重，越大越振调得越快
        R_diag=[0.02, 0.02, 0.02, 0.02],  #控制流量增量权重，越大越稳调得越慢
        F0=None,
        F_total_max=80000.0
    ):

        # ===== 1) 构造你真实的状态空间模型 =====
        A, B, C, D, meta = build_mimo_fopdt_ss(dt=dt)

        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self.nx = A.shape[0]
        self.nu = 4
        self.ny = 4

        self.Np = Np

        self.Q = np.diag(Q_diag)
        self.R = np.diag(R_diag)

        # 工作点
        if F0 is None:
            F0 = np.array([12500, 12500, 12500, 12500], dtype=float)
        self.F0 = F0

        self.F_total_max = F_total_max

        # 当前状态（必须从仿真系统同步）
        self.x = np.zeros(self.nx)

    # --------------------------------------------------
    # 外部每次仿真时更新当前状态
    # --------------------------------------------------
    def update_state(self, x_current):
        self.x = np.array(x_current).copy()

    # --------------------------------------------------
    # 主计算函数
    # --------------------------------------------------
    def compute(self, T_sp, T_current, F_ref=None):
        """
        输入:
            T_sp       : 绝对温度设定值 (4,)
            T_current  : 当前绝对温度 (4,)
            F_ref      : 当前流量设定值 (4,)，作为本步优化的基准；若为 None 则用 self.F0
        输出:
            F_sp       : 新的流量设定值 (4,) = F_ref + dU[0]
        """
        if F_ref is None:
            F_ref = np.array(self.F0, dtype=float)
        else:
            F_ref = np.array(F_ref, dtype=float).reshape(self.nu,)

        # ===== 1) 转为偏差变量 =====
        dT_sp = T_sp - T_current + self.C @ self.x
        # 注意：这里保证目标是未来 ΔT → (T_sp - T_current)

        # ===== 2) 定义优化变量 =====
        dU = cp.Variable((self.Np, self.nu))
        x = cp.Variable((self.Np + 1, self.nx))
        y = cp.Variable((self.Np, self.ny))

        cost = 0
        constraints = []

        # 初始状态约束
        constraints += [x[0] == self.x]

        # ===== 3) 构造预测模型（基准用当前设定值 F_ref，使设定值可多步累积）=====
        for k in range(self.Np):

            constraints += [
                x[k+1] == self.A @ x[k] + self.B @ dU[k],
                y[k]   == self.C @ x[k]
            ]

            # 输出误差
            cost += cp.quad_form(y[k] - dT_sp, self.Q)

            # 控制增量惩罚
            cost += cp.quad_form(dU[k], self.R)

            # 单路流量限制：F_abs = F_ref + dU[k]（相对当前设定值的增量）
            F_abs = F_ref + dU[k]

            constraints += [
                F_abs >= 0,
                F_abs <= 20000
            ]

            # 总流量约束
            constraints += [
                cp.sum(F_abs) <= self.F_total_max
            ]

        # ===== 4) 求解 =====
        problem = cp.Problem(cp.Minimize(cost), constraints)

        problem.solve(solver=cp.OSQP, warm_start=True)

        if dU.value is None:
            print("MPC 求解失败，保持当前设定值")
            return F_ref.copy()

        # ===== 5) 只执行第一步 =====
        dF_opt = dU.value[0]

        F_sp = F_ref + dF_opt

        return F_sp
