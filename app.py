# -*- coding: utf-8 -*-
"""
四回路 PID 流量仿真 + 4×4 流量-温度模型 Web 后端

控制对象为串级结构：
- 内环副对象 Gp2：阀门-流量模型（ValveFlowModel），阀门指令 u → 流量 Fcv
- 阀门环节 Gv：限幅（valve_limit）
- 外环主对象 Gp1：流量-温度模型（MIMOFOPDTPlant），流量 Fcv → 温度 Tcv

可选 MPC+PID 串级：温度设定 → MPC 求 F_sp → PID 跟踪 F_sp → Gv → Gp2 → Fcv → Gp1 → Tcv
"""

import threading
import time
import numpy as np
from flask import Flask, render_template, jsonify, request

from controlled_model import MIMOFOPDTPlant
from MPC_controller import PriorityMPCController
from valve_flow_model import ValveFlowModel, valve_limit

app = Flask(__name__)

# ============ 仿真参数 ============
NUM_LOOPS = 4
FLOW_MIN = 0.0
FLOW_MAX = 60000.0
INITIAL_SETPOINT_PER_LOOP = 12500.0   # 与 plant.F0 一致，总流量 50000
SAMPLE_TIME = 0.2    # PID 采样周期 (s)
PLANT_DT = 3.0      # 流量-温度模型（Gp1）采样周期 (s)
PLANT_STEPS_PER_CALL = int(PLANT_DT / SAMPLE_TIME)
VALVE_FLOW_TAU = 8.0   # Gp2 阀门-流量一阶时间常数 (s)

def _init_setpoints():
    return [INITIAL_SETPOINT_PER_LOOP] * NUM_LOOPS

def _init_flows():
    return [INITIAL_SETPOINT_PER_LOOP] * NUM_LOOPS

# ============ 全局状态 (线程安全) ============
_lock = threading.Lock()
_state = {
    "running": False,
    "setpoints": _init_setpoints(),
    "flows": _init_flows(),
    "controls": _init_flows(),
    "time": 0.0,
    "Kp": [10.0] * NUM_LOOPS,
    "Ki": [5.0] * NUM_LOOPS,
    "Kd": [0.0] * NUM_LOOPS,
    "integrals": [0.0] * NUM_LOOPS,
    "last_errors": [0.0] * NUM_LOOPS,
    "last_times": [None] * NUM_LOOPS,
    "temperatures": [500.0] * NUM_LOOPS,
    "control_mode": "pid_only",       # "pid_only" | "mpc_cascade"
    "T_setpoints": [500.0] * NUM_LOOPS,
}
_history_flow = []
_history_temp = []
_MAX_HISTORY_FLOW = 600
_MAX_HISTORY_TEMP = 360

# 外环主对象 Gp1：流量-温度模型
_plant = MIMOFOPDTPlant(dt=PLANT_DT)

# 内环副对象 Gp2：阀门-流量模型；Gv 限幅在仿真循环中通过 valve_limit 实现
_valve_flow = ValveFlowModel(
    dt=SAMPLE_TIME,
    tau=VALVE_FLOW_TAU,
    flow_min=FLOW_MIN,
    flow_max=FLOW_MAX,
    F0=np.array([INITIAL_SETPOINT_PER_LOOP] * NUM_LOOPS),
)

# MPC 控制器与状态（方案 A：与 controller_model 状态空间同步维护）
F0_MPC = np.array([12500.0] * NUM_LOOPS)
_mpc_controller = PriorityMPCController(dt=PLANT_DT, F0=F0_MPC, F_total_max=50000.0)
_mpc_state_x = np.zeros(_mpc_controller.nx)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def run_simulation():
    """仿真循环：4 路 PID + Gv 限幅 + Gp2 阀门-流量，每 PLANT_DT s 更新 Gp1 与 MPC"""
    global _state, _history_flow, _history_temp
    step_count = 0

    while True:
        with _lock:
            if not _state["running"]:
                time.sleep(0.05)
                continue

            t = _state["time"]
            setpoints = list(_state["setpoints"])
            flows = list(_state["flows"])  # Fcv，来自 Gp2 上一步输出
            Kp = list(_state["Kp"])
            Ki = list(_state["Ki"])
            Kd = list(_state["Kd"])
            integrals = list(_state["integrals"])
            last_errors = list(_state["last_errors"])
            last_times = list(_state["last_times"])

        controls = [0.0] * NUM_LOOPS
        dt = SAMPLE_TIME

        for i in range(NUM_LOOPS):
            sp = setpoints[i]
            fcv = flows[i]  # 内环反馈为 Gp2 输出 Fcv
            integral = integrals[i]
            last_error = last_errors[i]
            last_time = last_times[i]

            error = sp - fcv
            integral = integral + error * dt
            integral = clamp(integral, -1e6, 1e6)
            derivative = (error - last_error) / dt if last_time is not None else 0.0

            u = Kp[i] * error + Ki[i] * integral + Kd[i] * derivative
            controls[i] = u

            integrals[i] = integral
            last_errors[i] = error
            last_times[i] = t

        # Gv：阀门限幅（controls 为 PID 输出，u_limited 为实际送入 Gp2 的阀门指令）
        u_limited = valve_limit(np.array(controls), FLOW_MIN, FLOW_MAX)
        # Gp2：阀门-流量一步，得到本步 Fcv
        flows_new = list(_valve_flow.step(u_limited))

        with _lock:
            _state["flows"] = flows_new
            _state["controls"] = list(u_limited)
            _state["time"] = t + dt
            _state["integrals"] = integrals
            _state["last_errors"] = last_errors
            _state["last_times"] = last_times

            _history_flow.append({
                "time": round(t + dt, 2),
                "setpoints": [round(setpoints[j], 2) for j in range(NUM_LOOPS)],
                "flows": [round(flows_new[j], 2) for j in range(NUM_LOOPS)],
            })
            if len(_history_flow) > _MAX_HISTORY_FLOW:
                _history_flow.pop(0)

        step_count += 1
        if step_count >= PLANT_STEPS_PER_CALL:
            step_count = 0
            with _lock:
                F = np.array([float(_state["flows"][j]) for j in range(NUM_LOOPS)])
            T = _plant.step(F, noise_std=0.0, enforce_total_flow=(10000.0, 80000.0))
            with _lock:
                _state["temperatures"] = [round(float(T[j]), 2) for j in range(NUM_LOOPS)]
                _history_temp.append({
                    "time": round(_state["time"], 2),
                    "temperatures": list(_state["temperatures"]),
                })
                if len(_history_temp) > _MAX_HISTORY_TEMP:
                    _history_temp.pop(0)
                # 方案 A：与 controller_model 状态空间同步更新
                dF = F - _mpc_controller.F0
                _mpc_state_x[:] = _mpc_controller.A @ _mpc_state_x + _mpc_controller.B @ dF
                cascade = _state["control_mode"] == "mpc_cascade"
                if cascade:
                    T_sp = list(_state["T_setpoints"])
                    T_cur = list(_state["temperatures"])
                    x_copy = _mpc_state_x.copy()
                    F_ref = np.array([float(_state["setpoints"][j]) for j in range(NUM_LOOPS)])
            if cascade:
                _mpc_controller.update_state(x_copy)
                try:
                    F_sp = _mpc_controller.compute(np.array(T_sp), np.array(T_cur), F_ref=F_ref)
                    F_sp = [clamp(float(f), FLOW_MIN, FLOW_MAX) for f in F_sp]
                    with _lock:
                        _state["setpoints"] = F_sp
                except Exception:
                    pass

        time.sleep(SAMPLE_TIME)


_sim_thread = threading.Thread(target=run_simulation, daemon=True)
_sim_thread.start()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    with _lock:
        data = {
            "running": _state["running"],
            "setpoints": list(_state["setpoints"]),
            "flows": list(_state["flows"]),
            "controls": list(_state["controls"]),
            "temperatures": list(_state["temperatures"]),
            "time": _state["time"],
            "Kp": list(_state["Kp"]),
            "Ki": list(_state["Ki"]),
            "Kd": list(_state["Kd"]),
            "control_mode": _state["control_mode"],
            "T_setpoints": list(_state["T_setpoints"]),
            "history_flow": list(_history_flow),
            "history_temp": list(_history_temp),
        }
    return jsonify(data)


@app.route("/api/start", methods=["POST"])
def api_start():
    with _lock:
        _state["running"] = True
        _state["integrals"] = [0.0] * NUM_LOOPS
        _state["last_errors"] = [0.0] * NUM_LOOPS
        _state["last_times"] = [None] * NUM_LOOPS
        _history_flow.clear()
        _history_temp.clear()
    _plant.reset()
    _valve_flow.reset()
    _mpc_state_x[:] = 0.0
    with _lock:
        _state["flows"] = list(_valve_flow.get_output())
        current_T = _plant.get_output()
        _state["temperatures"] = [float(x) for x in current_T]
        temps = list(_state["temperatures"])
        _history_temp.append({"time": 0.0, "temperatures": temps})
        _history_temp.append({"time": 1.0, "temperatures": temps})
    return jsonify({"ok": True})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    with _lock:
        _state["running"] = False
    return jsonify({"ok": True})


@app.route("/api/setpoint", methods=["POST"])
def api_setpoint():
    data = request.get_json() or {}
    try:
        with _lock:
            if "setpoints" in data:
                sps = [float(x) for x in data["setpoints"]]
                for i in range(min(len(sps), NUM_LOOPS)):
                    _state["setpoints"][i] = clamp(sps[i], FLOW_MIN, FLOW_MAX)
            elif "index" in data and "setpoint" in data:
                i = int(data["index"])
                if 0 <= i < NUM_LOOPS:
                    _state["setpoints"][i] = clamp(float(data["setpoint"]), FLOW_MIN, FLOW_MAX)
            return jsonify({"ok": True, "setpoints": list(_state["setpoints"])})
    except (TypeError, ValueError, IndexError):
        return jsonify({"ok": False, "error": "无效设定值"}), 400


@app.route("/api/params", methods=["POST"])
def api_params():
    data = request.get_json() or {}
    try:
        with _lock:
            if "index" in data:
                i = int(data["index"])
                if 0 <= i < NUM_LOOPS:
                    if "Kp" in data:
                        _state["Kp"][i] = max(0.0, float(data["Kp"]))
                    if "Ki" in data:
                        _state["Ki"][i] = max(0.0, float(data["Ki"]))
                    if "Kd" in data:
                        _state["Kd"][i] = max(0.0, float(data["Kd"]))
            else:
                if isinstance(data.get("Kp"), (list, tuple)):
                    for i in range(min(len(data["Kp"]), NUM_LOOPS)):
                        _state["Kp"][i] = max(0.0, float(data["Kp"][i]))
                elif "Kp" in data:
                    v = max(0.0, float(data["Kp"]))
                    for i in range(NUM_LOOPS):
                        _state["Kp"][i] = v
                if isinstance(data.get("Ki"), (list, tuple)):
                    for i in range(min(len(data["Ki"]), NUM_LOOPS)):
                        _state["Ki"][i] = max(0.0, float(data["Ki"][i]))
                elif "Ki" in data:
                    v = max(0.0, float(data["Ki"]))
                    for i in range(NUM_LOOPS):
                        _state["Ki"][i] = v
                if isinstance(data.get("Kd"), (list, tuple)):
                    for i in range(min(len(data["Kd"]), NUM_LOOPS)):
                        _state["Kd"][i] = max(0.0, float(data["Kd"][i]))
                elif "Kd" in data:
                    v = max(0.0, float(data["Kd"]))
                    for i in range(NUM_LOOPS):
                        _state["Kd"][i] = v
            return jsonify({
                "ok": True,
                "Kp": list(_state["Kp"]),
                "Ki": list(_state["Ki"]),
                "Kd": list(_state["Kd"]),
            })
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "无效参数"}), 400


@app.route("/api/reset", methods=["POST"])
def api_reset():
    with _lock:
        _state["running"] = False
        _state["setpoints"] = _init_setpoints()
        _state["flows"] = _init_flows()
        _state["controls"] = _init_flows()
        _state["time"] = 0.0
        _state["integrals"] = [0.0] * NUM_LOOPS
        _state["last_errors"] = [0.0] * NUM_LOOPS
        _state["last_times"] = [None] * NUM_LOOPS
        _state["temperatures"] = [500.0] * NUM_LOOPS
        _history_flow.clear()
        _history_temp.clear()
    _plant.reset()
    _valve_flow.reset()
    _mpc_state_x[:] = 0.0
    return jsonify({"ok": True})


@app.route("/api/mpc/compute", methods=["POST"])
def api_mpc_compute():
    """调用 MPC 控制器：输入温度设定 T_sp，返回流量设定 F_sp。便于后续控制器升级替换。"""
    data = request.get_json() or {}
    try:
        T_sp = data.get("T_sp")
        if T_sp is None or len(T_sp) != NUM_LOOPS:
            return jsonify({"ok": False, "error": "缺少或无效 T_sp（需 4 个温度设定值）"}), 400
        T_sp = np.array([float(x) for x in T_sp])
        with _lock:
            T_current = np.array([float(_state["temperatures"][i]) for i in range(NUM_LOOPS)])
            if "T_current" in data and len(data["T_current"]) == NUM_LOOPS:
                T_current = np.array([float(x) for x in data["T_current"]])
            x_use = _mpc_state_x.copy()
            if "x" in data and data["x"] is not None and len(data["x"]) == _mpc_controller.nx:
                x_use = np.array([float(v) for v in data["x"]])
        _mpc_controller.update_state(x_use)
        F_sp = _mpc_controller.compute(T_sp, T_current)
        F_sp = [clamp(float(f), FLOW_MIN, FLOW_MAX) for f in F_sp]
        return jsonify({"ok": True, "F_sp": F_sp})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/control_mode", methods=["POST"])
def api_control_mode():
    """设置控制模式：pid_only（仅 PID，流量设定手动）| mpc_cascade（MPC+PID 串级，温度设定）"""
    data = request.get_json() or {}
    mode = data.get("control_mode")
    if mode not in ("pid_only", "mpc_cascade"):
        return jsonify({"ok": False, "error": "control_mode 需为 pid_only 或 mpc_cascade"}), 400
    with _lock:
        _state["control_mode"] = mode
    return jsonify({"ok": True, "control_mode": mode})


@app.route("/api/t_setpoints", methods=["POST"])
def api_t_setpoints():
    """设置 4 路温度设定值（串级模式下由 MPC 据此求解流量设定）。"""
    data = request.get_json() or {}
    try:
        with _lock:
            if "T_setpoints" in data:
                sps = [float(x) for x in data["T_setpoints"]]
                for i in range(min(len(sps), NUM_LOOPS)):
                    _state["T_setpoints"][i] = max(0.0, min(600.0, sps[i]))
            elif "index" in data and "value" in data:
                i = int(data["index"])
                if 0 <= i < NUM_LOOPS:
                    _state["T_setpoints"][i] = max(0.0, min(600.0, float(data["value"])))
            return jsonify({"ok": True, "T_setpoints": list(_state["T_setpoints"])})
    except (TypeError, ValueError, IndexError):
        return jsonify({"ok": False, "error": "无效温度设定值"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
