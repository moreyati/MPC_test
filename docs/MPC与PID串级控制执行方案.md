# MPC + PID 串级控制 执行方案

## 1. 目标

- 在 **app.py** 中新增**调用 MPC 控制器的接口**，便于后续控制器升级与替换。
- 将仿真系统做成 **MPC + PID 串级控制**：用户给定**四个温度设定值**，由 **MPC** 求解**最优流量设定值**并下发给 **PID**，**PID** 对流量设定值进行跟踪，流量进入**流量-温度模型**得到温度，从而实现对温度的控制。

---

## 2. 现有组件与依赖

| 组件 | 文件 | 作用 |
|------|------|------|
| 流量-温度被控对象 | controlled_model.py | MIMOFOPDTPlant：4 路流量 F → 4 路温度 T |
| MPC 控制器 | MPC_controller.py | PriorityMPCController：T_sp、T_current（及状态 x）→ 求解得到 F_sp |
| MPC 用状态空间模型 | controller_model.py | build_mimo_fopdt_ss(dt)：得到 A,B,C,D 及 meta（MPC 预测用） |
| 仿真与 API | app.py | 4 路 PID 跟踪流量设定值，流量送 plant 得温度；提供 /api/state、/api/setpoint 等 |

**MPC 控制器接口（当前）：**

- `update_state(x_current)`：更新内部状态 x（与 controller_model 的状态空间维度一致）。
- `compute(T_sp, T_current)`：输入 4 路温度设定值、当前 4 路温度；输出 4 路流量设定值 F_sp。内部使用偏差形式、总流量与单路流量约束，只执行第一步控制量。

**注意**：MPC 使用的状态 x 来自 **controller_model** 的离散状态空间，与 **controlled_model**（MIMOFOPDTPlant）的内部实现（4×4 一阶 + 纯滞后缓冲）不同。执行时需保证“MPC 所用状态”与仿真对象一致或可观测，见下文 4.2。

---

## 3. 串级结构（最终效果）

```
用户输入 4 路温度设定值 T_sp
        ↓
   ┌─────────────────┐
   │  MPC 控制器      │  ← 可选：当前温度 T、状态 x
   │  compute(...)    │
   └────────┬────────┘
            │ 输出 F_sp（4 路流量设定值）
            ↓
   ┌─────────────────┐
   │  4× PID 回路     │  流量设定值 F_sp → 跟踪得到 4 路流量 F
   └────────┬────────┘
            │ 4 路实际流量 F
            ↓
   ┌─────────────────┐
   │  流量-温度模型   │  F → 4 路温度 T
   └────────┬────────┘
            │ 4 路温度 T（反馈给 MPC 与界面）
            ↓
   显示 / 下一拍 MPC 再算
```

- **外环（温度环）**：MPC 根据温度设定 T_sp 与当前温度 T（及状态 x）计算**流量设定值 F_sp**。
- **内环（流量环）**：现有 PID 根据**流量设定值 F_sp** 跟踪得到实际流量 F；F 进入 plant 得到 T，完成闭环。

用户侧：只设定**四个温度设定值**；流量设定值由 MPC 自动给出，无需手填（手填仅用于“纯 PID”模式或调试）。

---

## 4. 执行步骤（如何做）

### 4.1 在 app.py 中新增“MPC 调用接口”

**目的**：对外提供统一的 MPC 调用方式，便于后续换用不同 MPC 实现（如不同参数、不同求解器或不同控制器类）。

**建议接口形态**（REST，具体路径与字段可按实现微调）：

- **POST /api/mpc/compute**（或 `/api/controller/mpc`）
  - **请求体**：`{ "T_sp": [t1, t2, t3, t4], "T_current": [可选], "x": [可选] }`
    - `T_sp`：4 路温度设定值（℃），必填。
    - `T_current`：当前 4 路温度；若不传则由后端用当前仿真状态中的温度。
    - `x`：MPC 状态空间状态；若不传则由后端根据当前仿真状态或观测器得到（见 4.2）。
  - **响应**：`{ "ok": true, "F_sp": [f1, f2, f3, f4] }` 或失败时 `{ "ok": false, "error": "..." }`。

**后端逻辑概要**：

1. 解析 `T_sp`（必填），可选解析 `T_current`、`x`。
2. 若未传 `T_current`，从 `_state["temperatures"]` 读取。
3. 若未传 `x`，用“当前仿真状态 → MPC 状态”的约定方式得到 x（见 4.2）。
4. 调用 `_mpc_controller.update_state(x)`，再调用 `_mpc_controller.compute(T_sp, T_current)`，得到 `F_sp`。
5. 将 `F_sp` 限幅到合法流量范围后返回；可选：同时写回 `_state["setpoints"]`（见 4.3）。

**依赖与初始化**：

- 在 app.py 中 `from MPC_controller import PriorityMPCController`（或通过“控制器工厂”间接引用，便于以后换控制器）。
- 在应用启动或首次使用时创建 MPC 实例（例如 `_mpc_controller = PriorityMPCController(dt=PLANT_DT, ...)`），与现有 `_plant` 的采样周期或仿真步长一致为宜。

这样，后续只要替换“调用谁、传什么参数”，即可升级或更换 MPC，而不必改一堆散落逻辑。

### 4.2 MPC 状态 x 的获取方式（与 controller_model 一致）

MPC 的 `update_state(x_current)` 需要的是 **controller_model.build_mimo_fopdt_ss** 对应的状态向量 x（维度见 meta），而不是 MIMOFOPDTPlant 内部的 4×4 与 buffers。可选做法：

- **方案 A（推荐，若两模型一致）**：在仿真中**同时维护** controller_model 对应的离散状态（用同一套 A,B,C,D，每步用当前 F 和采样周期更新）；每步或每次调用 MPC 前，用该状态作为 `x_current` 传入。这样 MPC 与仿真共用同一状态空间，预测一致。
- **方案 B**：不维护状态，用**当前 (F, T) 与模型**构造一个“等效初始状态”或简单观测器（如零状态 + 用 T 反推），仅作过渡；后续再替换为与 plant 一致的状态维护或正式观测器。
- **方案 C**：若 controller_model 与 controlled_model 结构完全一致，可考虑从 controlled_model 暴露内部状态并**映射**到 controller_model 的 x（需要两边的状态顺序、延迟链定义一致并在文档中写明）。

执行文档阶段只需选定一种并写明“MPC 状态由 xxx 提供”；实现时再在 app.py 与（可选）controller_model/controlled_model 中落实。

### 4.3 串级闭环：何时调用 MPC、何时写回流量设定值

- **调用时机**：与流量-温度模型步长对齐，例如每 **PLANT_DT** 秒（或每完成一次 plant.step）调用一次 MPC，这样外环（温度/MPC）与内环（流量/PID）时间尺度清晰。
- **写回流量设定值**：MPC 接口返回的 `F_sp` 除通过 API 返回给前端外，若当前为“串级模式”，应在仿真线程内将 `_state["setpoints"]` 更新为本次 `F_sp`（经限幅），这样内环 PID 自然跟踪 MPC 给出的流量设定值，形成串级。
- **模式区分**：可增加“控制模式”状态（如 `"pid_only"` / `"mpc_cascade"`）。`"pid_only"` 时保留当前行为（用户或前端直接设流量设定值）；`"mpc_cascade"` 时由 MPC 写回流量设定值，用户只设温度设定值。模式可通过配置或 API 切换。

### 4.4 前端与状态接口的配合（思路）

- 若为串级模式，前端可增加**四个温度设定值**输入（T_sp），并可选显示“当前 MPC 下发的流量设定值”（即 `_state["setpoints"]`）。
- `/api/state` 中可增加字段：`"T_setpoints"`（温度设定）、`"control_mode"`（如 `"pid_only"` / `"mpc_cascade"`），便于前端区分显示与编辑权限（例如串级时只允许改温度设定，流量设定只读显示）。
- 前端不直接调用 MPC 求解；由后端在仿真循环或定时任务中按 4.3 调用 MPC 并写回流量设定值即可。若需“单步测试 MPC”，可单独调用 **POST /api/mpc/compute**，返回的 `F_sp` 仅用于查看或手动写入设定值（调试用）。

---

## 5. 最终达成的效果（简要）

- **接口**：app.py 提供 **POST /api/mpc/compute**（或等价路径），传入 T_sp（及可选的 T_current、x），返回 F_sp；内部通过统一入口调用现有 MPC_controller，便于后续替换或升级控制器。
- **串级控制**：
  - 用户设定**四个温度设定值**；
  - 在串级模式下，每隔 PLANT_DT 或每个 plant 步，后端用当前 T 和状态 x 调用 MPC，得到 F_sp，并写回 `_state["setpoints"]`；
  - 现有 4 路 PID 继续用 `_state["setpoints"]` 作为流量设定值进行跟踪，输出 4 路流量 F；
  - F 进入 MIMOFOPDTPlant 得到 4 路温度 T，形成“温度设定 → MPC → 流量设定 → PID → 流量 → 对象 → 温度”的闭环。
- **兼容与扩展**：保留纯 PID 模式（用户直接设流量设定值）；通过“控制模式”和统一 MPC 调用接口，后续可更换或升级 MPC 而不改动业务逻辑，仅改“谁被调用、参数从哪来”。

---

## 6. 小结表

| 项目 | 内容 |
|------|------|
| 新增接口 | POST /api/mpc/compute，入参 T_sp（必填）、T_current/x（可选），出参 F_sp |
| 调用对象 | MPC_controller.PriorityMPCController（或通过工厂/配置指定，便于升级） |
| 状态 x | 与 controller_model 状态空间一致，由仿真维护或观测得到（见 4.2） |
| 串级逻辑 | 每 PLANT_DT 或每 plant 步调用 MPC，将返回的 F_sp 写回流量设定值，PID 跟踪 F_sp |
| 用户输入 | 串级模式下仅输入 4 路温度设定值；流量设定值由 MPC 自动下发 |
| 最终效果 | 温度设定 → MPC 求 F_sp → PID 跟踪 F_sp → 流量 → 对象 → 温度，实现基于 MPC+PID 串级的温度控制 |

按上述步骤实现后，即可在现有仿真系统上形成“MPC + PID 串级控制”，并保留清晰的 MPC 调用接口供后续控制器升级与修改。
