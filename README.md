# 四回路 PID + 流量-温度仿真 Web 程序

4 个 PID 流量回路 + 1 个 4×4 流量-温度模型（MIMO FOPDT）。每条回路可设流量设定值与 PID 参数；4 路流量进入模型后输出 4 路温度并显示。

## 环境

- Python 3.8+
- 依赖见 `requirements.txt`（Flask、numpy）

## 安装与运行

```bash
pip install -r requirements.txt
python app.py
```

浏览器打开 **http://127.0.0.1:5000**。

## 功能说明

- **流量设定值**：4 路，0–60 000 Nm³/h，初始每路 12 500（总 50 000），可随时修改。
- **PID 参数**：每路独立 Kp、Ki、Kd（≥0），点击「应用全部 PID 参数」生效。
- **仿真控制**：开始 / 停止 / 重置。重置将 4 路设定值与流量恢复为 12 500，温度恢复为 500 ℃，并清空曲线。
- **显示**：4 路设定值、4 路流量、4 路温度；流量曲线（设定值 + 实际流量）、温度曲线。

## 文件说明

- `app.py`：Flask 后端，4 路 PID + 一阶过程 + MIMOFOPDTPlant，REST API。
- `controlled_model.py`：4×4 流量-温度 MIMO 模型（FOPDT）。
- `templates/index.html`：前端（4 路设定值/PID、双图表）。
- `requirements.txt`：Python 依赖。
