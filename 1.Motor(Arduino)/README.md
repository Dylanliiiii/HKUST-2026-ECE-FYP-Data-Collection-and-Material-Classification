# Motor 模块说明（Arduino）

本目录用于三轴滑台（X/Y/Z）执行控制，核心程序为：

- `sketch_jan23a/sketch_jan23a.ino`

## 功能概述

该程序运行在 Arduino + CNC Shield V3 上，接收串口单字符命令并驱动步进电机：

- `D`：Z 轴下移
- `U`：Z 轴上移
- `X`：X 轴正向
- `x`：X 轴反向
- `Y`：Y 轴正向
- `y`：Y 轴反向
- `S`：全部停止

`2.Sensor(Vscode)/force_control_collect.py` 就是通过这些命令控制位移节奏。

## 依赖

- Arduino IDE
- `AccelStepper` 库

## 上传步骤

1. 在 Arduino IDE 打开：

`1.Motor(Arduino)/sketch_jan23a/sketch_jan23a.ino`

2. 安装并确认 `AccelStepper` 库可用。  
3. 选择正确开发板与串口后上传。  
4. 打开串口监视器（115200）可看到启动信息：`XYZ executor ready`。

## 关键参数（代码内）

- `STEPS_PER_MM = 1600.0`
- `STEP_MM = 1`
- X/Y 轴速度与加速度：`8000`
- Z 轴速度与加速度：`1000`

如更换丝杆导程、细分或驱动器参数，需要同步调整 `STEPS_PER_MM`。

## 与 Sensor 模块联动

- 默认通信波特率：`115200`
- `force_control_collect.py` 默认 Arduino 串口：`COM5`

如果电脑识别串口不同，请在 `2.Sensor(Vscode)/force_control_collect.py` 中修改 `ARDUINO_PORT`。

## 安全建议

- 首次调试时先降低速度和行程，确认运动方向正确。
- 设置机械限位或预留软限位，避免超程撞击。
- 异常时发送 `S` 或断电急停。
