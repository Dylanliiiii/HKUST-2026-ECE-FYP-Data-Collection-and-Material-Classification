# 第一阶段：数据采集与材料分类

本仓库为 HKUST FYP 项目的第一阶段代码，主要包含：
- 电机控制（Arduino）
- 触觉传感器数据采集与材料分类（Python）

## 仓库结构

```text
1.Data Collection and Material Classification/
├── 1.Motor(Arduino)/
│   └── sketch_jan23a/
│       └── sketch_jan23a.ino
├── 2.Sensor(Vscode)/
│   ├── main_pipeline.py
│   ├── predict_pipeline.py
│   ├── requirements.txt
│   ├── README.md
│   └── modules/
├── .gitignore
└── README.md
```

## 快速开始

1. 进入传感器模块目录：

```bash
cd "2.Sensor(Vscode)"
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 检查运行环境：

```bash
python check_system.py
```

4. 运行完整训练与评估流程：

```bash
python main_pipeline.py
```

5. 运行预测流程：

```bash
python predict_pipeline.py
```

## 文档说明

- 详细流程、数据集说明与故障排查请查看：
  - `2.Sensor(Vscode)/README.md`

## 备注

- 根目录 `.gitignore` 已统一配置 Python 缓存、临时文件和模型产物忽略规则。
- 硬件串口等参数请查看：
  - `2.Sensor(Vscode)/force_control_collect.py`
