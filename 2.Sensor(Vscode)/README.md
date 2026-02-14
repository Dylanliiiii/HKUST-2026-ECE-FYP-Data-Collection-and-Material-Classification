# 材料分类系统（Sensor 模块）

基于触觉传感器的材料识别与分类系统，包含数据采集、预处理、特征提取、模型训练和现场预测流程。

## 项目概述

本模块对应 FYP 第一阶段中“数据采集与材料分类”部分，核心目标：
- 采集材料接触过程中的力学信号（Fx, Fy, Fz）
- 提取可用于区分材料的统计与物理特征
- 训练分类器（SVM / Random Forest / KNN）
- 对现场新样本做批量预测

## 目录结构

```text
2.Sensor(Vscode)/
├── Config.py
├── check_system.py
├── force_control_collect.py
├── main_pipeline.py
├── predict_pipeline.py
├── requirements.txt
├── prediction_results.csv
├── modules/
│   ├── preprocess.py
│   ├── feature_extraction.py
│   ├── train_classifier.py
│   ├── visualize.py
│   └── predict.py
├── data_raw/
├── data_preprocess/
├── data_features/
├── models/
├── prediction_raw/
├── prediction_preprocessed/
└── visualizations/
```

## 快速开始

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 检查环境

```bash
python check_system.py
```

3. 运行完整流程（预处理 -> 特征 -> 可视化 -> 训练 -> 测试预测）

```bash
python main_pipeline.py
```

4. 运行现场预测流程

```bash
python predict_pipeline.py
```

可选参数：

```bash
python predict_pipeline.py --model svm
python predict_pipeline.py --model rf
python predict_pipeline.py --model knn
python predict_pipeline.py --skip-preprocess
```

## 核心脚本用途

- `force_control_collect.py`：采集传感器数据并保存 CSV（默认写入 `data_raw/`）
- `main_pipeline.py`：一键跑训练主流程并保存模型
- `predict_pipeline.py`：读取 `prediction_raw/`，预处理后批量预测
- `check_system.py`：检查环境、依赖、目录和核心文件完整性

## 数据与产物（当前仓库快照）

以下数量来自当前仓库文件（2026-02-14）：

- `data_raw/`：75 个文件（5 种材料，各 15 份）
- `data_preprocess/`：75 个文件
- `data_features/`：1 个文件（`features_global.csv`）
- `models/`：3 个模型（`classifier_svm.pkl` / `classifier_rf.pkl` / `classifier_knn.pkl`）
- `prediction_raw/`：6 个文件
- `prediction_preprocessed/`：6 个文件
- `visualizations/`：5 张图
- `prediction_results.csv`：现场预测汇总结果

## 特征说明

`modules/feature_extraction.py` 目前输出 10 个数值特征：

- 核心物理特征（6）：`k_eff`, `Fz_peak`, `mu_mean`, `mu_std`, `slip`, `micro`
- 额外统计特征（4）：`Fz_mean`, `Fz_std`, `Ft_mean`, `Ft_std`

标签列为 `material`。

## 关键配置

`Config.py` 中维护主要路径：

- `RAW_DIR`, `PREPROCESS_DIR`, `FEATURES_DIR`
- `MODELS_DIR`, `VISUALIZATIONS_DIR`
- `PREDICTION_RAW_DIR`, `PREDICTION_PREPROCESS_DIR`

`force_control_collect.py` 的默认关键参数：

- 串口：`SENSOR_PORT='COM4'`, `ARDUINO_PORT='COM5'`, `BAUD=115200`
- 接触阈值：`FZ_CONTACT=2.2`
- 力上限：`FZ_MAX=10.0`
- 压入深度：`INDENT_MM=3.0`
- 滑动距离：`SLIDE_MM=20.0`

## 常见问题

### 1) 依赖缺失

```bash
pip install -r requirements.txt
python check_system.py
```

### 2) 预测时报“模型文件不存在”

先训练模型再预测：

```bash
python main_pipeline.py
python predict_pipeline.py --model svm
```

### 3) 串口无数据或被占用

- 检查 `force_control_collect.py` 的 `COM` 口配置
- 关闭可能占用串口的上位机/串口监视器
- 确认传感器和 Arduino 已上电

## 更新记录（精简）

### 2026-02-14
- README 按当前代码与目录重整
- 删除重复章节（含“分享给队友”等）
- 数据统计更新为 75 样本（每类 15）

### 2026-02-13
- 新增现场预测流程 `predict_pipeline.py`
- 新增 `prediction_raw/` 与 `prediction_preprocessed/` 目录支持
