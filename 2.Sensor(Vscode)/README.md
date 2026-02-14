# 材料分类系统 (Material Classification System)

基于触觉传感器的材料识别与分类系统

---

## 📋 项目概述

本项目是HKUST毕设项目的一部分（YX02a-25），实现了基于三轴滑台和触觉传感器的材料分类系统。通过采集材料表面的力学特征数据，使用机器学习算法（SVM/Random Forest/KNN）进行材料识别。

**当前状态**: ✅ 已完成数据采集、预处理、特征提取、模型训练和预测功能

---

## 🎯 项目阶段

### ✅ 已完成（Objective 1 & 2）

- [x] **硬件搭建**: 三轴滑台 + AgileReach触觉传感器
- [x] **数据采集**: 5种材料 × 15次采集 = 75个样本
- [x] **数据预处理**: Fz阈值过滤、异常值检测、数据平滑
- [x] **特征提取**: 6个物理特征（k_eff, Fz_peak, mu_mean, mu_std, slip, micro）
- [x] **模型训练**: SVM/RF/KNN三种分类器
- [x] **模型评估**: 准确率测试、混淆矩阵、分类报告
- [x] **数据可视化**: 特征分布图、散点图矩阵、PCA/t-SNE降维
- [x] **预测功能**: 实时预测、批量预测

### ⬜ 待完成（Objective 3）

- [ ] **机械臂集成**: ROS2驱动、夹爪力控制
- [ ] **强化学习**: 仿真环境、策略训练
- [ ] **实时分类**: 边缘设备部署（Jetson Nano）

---

## 📁 项目结构

```
Sensor/
├── Config.py                      # 配置文件（路径配置）
├── force_control_collect.py       # 数据采集程序
├── main_pipeline.py               # 主流程控制文件 ⭐
├── predict_pipeline.py            # 现场预测流程（真正开始预测）
├── check_system.py                # 环境检查脚本
├── requirements.txt               # Python依赖列表
├── prediction_results.csv         # 预测结果汇总（自动生成）
├── README.md                      # 本文档
│
├── modules/                       # 功能模块目录
│   ├── __init__.py                # 包初始化
│   ├── preprocess.py              # 数据预处理
│   ├── feature_extraction.py      # 特征提取
│   ├── train_classifier.py        # 模型训练
│   ├── visualize.py               # 数据可视化
│   └── predict.py                 # 预测功能
│
├── data_raw/                      # 原始数据（75个CSV文件）
│   ├── Material_Wood_raw_1.csv
│   ├── Material_Wood_raw_2.csv
│   └── ... (每种材料15个文件)
│
├── data_preprocess/               # 预处理数据（自动生成）
├── data_features/                 # 特征数据（自动生成）
├── models/                        # 训练好的模型（自动生成）
├── visualizations/                # 可视化图表（自动生成）
├── prediction_raw/                # 现场预测输入（原始CSV）
└── prediction_preprocessed/       # 现场预测中间结果（自动生成）
```

---

## 🚀 快速开始

### 第一次使用（推荐流程）

#### 步骤1: 检查环境

```bash
cd cd "2.Sensor(Vscode)"
python check_system.py
```

这个脚本会自动检查：
- ✓ Python版本（需要3.7+）
- ✓ 所有依赖包是否安装
- ✓ 项目文件是否完整
- ✓ 数据文件是否存在
- ✓ 模块能否正常导入

**如果检查全部通过**，直接跳到步骤3。

#### 步骤2: 安装依赖（如果步骤1失败）

```bash
pip install -r requirements.txt
```

安装完成后，再次运行 `python check_system.py` 验证。

需要的包：
- numpy, pandas, scipy - 数据处理
- scikit-learn - 机器学习
- matplotlib, seaborn - 可视化
- pyserial, keyboard - 数据采集（可选，仅采集时需要）

#### 步骤3: 运行完整流程

```bash
python main_pipeline.py
```

这个命令会自动执行：
1. ✅ 数据预处理（74个原始文件 → 74个预处理文件）
2. ✅ 特征提取（74个样本，每个10个特征）
3. ✅ 数据可视化（5张图表保存到 `visualizations/`）
4. ✅ 模型训练（SVM/RF/KNN对比）
5. ✅ 预测测试（测试集准确率评估）

**预计运行时间**: 2-5分钟

**输出结果**：
- `data_preprocess/` - 74个预处理CSV文件
- `data_features/features_global.csv` - 提取的特征
- `models/` - 3个训练好的模型（.pkl文件）
- `visualizations/` - 5张可视化图表（.png文件）

#### 步骤4: 真正开始预测（新增）

把现场采集到的待预测 CSV 放到 `prediction_raw/`，然后运行：

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

输出结果：
- `prediction_preprocessed/` - 预测前预处理文件
- `prediction_results.csv` - 每个文件的预测材料与置信度

---

## 👥 分享给队友

### 打包方式1: 不包含数据（推荐）

**适用场景**：数据文件太大，或队友需要自己采集数据

**打包内容**：
```
Sensor/
├── *.py（所有Python脚本）
├── modules/（所有模块）
├── requirements.txt
├── README.md
└── .gitignore
```

**队友使用步骤**：
1. 解压缩到本地
2. `cd Sensor`
3. `python check_system.py` - 会提示缺少数据
4. 自己采集数据或从你那里获取 `data_raw/` 文件夹
5. `python main_pipeline.py`

### 打包方式2: 包含原始数据

**适用场景**：直接给队友完整的数据集

**打包内容**：
```
Sensor/
├── 所有代码文件
├── data_raw/（74个原始数据CSV）
└── 其他文件
```

**队友使用步骤**：
1. 解压缩到本地
2. `cd Sensor`
3. `python check_system.py` - 应该全部通过
4. `python main_pipeline.py` - 直接运行

### 打包方式3: 包含训练好的模型

**适用场景**：队友只需要预测功能，不需要重新训练

**额外包含**：
```
models/
├── classifier_svm.pkl
├── classifier_rf.pkl
└── classifier_knn.pkl
```

**队友使用**：
```python
from modules.predict import MaterialPredictor
predictor = MaterialPredictor(classifier_type='svm')
result = predictor.predict_from_raw_data('新数据.csv')
```

---

## 🔄 工作流程详解

### 完整工作流（从数据采集到预测）

```
1. 数据采集
   ↓ force_control_collect.py
   data_raw/*.csv (原始数据)
   
2. 数据预处理
   ↓ modules/preprocess.py
   data_preprocess/*.csv (清洗后的数据)
   
3. 特征提取
   ↓ modules/feature_extraction.py
   data_features/features_global.csv (特征向量)
   
4. 模型训练
   ↓ modules/train_classifier.py
   models/*.pkl (训练好的模型)
   
5. 数据可视化
   ↓ modules/visualize.py
   visualizations/*.png (分析图表)
   
6. 预测应用
   ↓ modules/predict.py
   预测结果

7. 现场批量预测（新增）
   ↓ predict_pipeline.py
   prediction_raw/*.csv → prediction_preprocessed/*.csv → prediction_results.csv
```

### 只运行部分流程

#### 场景A: 已有模型，只做预测

```python
from modules.predict import MaterialPredictor

# 加载已训练的模型
predictor = MaterialPredictor(classifier_type='svm')

# 预测新数据（会自动预处理和提取特征）
result = predictor.predict_from_raw_data('data_raw/新材料_raw.csv')
print(f"预测材料: {result}")
```

#### 场景B: 重新训练模型

```python
from modules.train_classifier import MaterialClassifier

# 创建分类器
classifier = MaterialClassifier(classifier_type='svm')

# 加载特征数据
X, y = classifier.load_features('features_global.csv')

# 准备数据
X_train, X_test, y_train, y_test = classifier.prepare_data(X, y)

# 训练
classifier.train(X_train, y_train, use_grid_search=True)

# 评估
classifier.evaluate(X_test, y_test)

# 保存
classifier.save_model()
```

#### 场景C: 只生成可视化

```python
from modules.visualize import DataVisualizer

visualizer = DataVisualizer()
visualizer.plot_all('features_global.csv')
```

---

## 🎯 不同使用场景

---

## 📊 核心功能说明

### 1. 数据采集 (`force_control_collect.py`)

**功能**：
- 控制三轴滑台执行"Press-Hold-Slide"动作
- 实时采集传感器数据（Fx, Fy, Fz）
- 计算派生特征（Ft, mu, k_eff等）
- 保存为CSV文件

**使用方法**：
```python
# 编辑文件，修改材料标签
MATERIAL_LABEL = "Material_新材料_raw_1"

# 运行程序
python force_control_collect.py

# 操作：
# - 按 's' 开始自动采集
# - 按 'p' 暂停
# - Ctrl+C 退出
```

**注意**：
- 每种材料建议采集10-15次
- 文件命名格式：`Material_材料名_raw_序号.csv`

### 2. 数据预处理 (`modules/preprocess.py`)

**功能**：
- Fz阈值过滤（去除未接触数据，Fz > 2.2N）
- 异常值检测（IQR方法）
- 数据平滑（移动平均，可选）
- 计算派生特征（Ft, mu, dFz/dt）

**特点**：
- 自动处理所有原始文件
- 支持带序号的文件名（Material_XXX_raw_N.csv）

### 3. 特征提取 (`modules/feature_extraction.py`)

**提取的6个核心特征**（基于报告）：

| 特征 | 说明 | 物理意义 |
|------|------|---------|
| `k_eff` | 有效刚度 | 材料抗压缩能力 |
| `Fz_peak` | 峰值法向力 | 最大压缩力 |
| `mu_mean` | 平均摩擦系数 | 表面摩擦特性 |
| `mu_std` | 摩擦稳定性 | 摩擦一致性 |
| `slip` | 滑动不稳定强度 | 突发滑动强度 |
| `micro` | 微振动 | 表面纹理粗糙度 |

**方法**：
- `global`: 每个文件提取1个样本（整体特征）
- `sliding`: 每个文件提取多个样本（滑动窗口）

### 4. 模型训练 (`modules/train_classifier.py`)

**支持的分类器**：

| 分类器 | 特点 | 适用场景 | 预期准确率 |
|--------|------|---------|-----------|
| **SVM** | 小样本效果好 | 样本少（<100） | >90% |
| **Random Forest** | 鲁棒性强 | 样本多（>50） | >85% |
| **KNN** | 简单快速 | 快速原型 | >80% |

**训练过程**：
1. 数据分割（80%训练，20%测试）
2. 网格搜索优化参数
3. 交叉验证评估
4. 保存最佳模型

**输出**：
- 模型文件：`models/classifier_svm.pkl`
- 评估报告：准确率、混淆矩阵、分类报告

### 5. 数据可视化 (`modules/visualize.py`)

**生成的图表**：
1. **特征分布图** - 每个特征按材料分布
2. **散点图矩阵** - 特征两两关系（类似报告Figure 10）
3. **相关性矩阵** - 特征相关性热图
4. **PCA降维** - 2D可视化（主成分分析）
5. **t-SNE降维** - 2D可视化（非线性降维）

**输出位置**：`visualizations/` 目录

### 6. 预测功能 (`modules/predict.py`)

**预测方式**：
- 从原始数据预测（自动预处理+特征提取）
- 从预处理数据预测
- 批量预测多个文件

**示例**：
```python
from modules.predict import MaterialPredictor

# 加载模型
predictor = MaterialPredictor(classifier_type='svm')

# 预测单个文件
prediction = predictor.predict_from_raw_data('data_raw/Material_Test_raw.csv')
print(f"预测材料: {prediction}")

# 批量预测
results = predictor.predict_batch(file_list, file_type='raw')
```

### 7. 现场预测流程 (`predict_pipeline.py`)【新增】

**功能**：
- 读取 `prediction_raw/` 下的现场原始数据
- 自动预处理并保存到 `prediction_preprocessed/`
- 加载训练好的模型（SVM/RF/KNN）进行批量预测
- 输出汇总到 `prediction_results.csv`

**使用方式**：
```bash
# 默认：SVM + 预处理 + 预测
python predict_pipeline.py

# 指定模型
python predict_pipeline.py --model rf

# 跳过预处理，直接用 prediction_preprocessed 预测
python predict_pipeline.py --skip-preprocess
```

---

### 场景1：添加新材料

```bash
# 1. 采集新材料数据（10-15次）
python force_control_collect.py
# 修改 MATERIAL_LABEL = "Material_新材料_raw_1"
# 按 's' 开始，完成后修改为 "_2", "_3" 继续采集

# 2. 重新训练模型
python main_pipeline.py
```

### 场景2：更新现有材料数据

```bash
# 1. 继续采集同一材料（增加样本）
# 例如：Wood已有15个，现在采集第16个
# MATERIAL_LABEL = "Material_Wood_raw_16"

# 2. 重新运行流程
python main_pipeline.py
```

### 场景3：只预测新样本

```python
from modules.predict import MaterialPredictor

predictor = MaterialPredictor(classifier_type='svm')
result = predictor.predict_from_raw_data('新数据.csv')
print(f"预测材料: {result}")
```

### 场景4：批量预测多个样本

```python
from modules.predict import MaterialPredictor
import glob

predictor = MaterialPredictor(classifier_type='svm')

# 预测data_raw下所有新文件
new_files = glob.glob('data_raw/Material_Test_*.csv')
results = predictor.predict_batch(new_files, file_type='raw')

for result in results:
    print(f"{result['file']}: {result['prediction']}")
```

### 场景5：单独运行某个步骤

```python
# 只做预处理
from modules.preprocess import DataPreprocessor
preprocessor = DataPreprocessor()
preprocessor.process_all()

# 只做特征提取
from modules.feature_extraction import FeatureExtractor
extractor = FeatureExtractor()
features = extractor.process_all()

# 只做可视化
from modules.visualize import DataVisualizer
visualizer = DataVisualizer()
visualizer.plot_all()
```

### 场景6：现场数据一键批量预测（新增）

```bash
# 1) 把新采集数据放到 prediction_raw/
# 2) 执行现场预测流程
python predict_pipeline.py --model svm

# 3) 查看结果
# - prediction_results.csv
```

---

## 📈 当前数据集

### 采集的材料

| 材料 | 采集次数 | 文件数 | 样本数 |
|------|---------|--------|--------|
| Wood（木头） | 15次 | 15个 | 15个 |
| Metal（金属） | 15次 | 15个 | 15个 |
| POM（塑料） | 14次 | 14个 | 14个 |
| Silicone（硅胶） | 15次 | 15个 | 15个 |
| EPEfoam（泡沫） | 15次 | 15个 | 15个 |
| **总计** | **74次** | **74个** | **74个** |

### 数据采集参数

- **采样率**: 100 Hz
- **接触阈值**: Fz > 2.2 N
- **最大力**: Fz < 10.0 N
- **压痕深度**: 2.8 mm
- **滑动距离**: 20.0 mm
- **动作序列**: 下降 → 压痕 → 保持 → 滑动(X) → 滑动(Y) → 返回 → 上升

---

## 🔧 配置说明

### `Config.py`

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data_raw")           # 原始数据
PREPROCESS_DIR = os.path.join(BASE_DIR, "data_preprocess")  # 预处理数据
FEATURES_DIR = os.path.join(BASE_DIR, "data_features")      # 特征数据
MODELS_DIR = os.path.join(BASE_DIR, "models")               # 模型文件
VISUALIZATIONS_DIR = os.path.join(BASE_DIR, "visualizations")  # 可视化图表
PREDICTION_RAW_DIR = os.path.join(BASE_DIR, "prediction_raw")  # 现场预测输入
PREDICTION_PREPROCESS_DIR = os.path.join(BASE_DIR, "prediction_preprocessed")  # 现场预测预处理输出
```

### `force_control_collect.py` 关键参数

```python
SENSOR_PORT = 'COM4'      # 传感器串口
ARDUINO_PORT = 'COM5'     # Arduino串口
BAUD = 115200             # 波特率

FZ_CONTACT = 2.2          # 接触阈值 (N)
FZ_MAX = 10.0             # 最大力 (N)
INDENT_MM = 2.8           # 压痕深度 (mm)
SLIDE_MM = 20.0           # 滑动距离 (mm)
```

---

## ❓ 常见问题与解决方案

### Q1: `check_system.py` 报错依赖包缺失？

**解决**:
```bash
# 方案1: 直接安装所有依赖
pip install -r requirements.txt

# 方案2: 单独安装缺失的包
pip install numpy pandas scipy scikit-learn matplotlib seaborn

# 验证
python check_system.py
```

### Q2: 训练失败，提示样本数太少？

**错误信息**: `Cannot have number of splits n_splits=5 greater than the number of samples`

**原因**: 每种材料样本少于2个，或总样本数少于5个

**解决**: 
```bash
# 1. 检查数据文件数量
python check_system.py  # 查看"数据文件"部分

# 2. 每种材料至少采集5次（推荐10-15次）
python force_control_collect.py

# 3. 确保文件命名格式正确
# 正确: Material_Wood_raw_1.csv
# 错误: Wood_data.csv
```

### Q3: 预测准确率很低（<50%）？

**诊断步骤**:

1. **检查样本数**:
   ```bash
   python check_system.py  # 查看每种材料样本数
   ```
   - 每种材料至少10个样本
   - 总样本数至少50个

2. **查看可视化**:
   ```bash
   python main_pipeline.py  # 运行后查看visualizations/
   ```
   - 打开 `scatter_matrix.png`
   - 检查不同材料是否有明显分离
   - 如果重叠严重，说明特征区分度不够

3. **检查数据质量**:
   - 采集时是否接触充分（Fz > 2.2N）
   - 不同材料采集条件是否一致
   - 传感器是否正常工作

**解决方案**:
- ✅ 增加采集次数（每种材料15-20次）
- ✅ 重新采集质量不好的数据
- ✅ 检查材料标签是否正确

### Q4: 模块导入失败 `No module named 'modules'`？

**解决**:
```bash
# 确保在正确的目录
cd Sensor  # 必须在Sensor目录下
pwd        # Windows用 cd（不带参数），Linux/Mac用 pwd

# 检查目录结构
ls -l      # Windows用 dir

# 应该看到:
# - modules/（文件夹）
# - main_pipeline.py
# - Config.py
# 等文件
```

### Q5: 数据采集时传感器无数据？

**检查清单**:
1. ✓ 串口号是否正确（`SENSOR_PORT = 'COM4'`）
2. ✓ 传感器是否上电
3. ✓ 波特率是否匹配（`BAUD = 115200`）
4. ✓ 是否有其他程序占用串口

**测试串口**:
```python
import serial
sensor = serial.Serial('COM4', 115200, timeout=1)
print(sensor.read(100))  # 应该有数据输出
```

### Q6: 可视化图表显示中文乱码？

**解决** - 修改 `modules/visualize.py`:
```python
# 第40行左右
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
```

### Q7: 预测时提示 "模型文件不存在"？

**原因**: 还没有训练模型

**解决**:
```bash
# 先运行一次完整流程，生成模型
python main_pipeline.py

# 然后才能用预测功能
```

### Q8: 如何添加新特征？

**步骤**:

1. **修改特征提取** - `modules/feature_extraction.py`:
```python
def extract_features_from_window(self, df_window):
    features = {}
    
    # 原有6个特征
    features['k_eff'] = np.mean(np.abs(df_window['dFz_dt']))
    # ...
    
    # 添加新特征（示例：最大切向力）
    features['Ft_max'] = np.max(df_window['Ft'])
    
    # 添加新特征（示例：力变化率）
    features['dFt_dt'] = np.std(np.diff(df_window['Ft']))
    
    return features
```

2. **重新运行流程**:
```bash
python main_pipeline.py
```

新特征会自动用于训练。

### Q9: 如何调整模型参数？

**修改训练参数** - `modules/train_classifier.py`:

```python
# 第125-130行，修改SVM参数搜索范围
param_grid = {
    'C': [0.1, 1, 10, 100],        # 正则化参数
    'gamma': [0.001, 0.01, 0.1],   # 核函数系数
    'kernel': ['rbf']               # 只用RBF核
}
```

### Q10: 队友运行时报错路径问题？

**原因**: 绝对路径不同

**解决**: 使用 `Config.py` 中的相对路径，不要硬编码绝对路径

```python
# ✗ 错误
data_path = "D:/Education/HKUST/FYP_Code/Sensor/data_raw"

# ✓ 正确
import Config
data_path = Config.RAW_DIR
```

### Q11: `predict_pipeline.py` 报“模型加载失败”或“模型文件不存在”？

**原因**: 还没有训练模型，或 `models/` 下缺少对应分类器文件。  

**解决**:
```bash
# 先训练模型
python main_pipeline.py

# 再执行现场预测
python predict_pipeline.py --model svm
```

---

## 📚 技术细节

### 特征提取算法

```python
# 1. 有效刚度
k_eff = mean(|dFz/dt|)

# 2. 峰值法向力
Fz_peak = max(Fz)

# 3. 平均摩擦系数
mu_mean = mean(Ft / Fz)

# 4. 摩擦稳定性
mu_std = std(Ft / Fz)

# 5. 滑动不稳定强度
slip = max(|dFz/dt|)

# 6. 微振动
micro = RMS(Ft - mean(Ft))
```

### SVM参数（网格搜索后的最优值）

```python
SVC(
    kernel='rbf',      # 径向基核函数
    C=10,              # 正则化参数
    gamma=0.1,         # 核函数系数
    probability=True   # 输出概率
)
```

---

## 📖 参考文献

基于HKUST FYP报告（YX02a-25）:
- Objective 1: Multi-Task Tactile Perception
- Objective 2: Material Classification and Grip Adaptation
- Task 2.2: Behavior-Based Mechanical Feature Extraction

相关论文：
- [1] Z. Zhao et al., "Embedding high-resolution touch across robotic hands enables adaptive human-like grasping," Nature Machine Intelligence, 2025.
- [2] Z. Kappassov et al., "Tactile sensing in dexterous robot hands," Robotics and Autonomous Systems, 2015.
- [4] Y. Wang et al., "Multimodal tactile sensing fused with vision for dexterous robotic housekeeping," Nature Communications, 2024.

---

## 👥 项目成员

- **LI Yuntong** (20944800) - 硬件搭建、数据采集、特征提取
- **Guo Ruopu** (20945464) - 电机控制、模型训练、系统集成

**指导老师**: Yang Xiong

---

## 🛠️ 故障排查指南

### 运行前检查清单

```bash
# 1. 环境检查
python check_system.py

# 2. 查看Python版本
python --version  # 应该是 3.7+

# 3. 查看安装的包
pip list | grep numpy  # 或 pip list | findstr numpy (Windows)

# 4. 查看数据文件
ls data_raw/  # 或 dir data_raw\ (Windows)

# 5. 测试模块导入
python -c "from modules import preprocess; print('OK')"
```

### 详细日志查看

如果 `main_pipeline.py` 运行出错，查看详细错误：

```bash
# 方法1: 直接运行，查看终端输出
python main_pipeline.py

# 方法2: 保存日志到文件
python main_pipeline.py > output.log 2>&1
# 然后查看 output.log 文件

# 方法3: 单步运行各个模块
python -c "from modules.preprocess import DataPreprocessor; p=DataPreprocessor(); p.process_all()"
```

### 性能问题

如果运行很慢：

1. **检查数据量**:
   - 样本太多（>200个）会导致训练慢
   - 考虑使用部分数据或更快的分类器（KNN）

2. **关闭网格搜索**:
   ```python
   # 修改 main_pipeline.py 第137行
   classifier.train(X_train, y_train, use_grid_search=False)
   ```

3. **减少交叉验证折数**:
   - 已自动调整，样本少时会减少折数

---

## 📝 更新日志

### 2026-02-13 v1.1
- ✅ 新增现场预测流程 `predict_pipeline.py`
- ✅ 新增预测目录 `prediction_raw/`、`prediction_preprocessed/`
- ✅ 新增预测结果文件 `prediction_results.csv`
- ✅ README 局部补充“真正开始预测”章节与命令说明

### 2026-02-13 v1.0
- ✅ 完成模块化代码重构
- ✅ 支持多次采集数据（带序号格式）
- ✅ 修复材料标签识别问题（统一去除序号）
- ✅ 完成74个样本的数据采集（5种材料）
- ✅ 实现SVM/RF/KNN三种分类器
- ✅ 生成完整的可视化报告
- ✅ 添加系统检查脚本 `check_system.py`
- ✅ 添加 `.gitignore` 配置
- ✅ 完善README文档

### 已知问题
- POM材料缺少1个样本（只有14个，其他材料都是15个）
- 建议补充采集

### 未来计划（Objective 3）
- [ ] 机械臂ROS2集成（Han's Elfin）
- [ ] 夹爪力控制（DH-Robotics PGI）
- [ ] 实时分类系统（<100ms延迟）
- [ ] 强化学习策略训练
- [ ] Jetson Nano边缘部署

---

## 📧 技术支持

### 遇到问题时的处理顺序

1. **运行系统检查**:
   ```bash
   python check_system.py
   ```

2. **查看本README的"常见问题"部分**

3. **检查终端错误信息**:
   - 仔细阅读完整的错误堆栈
   - Google搜索关键错误信息

4. **查看模块源代码**:
   - 所有模块都有详细注释
   - 理解代码逻辑有助于排查问题

5. **联系项目成员**:
   - LI Yuntong (20944800)
   - Guo Ruopu (20945464)

### 提问时请提供

- ✓ 完整的错误信息（截图或文本）
- ✓ 运行的命令
- ✓ Python版本 (`python --version`)
- ✓ 操作系统（Windows/Mac/Linux）
- ✓ `check_system.py` 的输出

---

## 📚 延伸阅读

- [scikit-learn官方文档](https://scikit-learn.org/) - 机器学习算法
- [pandas文档](https://pandas.pydata.org/) - 数据处理
- [matplotlib教程](https://matplotlib.org/stable/tutorials/index.html) - 数据可视化
- FYP中期报告 - 项目详细说明和实验结果

---

**最后更新**: 2026-02-13  
**版本**: v1.0  
**项目状态**: ✅ Objective 1 & 2 完成，Objective 3 进行中  
**代码行数**: ~1500 lines  
**文档字数**: ~3000 words
