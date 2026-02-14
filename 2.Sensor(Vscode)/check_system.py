"""
系统环境检查脚本
运行此脚本检查项目环境是否配置正确
"""

import sys
import os

def print_header(title):
    """打印标题"""
    print("\n" + "=" * 70)
    print(f" {title} ".center(70, "="))
    print("=" * 70 + "\n")

def check_python_version():
    """检查Python版本"""
    print("✓ 检查Python版本...")
    version = sys.version_info
    print(f"  Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("  ⚠️  警告: 建议使用Python 3.7+")
        return False
    else:
        print("  ✓ 版本符合要求")
        return True

def check_dependencies():
    """检查依赖包"""
    print("\n✓ 检查依赖包...")
    
    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scipy': 'scipy',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'serial': 'pyserial',
        'keyboard': 'keyboard'
    }
    
    missing = []
    installed = []
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            installed.append(package_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            missing.append(package_name)
            print(f"  ✗ {package_name} - 未安装")
    
    if missing:
        print(f"\n  ⚠️  缺少 {len(missing)} 个依赖包")
        print(f"  请运行: pip install {' '.join(missing)}")
        return False
    else:
        print(f"\n  ✓ 所有依赖包已安装 ({len(installed)}/{len(required_packages)})")
        return True

def check_directory_structure():
    """检查目录结构"""
    print("\n✓ 检查目录结构...")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    required_dirs = [
        'modules',
        'data_raw'
    ]
    
    created_dirs = [
        'data_preprocess',
        'data_features',
        'models',
        'visualizations'
    ]
    
    all_good = True
    
    # 检查必需目录
    for dir_name in required_dirs:
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.exists(dir_path):
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ - 不存在")
            all_good = False
    
    # 检查自动创建的目录（不存在是正常的）
    for dir_name in created_dirs:
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.exists(dir_path):
            print(f"  ✓ {dir_name}/ (已创建)")
        else:
            print(f"  ○ {dir_name}/ (运行时自动创建)")
    
    return all_good

def check_core_files():
    """检查核心文件"""
    print("\n✓ 检查核心文件...")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    required_files = [
        'Config.py',
        'main_pipeline.py',
        'force_control_collect.py',
        'requirements.txt',
        'README.md',
        'modules/__init__.py',
        'modules/preprocess.py',
        'modules/feature_extraction.py',
        'modules/train_classifier.py',
        'modules/visualize.py',
        'modules/predict.py'
    ]
    
    missing = []
    
    for file_name in required_files:
        file_path = os.path.join(base_dir, file_name)
        if os.path.exists(file_path):
            print(f"  ✓ {file_name}")
        else:
            print(f"  ✗ {file_name} - 不存在")
            missing.append(file_name)
    
    if missing:
        print(f"\n  ⚠️  缺少 {len(missing)} 个核心文件")
        return False
    else:
        print(f"\n  ✓ 所有核心文件完整")
        return True

def check_data_files():
    """检查数据文件"""
    print("\n✓ 检查数据文件...")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, 'data_raw')
    
    if not os.path.exists(raw_dir):
        print("  ⚠️  data_raw/ 目录不存在")
        return False
    
    csv_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("  ⚠️  data_raw/ 目录为空")
        print("  提示: 需要先采集数据")
        return False
    
    # 统计材料类型
    materials = {}
    for filename in csv_files:
        if 'Material_' in filename:
            # 提取材料名（去除序号）
            import re
            match = re.search(r'Material_([A-Za-z]+)', filename)
            if match:
                material = match.group(1)
                materials[material] = materials.get(material, 0) + 1
    
    print(f"  ✓ 找到 {len(csv_files)} 个数据文件")
    print(f"\n  材料分布:")
    for material, count in sorted(materials.items()):
        print(f"    - {material}: {count} 个样本")
    
    total = sum(materials.values())
    if total < 10:
        print(f"\n  ⚠️  样本总数较少 ({total}个)")
        print(f"  建议: 每种材料至少采集10次")
        return False
    else:
        print(f"\n  ✓ 样本数量充足 (总计 {total} 个)")
        return True

def check_module_imports():
    """检查模块导入"""
    print("\n✓ 检查模块导入...")
    
    modules_to_test = [
        ('modules.preprocess', 'DataPreprocessor'),
        ('modules.feature_extraction', 'FeatureExtractor'),
        ('modules.train_classifier', 'MaterialClassifier'),
        ('modules.visualize', 'DataVisualizer'),
        ('modules.predict', 'MaterialPredictor')
    ]
    
    failed = []
    
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"  ✓ {module_name}.{class_name}")
        except Exception as e:
            print(f"  ✗ {module_name}.{class_name} - {str(e)}")
            failed.append(module_name)
    
    if failed:
        print(f"\n  ⚠️  {len(failed)} 个模块导入失败")
        return False
    else:
        print(f"\n  ✓ 所有模块导入成功")
        return True

def main():
    """主函数"""
    print_header("材料分类系统 - 环境检查")
    
    results = {
        'Python版本': check_python_version(),
        '依赖包': check_dependencies(),
        '目录结构': check_directory_structure(),
        '核心文件': check_core_files(),
        '数据文件': check_data_files(),
        '模块导入': check_module_imports()
    }
    
    # 总结
    print_header("检查结果总结")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for check_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {check_name:15s}: {status}")
    
    print(f"\n  总计: {passed}/{total} 项检查通过")
    
    if passed == total:
        print("\n✓ 所有检查通过！系统可以正常使用。")
        print("\n下一步:")
        print("  运行: python main_pipeline.py")
    else:
        print("\n✗ 部分检查未通过，请根据上述提示修复问题。")
        print("\n常见解决方案:")
        print("  1. 安装依赖: pip install -r requirements.txt")
        print("  2. 采集数据: python force_control_collect.py")
        print("  3. 检查文件: 确保所有核心文件存在")
    
    print("\n" + "=" * 70)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
