"""
特征提取模块
基于报告中的6个物理特征：
1. Effective Stiffness (k_eff) - 有效刚度
2. Peak Normal Force (Fz_peak) - 峰值法向力
3. Mean Friction Level (mu_mean) - 平均摩擦系数
4. Friction Stability (mu_std) - 摩擦稳定性
5. Sliding Instability Strength (slip) - 滑动不稳定强度
6. Micro-vibration (micro) - 微振动
"""

import pandas as pd
import numpy as np
import os
import sys

# 添加父目录到路径以导入Config
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import Config


class FeatureExtractor:
    """特征提取器"""
    
    def __init__(self, window_size=100):
        """
        初始化特征提取器
        
        Args:
            window_size: 特征计算窗口大小（样本数）
        """
        self.window_size = window_size
        
    def extract_features_from_window(self, df_window):
        """
        从一个数据窗口提取特征
        
        Args:
            df_window: 数据窗口DataFrame
            
        Returns:
            特征字典
        """
        features = {}
        
        # 1. Effective Stiffness (k_eff)
        if 'dFz_dt' in df_window.columns:
            features['k_eff'] = np.mean(np.abs(df_window['dFz_dt']))
        else:
            # 如果没有dFz_dt，手动计算
            fz_values = df_window['Fz'].values
            if len(fz_values) > 1:
                dfz = np.diff(fz_values)
                features['k_eff'] = np.mean(np.abs(dfz))
            else:
                features['k_eff'] = 0.0
        
        # 2. Peak Normal Force (Fz_peak)
        features['Fz_peak'] = np.max(df_window['Fz'])
        
        # 3. Mean Friction Level (mu_mean)
        features['mu_mean'] = np.mean(df_window['mu'])
        
        # 4. Friction Stability (mu_std)
        features['mu_std'] = np.std(df_window['mu'])
        
        # 5. Sliding Instability Strength (slip)
        if 'dFz_dt' in df_window.columns:
            features['slip'] = np.max(np.abs(df_window['dFz_dt']))
        else:
            fz_values = df_window['Fz'].values
            if len(fz_values) > 1:
                dfz = np.diff(fz_values)
                features['slip'] = np.max(np.abs(dfz)) if len(dfz) > 0 else 0.0
            else:
                features['slip'] = 0.0
        
        # 6. Micro-vibration (micro)
        ft_values = df_window['Ft'].values
        ft_mean = np.mean(ft_values)
        features['micro'] = np.sqrt(np.mean((ft_values - ft_mean)**2))
        
        # 额外统计特征
        features['Fz_mean'] = np.mean(df_window['Fz'])
        features['Fz_std'] = np.std(df_window['Fz'])
        features['Ft_mean'] = np.mean(df_window['Ft'])
        features['Ft_std'] = np.std(df_window['Ft'])
        
        return features
    
    def extract_features_sliding_window(self, df, overlap=0.5):
        """
        使用滑动窗口提取特征
        
        Args:
            df: 预处理后的DataFrame
            overlap: 窗口重叠率
            
        Returns:
            特征DataFrame
        """
        step = int(self.window_size * (1 - overlap))
        features_list = []
        
        # 统一材料标签（去掉序号）
        material_label = None
        if 'material' in df.columns:
            import re
            material_label = df.iloc[0]['material']
            material_label = re.sub(r'_\d+$', '', material_label)
        
        for start in range(0, len(df) - self.window_size + 1, step):
            end = start + self.window_size
            window = df.iloc[start:end]
            
            features = self.extract_features_from_window(window)
            
            # 添加材料标签
            if material_label:
                features['material'] = material_label
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def extract_features_global(self, df):
        """
        从整个接触序列提取全局特征（单样本）
        
        Args:
            df: 预处理后的DataFrame
            
        Returns:
            特征字典
        """
        features = self.extract_features_from_window(df)
        
        # 添加材料标签（统一格式，去掉序号）
        if 'material' in df.columns:
            material_label = df.iloc[0]['material']
            # 去掉序号后缀：Material_Wood_raw_1 -> Material_Wood_raw
            import re
            material_label = re.sub(r'_\d+$', '', material_label)
            features['material'] = material_label
        
        return features
    
    def process_file(self, filename, method='global'):
        """
        处理单个预处理文件并提取特征
        
        Args:
            filename: 预处理后的文件名
            method: 'global' 或 'sliding' - 提取方法
            
        Returns:
            特征DataFrame
        """
        filepath = os.path.join(Config.PREPROCESS_DIR, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"文件不存在: {filepath}")
        
        print(f"正在提取特征: {filename}")
        df = pd.read_csv(filepath)
        
        if method == 'global':
            features = self.extract_features_global(df)
            features_df = pd.DataFrame([features])
        elif method == 'sliding':
            features_df = self.extract_features_sliding_window(df)
        else:
            raise ValueError(f"未知方法: {method}")
        
        print(f"  提取了 {len(features_df)} 个特征样本")
        return features_df
    
    def process_all(self, method='global', save=True):
        """
        处理所有预处理文件
        
        Args:
            method: 'global' 或 'sliding'
            save: 是否保存特征文件
            
        Returns:
            合并的特征DataFrame
        """
        all_features = []
        
        for filename in os.listdir(Config.PREPROCESS_DIR):
            # 匹配所有预处理文件（包括带序号的）
            if not filename.endswith(".csv") or "preprocessed" not in filename:
                continue
            
            try:
                features_df = self.process_file(filename, method=method)
                all_features.append(features_df)
            except Exception as e:
                print(f"处理 {filename} 时出错: {str(e)}")
        
        # 合并所有特征
        if all_features:
            combined_df = pd.concat(all_features, ignore_index=True)
            print(f"\n总共提取 {len(combined_df)} 个样本")
            
            # 保存
            if save:
                features_dir = os.path.join(Config.BASE_DIR, "data_features")
                os.makedirs(features_dir, exist_ok=True)
                
                output_file = os.path.join(features_dir, f"features_{method}.csv")
                combined_df.to_csv(output_file, index=False)
                print(f"特征已保存到: {output_file}")
            
            return combined_df
        else:
            print("没有提取到任何特征")
            return None


def main():
    """测试函数"""
    extractor = FeatureExtractor(window_size=100)
    
    # 使用全局方法（每个文件一个样本）
    features_df = extractor.process_all(method='global', save=True)
    
    if features_df is not None:
        print("\n特征统计:")
        print(features_df.describe())
        
        print("\n材料分布:")
        if 'material' in features_df.columns:
            print(features_df['material'].value_counts())


if __name__ == "__main__":
    main()
