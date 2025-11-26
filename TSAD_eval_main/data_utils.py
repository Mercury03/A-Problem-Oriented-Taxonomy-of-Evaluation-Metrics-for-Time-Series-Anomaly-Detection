"""
实验数据保存和加载工具模块

用于将实验结果保存为可重复使用的格式，实现实验和绘图分离
"""

import pickle
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import numpy as np
import sys
import os

# 确保能导入exp_better中的类（pickle反序列化需要）
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

# 预先导入必要的类以支持pickle
try:
    from exp import DatasetResult, MetricResult, ExperimentConfig
    _CLASSES_IMPORTED = True
except ImportError:
    _CLASSES_IMPORTED = False
    print("警告: 无法从exp导入类定义，pickle加载可能失败")

class NumpyEncoder(json.JSONEncoder):
    """支持numpy类型的JSON编码器"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)


def save_experiment_data(results: Dict[str, Any], 
                         summary_stats: Dict[str, Any],
                         config: Any,
                         output_dir: Path,
                         timestamp: Optional[str] = None,
                         all_runs: Optional[list] = None) -> Dict[str, Path]:
    """
    保存完整的实验数据
    
    Args:
        results: 实验结果字典（已聚合）
        summary_stats: 汇总统计（已聚合）
        config: 实验配置
        output_dir: 输出目录
        timestamp: 时间戳（可选，自动生成）
        all_runs: 所有独立运行的原始数据（可选，用于多次实验）
    
    Returns:
        包含保存文件路径的字典
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    # 1. 保存完整数据为pickle（包含所有numpy数组等）
    pickle_file = output_dir / f"experiment_data_{timestamp}.pkl"
    data_to_save = {
        'results': results,
        'summary_stats': summary_stats,
        'config': config,
        'timestamp': timestamp
    }
    
    # 如果有多次运行的原始数据，也保存
    if all_runs is not None:
        data_to_save['all_runs'] = all_runs
    
    with open(pickle_file, 'wb') as f:
        pickle.dump(data_to_save, f)
    saved_files['pickle'] = pickle_file
    print(f"  ✓ 完整数据已保存: {pickle_file}")
    
    # 2. 保存JSON格式（便于人类阅读和其他工具使用）
    json_file = output_dir / f"experiment_data_{timestamp}.json"
    
    # 序列化配置
    from dataclasses import asdict, is_dataclass
    if is_dataclass(config):
        config_dict = asdict(config)
    else:
        config_dict = config
    
    # 序列化results
    serializable_results = {}
    for dataset_id, dataset_result in results.items():
        if is_dataclass(dataset_result):
            dataset_dict = asdict(dataset_result)
        else:
            dataset_dict = dataset_result
        
        # 确保metric_results也被序列化
        if 'metric_results' in dataset_dict:
            metric_results_serialized = {}
            for metric_name, metric_result in dataset_dict['metric_results'].items():
                if is_dataclass(metric_result):
                    metric_results_serialized[metric_name] = asdict(metric_result)
                else:
                    metric_results_serialized[metric_name] = metric_result
            dataset_dict['metric_results'] = metric_results_serialized
        
        serializable_results[dataset_id] = dataset_dict
    
    json_data = {
        'config': config_dict,
        'results': serializable_results,
        'summary_stats': summary_stats,
        'timestamp': timestamp
    }
    
    # 如果有多次运行的原始数据，也保存（但JSON不保存all_runs以节省空间）
    if all_runs is not None:
        json_data['n_runs'] = len(all_runs)
        json_data['has_multiple_runs'] = True
    
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2, cls=NumpyEncoder)
    saved_files['json'] = json_file
    print(f"  ✓ JSON数据已保存: {json_file}")
    
    # 3. 保存一个简单的元数据文件
    metadata_file = output_dir / f"experiment_metadata_{timestamp}.txt"
    with open(metadata_file, 'w') as f:
        f.write(f"实验时间: {timestamp}\n")
        f.write(f"数据集数量: {len(results)}\n")
        f.write(f"评估指标数量: {len(summary_stats.get('metric_performance', {}))}\n")
        if 'metric_performance' in summary_stats:
            f.write(f"\n指标列表:\n")
            for i, metric_name in enumerate(summary_stats['metric_performance'].keys(), 1):
                f.write(f"  {i}. {metric_name}\n")
    saved_files['metadata'] = metadata_file
    print(f"  ✓ 元数据已保存: {metadata_file}")
    
    return saved_files


def load_experiment_data(data_file: Path) -> Dict[str, Any]:
    """
    加载实验数据
    
    Args:
        data_file: 数据文件路径（.pkl或.json）
    
    Returns:
        包含results, summary_stats, config的字典
    """
    data_file = Path(data_file)
    
    if not data_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_file}")
    
    # 根据文件扩展名选择加载方式
    if data_file.suffix == '.pkl':
        print(f"从pickle文件加载: {data_file}")
        
        if not _CLASSES_IMPORTED:
            print("  警告: 类定义未成功导入，尝试重新导入...")
            try:
                from exp_better import DatasetResult, MetricResult, ExperimentConfig
            except ImportError as e:
                raise ImportError(f"无法导入必要的类定义，pickle加载失败: {e}")
        
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        print(f"  ✓ 数据已加载")
        return data
    
    elif data_file.suffix == '.json':
        print(f"从JSON文件加载: {data_file}")
        with open(data_file, 'r') as f:
            data = json.load(f)
        print(f"  ✓ 数据已加载")
        
        # 重构dataclass对象（如果需要）
        # 这里返回原始字典，绘图函数应该能够处理
        return data
    
    else:
        raise ValueError(f"不支持的文件格式: {data_file.suffix}，请使用.pkl或.json")


def list_saved_experiments(output_dir: Path) -> list:
    """
    列出所有保存的实验数据
    
    Args:
        output_dir: 输出目录
    
    Returns:
        实验数据文件路径列表
    """
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        print(f"目录不存在: {output_dir}")
        return []
    
    # 查找所有pickle文件
    pkl_files = sorted(output_dir.glob("experiment_data_*.pkl"), reverse=True)
    
    if not pkl_files:
        print(f"在 {output_dir} 中未找到实验数据")
        return []
    
    print(f"\n找到 {len(pkl_files)} 个实验数据文件:")
    for i, file in enumerate(pkl_files, 1):
        # 提取时间戳
        timestamp = file.stem.replace("experiment_data_", "")
        file_size = file.stat().st_size / (1024 * 1024)  # MB
        print(f"  {i}. {file.name} ({file_size:.2f} MB) - {timestamp}")
    
    return pkl_files


def get_latest_experiment(output_dir: Path) -> Optional[Path]:
    """
    获取最新的实验数据文件
    
    Args:
        output_dir: 输出目录
    
    Returns:
        最新的实验数据文件路径，如果不存在则返回None
    """
    experiments = list_saved_experiments(output_dir)
    
    if experiments:
        latest_pkl = experiments[0]
        # 优先使用JSON格式（避免pickle的兼容性问题）
        json_file = latest_pkl.with_suffix('.json')
        if json_file.exists():
            latest = json_file
            print(f"\n最新实验（JSON格式）: {latest.name}")
        else:
            latest = latest_pkl
            print(f"\n最新实验（pickle格式）: {latest.name}")
        return latest
    
    return None
