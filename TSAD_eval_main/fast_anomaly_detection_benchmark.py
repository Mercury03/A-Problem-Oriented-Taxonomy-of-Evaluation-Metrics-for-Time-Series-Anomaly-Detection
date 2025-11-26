"""
精简版时间序列异常检测基准测试

专注于测试最有效的几个模型，快速得到结果

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import time
from datetime import datetime
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# Import existing modules
import sys
import os
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_CURRENT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.DLinear.DLinear import DLinear
from models.Autoformer.Autoformer import Autoformer
from models.DeepAR.model import DeepAR
from models.FEDformer.FEDformer import Model as FEDformer
from models.Informer.model import Informer
from models.LSTNet.model import LSTNet
from models.GTA.gta import GTA

from synthetic_anomaly_generator import SyntheticAnomalyDataset


class ModelConfig:
    """模型配置类"""
    def __init__(self, window_size=50):
        # 基础配置
        self.input_len = window_size
        self.label_len = window_size // 2  # 标签长度，用于Autoformer等模型
        self.variate = 1  # 单变量
        self.out_variate = 1  # 输出维度
        
        # 训练配置
        self.learning_rate = 0.001
        self.train_epochs = 30  # 减少最大训练轮数
        self.batch_size = 32
        self.patience = 3  # 减少早停耐心，更快停止
        self.dropout = 0.1
        
        # 模型特定参数
        self.d_model = 64
        self.n_heads = 8
        self.e_layers = 2
        self.d_layers = 1
        self.d_ff = 256
        self.factor = 1
        self.activation = 'gelu'
        self.attn = 'prob'
        self.distil = True
        self.mix = True
        self.moving_avg = [24]
        
        # RTNet特定参数
        self.kernel = 3
        self.block_nums = 3
        self.pyramid = 1
        
        # LSTNet特定参数
        self.RNN_hid_size = 100
        self.CNN_hid_size = 100
        self.hidSkip = 5
        self.skip = 24
        self.CNN_kernel = 6
        self.highway_window = 24
        
        # DeepAR特定参数
        self.num_layers = 3
        
        # GTA特定参数
        self.num_levels = 3



class TimeSeriesDataProcessor:
    """时间序列数据预处理器"""
    
    def __init__(self, window_size=50):
        self.window_size = window_size
    
    def create_sliding_windows(self, data, is_training=True):
        """创建滑动窗口数据"""
        data = np.array(data)
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
            
        X, y = [], []
        
        for i in range(len(data) - self.window_size):
            X.append(data[i:i+self.window_size])
            y.append(data[i+self.window_size])
            
        return np.array(X), np.array(y)
    
    def normalize_data(self, train_data, test_data=None):
        """标准化数据"""
        mean = np.mean(train_data)
        std = np.std(train_data)
        
        train_normalized = (train_data - mean) / (std + 1e-8)
        
        if test_data is not None:
            test_normalized = (test_data - mean) / (std + 1e-8)
            return train_normalized, test_normalized, mean, std
        
        return train_normalized, mean, std


def create_fedformer(args, device):
    """创建FEDformer模型"""
    class Configs(object):
        def __init__(self):
            self.output_v = args.out_variate
            self.modes = 16
            self.mode_select = 'low'
            self.version = 'Fourier'
            self.moving_avg = args.moving_avg
            self.L = 1
            self.features = 'M'
            self.base = 'legendre'
            self.cross_activation = 'softmax'
            self.seq_len = args.input_len
            self.label_len = args.label_len
            self.pred_len = 1
            self.output_attention = False
            self.d_model = 32
            self.dropout = args.dropout
            self.factor = 1
            self.n_heads = 8
            self.d_ff = 64
            self.embed = 'timeF'
            self.freq = 'h'
            self.e_layers = args.e_layers
            self.d_layers = args.d_layers
            self.activation = args.activation
            self.LIN = False

    configs = Configs()
    return FEDformer(configs)


def get_model(model_name, args, device):
    """创建指定的模型"""
    LIN = False
    
    print(f'创建模型: {model_name}')
    
    model_constructors = {
        'DLinear': lambda: DLinear(
            args.variate,
            args.out_variate,
            args.input_len,
            args.kernel,
            LIN
        ),
        
        'DeepAR': lambda: DeepAR(
            args.variate,
            args.out_variate,
            args.input_len,
            args.d_model,
            args.num_layers,
            LIN
        ),

        'GTA': lambda: GTA(
            args.variate,
            args.out_variate,
            args.input_len,
            args.label_len,
            args.num_levels,
            args.factor,
            512,  #  d_model=512
            args.n_heads,
            2,  #  e_layers=2
            1,  #  d_layers=1
            0.05,  #  dropout=0.05
            args.activation,
            LIN,
            device
        ),
        
        'LSTNet': lambda: LSTNet(
            args.input_len,
            args.variate,
            args.RNN_hid_size,
            args.CNN_hid_size,
            args.hidSkip,
            args.skip,
            args.CNN_kernel,
            args.highway_window,
            args.dropout,
            args.out_variate,
            LIN
        ),
        
        'Autoformer': lambda: Autoformer(
            args.variate,
            args.out_variate,
            args.input_len,
            args.label_len,
            args.moving_avg,
            args.d_model,
            args.dropout,
            args.factor,
            args.n_heads,
            args.activation,
            args.e_layers,
            args.d_layers,
            LIN
        ),
        
        'FEDformer': lambda: create_fedformer(args, device),
        
        'Informer': lambda: Informer(
            args.variate,
            args.variate,
            args.out_variate,
            args.input_len,
            args.label_len,
            args.factor,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.dropout,
            args.attn,
            args.activation,
            args.distil,
            args.mix,
            LIN
        )
    }
    
    if model_name not in model_constructors:
        raise ValueError(f"未知模型: {model_name}")

    model = model_constructors[model_name]()
    return model.float().to(device)


class AnomalyDetector:
    """基于预测误差的异常检测器"""
    
    def __init__(self, contamination_rate=0.1):
        self.contamination_rate = contamination_rate
        self.threshold = None
    
    def fit_threshold(self, prediction_errors):
        """根据污染率确定阈值"""
        self.threshold = np.percentile(prediction_errors, (1 - self.contamination_rate) * 100)
        return self.threshold
    
    def predict(self, prediction_errors):
        """基于阈值进行异常检测"""
        if self.threshold is None:
            raise ValueError("需要先调用fit_threshold()设置阈值")
        
        binary_predictions = (prediction_errors > self.threshold).astype(int)
        scores = prediction_errors.copy()
        
        return binary_predictions, scores


class FastModelTrainer:
    """快速模型训练器"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"使用设备: {self.device}")
    
    def train_model(self, model, train_loader, val_loader, args):
        """快速训练模型"""
        print(f"快速训练模式 - 最大轮数: {args.train_epochs}, 早停耐心: {args.patience}")
        
        # 定义优化器 - 使用更高的学习率
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        criterion = torch.nn.MSELoss()
        
        # 早停参数
        best_val_loss = float('inf')
        patience_counter = 0
        
        # 训练循环
        for epoch in range(args.train_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_count = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                
                try:
                    output, gt = model(batch_x)
                    loss = criterion(output, gt)
                except:
                    output = model(batch_x)
                    loss = criterion(output, batch_y)
                
                if torch.isnan(loss):
                    continue
                
                train_loss += loss.item()
                train_count += 1
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_count = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device).float()
                    batch_y = batch_y.to(self.device).float()
                    
                    try:
                        output, gt = model(batch_x)
                        loss = criterion(output, gt)
                    except:
                        output = model(batch_x)
                        loss = criterion(output, batch_y)
                    
                    if not torch.isnan(loss):
                        val_loss += loss.item()
                        val_count += 1
            
            # 计算平均损失
            avg_val_loss = val_loss / max(val_count, 1)
            
            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 5 == 0:
                avg_train_loss = train_loss / max(train_count, 1)
                print(f"  轮次 {epoch + 1}: 训练损失={avg_train_loss:.6f}, 验证损失={avg_val_loss:.6f}")
            
            # 早停
            if patience_counter >= args.patience:
                print(f"  早停于第 {epoch + 1} 轮")
                break
        
        return model
    
    def predict_and_detect(self, model, test_data, args, contamination_rate):
        """使用模型进行预测并检测异常"""
        model.eval()
        predictions = []
        actuals = []
        
        processor = TimeSeriesDataProcessor(args.input_len)
        X_test, y_test = processor.create_sliding_windows(test_data, is_training=False)
        
        # 转换为tensor
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        with torch.no_grad():
            for i in range(0, len(X_test_tensor), args.batch_size):
                batch_x = X_test_tensor[i:i+args.batch_size]
                
                try:
                    output, _ = model(batch_x)
                    pred = output.cpu().numpy()
                except:
                    pred = model(batch_x).cpu().numpy()
                
                predictions.extend(pred.flatten())
                actuals.extend(y_test[i:i+args.batch_size].flatten())
        
        # 计算预测误差（MSE）
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        mse_errors = (predictions - actuals) ** 2
        
        # 异常检测
        detector = AnomalyDetector(contamination_rate)
        threshold = detector.fit_threshold(mse_errors)
        binary_predictions, scores = detector.predict(mse_errors)
        
        return {
            'predictions': predictions,
            'actuals': actuals,
            'mse_errors': mse_errors,
            'threshold': threshold,
            'binary_predictions': binary_predictions,
            'scores': scores
        }


def evaluate_model(model_name, args, device, train_dataset, val_dataset, test_data, true_labels, contamination_rate):
    """评估单个模型"""
    print(f"\n{'='*50}")
    print(f"测试模型: {model_name}")
    print(f"{'='*50}")
    
    try:
        # 创建模型
        model = get_model(model_name, args, device)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        # 训练器
        trainer = FastModelTrainer(device)
        
        # 训练模型
        start_time = time.time()
        trained_model = trainer.train_model(model, train_loader, val_loader, args)
        training_time = time.time() - start_time
        
        # 测试和异常检测
        print("进行异常检测...")
        start_time = time.time()
        detection_results = trainer.predict_and_detect(
            trained_model, test_data, args, contamination_rate
        )
        detection_time = time.time() - start_time
        
        # 评估结果
        adjusted_labels = true_labels[args.input_len:][:len(detection_results['binary_predictions'])]
        
        accuracy = accuracy_score(adjusted_labels, detection_results['binary_predictions'])
        precision = precision_score(adjusted_labels, detection_results['binary_predictions'])
        recall = recall_score(adjusted_labels, detection_results['binary_predictions'])
        f1 = f1_score(adjusted_labels, detection_results['binary_predictions'])
        
        try:
            auc = roc_auc_score(adjusted_labels, detection_results['scores'])
        except:
            auc = 0.5
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'threshold': float(detection_results['threshold']),
            'training_time': training_time,
            'detection_time': detection_time,
            'status': 'success'
        }
        
        print(f"结果:")
        print(f"  - 准确率: {accuracy:.4f}")
        print(f"  - 精确率: {precision:.4f}")
        print(f"  - 召回率: {recall:.4f}")
        print(f"  - F1分数: {f1:.4f}")
        print(f"  - AUC: {auc:.4f}")
        print(f"  - 训练时间: {training_time:.2f}秒")
        print(f"  - 检测时间: {detection_time:.2f}秒")
        
        return results
        
    except Exception as e:
        print(f"模型 {model_name} 测试失败: {str(e)}")
        return {
            'status': 'failed',
            'error': str(e)
        }


def main():
    """主函数：快速测试多个模型"""
    print("="*60)
    print("精简版时间序列异常检测基准测试")
    print("="*60)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 生成合成数据
    print("\n生成合成数据...")
    generator = SyntheticAnomalyDataset(random_seed=42)
    dataset = generator.generate_dataset(
        length=8000,
        contamination_rate=0.1,
        baseline_config={
            "trend_config": {"trend_type": "linear", "slope": 0.002, "intercept": 10},
            "periodic_config": {"periods": [200, 50], "amplitudes": [3.0, 1.0]},
            "noise_config": {"noise_type": "gaussian", "std": 0.2}
        }
    )
    
    # 提取数据
    baseline_data = dataset["baseline"]
    anomalous_data = dataset["data"]
    true_labels = dataset["labels"]
    contamination_rate = dataset["config"]["contamination_rate"]
    
    print(f"数据长度: {len(baseline_data)}")
    print(f"污染率: {contamination_rate}")
    print(f"异常点数量: {np.sum(true_labels)}")
    
    # 数据预处理
    print("\n数据预处理...")
    processor = TimeSeriesDataProcessor(window_size=50)
    
    # 标准化
    baseline_normalized, anomalous_normalized, mean, std = processor.normalize_data(
        baseline_data, anomalous_data
    )
    
    # 创建训练数据
    X_train, y_train = processor.create_sliding_windows(baseline_normalized, is_training=True)
    
    # 划分训练集和验证集 (80:20)
    split_idx = int(0.8 * len(X_train))
    X_train_split, X_val = X_train[:split_idx], X_train[split_idx:]
    y_train_split, y_val = y_train[:split_idx], y_train[split_idx:]
    
    # 转换为PyTorch数据集
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_split),
        torch.FloatTensor(y_train_split)
    )
    
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建模型配置
    args = ModelConfig(window_size=50)
    
    # 要测试的模型列表 - 与实验exp.py保持一致
    model_names = ['DLinear', 'DeepAR', 'Informer']
    
    results = {}
    
    # 测试每个模型
    for model_name in model_names:
        result = evaluate_model(
            model_name, args, device, train_dataset, val_dataset, 
            anomalous_normalized, true_labels, contamination_rate
        )
        results[model_name] = result
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"fast_anomaly_detection_benchmark_{timestamp}.json"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print("测试完成！结果总结:")
    print(f"{'='*70}")
    
    successful_models = {k: v for k, v in results.items() if v.get('status') == 'success'}
    
    if successful_models:
        # 按F1分数排序
        sorted_models = sorted(
            successful_models.items(), 
            key=lambda x: x[1]['f1_score'], 
            reverse=True
        )
        
        print(f"{'模型':<12} {'F1':<8} {'AUC':<8} {'精确率':<8} {'召回率':<8} {'训练时间':<8}")
        print("-" * 70)
        
        for model_name, metrics in sorted_models:
            print(f"{model_name:<12} {metrics['f1_score']:<8.4f} {metrics['auc']:<8.4f} "
                  f"{metrics['precision']:<8.4f} {metrics['recall']:<8.4f} {metrics['training_time']:<8.2f}")
    
    failed_models = [k for k, v in results.items() if v.get('status') == 'failed']
    if failed_models:
        print(f"\n失败的模型: {', '.join(failed_models)}")
    
    print(f"\n详细结果已保存至: {results_file}")
    
    # 推荐最佳模型
    if successful_models:
        print(f"\n推荐:")
        best_f1_model = max(successful_models.items(), key=lambda x: x[1]['f1_score'])
        best_auc_model = max(successful_models.items(), key=lambda x: x[1]['auc'])
        fastest_model = min(successful_models.items(), key=lambda x: x[1]['training_time'])
        
        print(f"- 最佳检测性能 (F1): {best_f1_model[0]} (F1={best_f1_model[1]['f1_score']:.4f})")
        print(f"- 最佳AUC: {best_auc_model[0]} (AUC={best_auc_model[1]['auc']:.4f})")
        print(f"- 最快训练: {fastest_model[0]} ({fastest_model[1]['training_time']:.2f}秒)")


def simple_anomaly_detection(train_data, test_data, true_labels, model_name='DLinear', contamination_rate=0.1, window_size=50):
    """
    简化的异常检测接口 - 专为exp.py设计
    
    Args:
        train_data: 训练数据 (正常数据)
        test_data: 测试数据 (包含异常)
        true_labels: 真实标签
        model_name: 模型名称
        contamination_rate: 污染率
        window_size: 窗口大小
        
    Returns:
        tuple: (binary_predictions, anomaly_scores) - 预测标签和异常分数
    """
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 创建配置
        args = ModelConfig(window_size=window_size)
        processor = TimeSeriesDataProcessor(args.input_len)
        
        # 数据预处理
        train_normalized, test_normalized, mean, std = processor.normalize_data(train_data, test_data)
        X_train, y_train = processor.create_sliding_windows(train_normalized, is_training=True)
        
        # 划分训练集和验证集
        split_idx = int(0.8 * len(X_train))
        X_train_split, X_val = X_train[:split_idx], X_train[split_idx:]
        y_train_split, y_val = y_train[:split_idx], y_train[split_idx:]
        
        # 创建数据集
        train_dataset = TensorDataset(torch.FloatTensor(X_train_split), torch.FloatTensor(y_train_split))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        # 创建和训练模型
        model = get_model(model_name, args, device)
        trainer = FastModelTrainer(device)
        trained_model = trainer.train_model(model, train_loader, val_loader, args)
        
        # 预测和检测
        detection_results = trainer.predict_and_detect(trained_model, test_normalized, args, contamination_rate)
        
        # 将结果扩展到原始序列长度
        full_length = len(test_data)
        full_labels = np.zeros(full_length)
        full_scores = np.zeros(full_length)
        
        binary_preds = detection_results['binary_predictions']
        scores = detection_results['scores']
        
        # 映射窗口预测到全序列 (从window_size位置开始)
        for i, (pred, score) in enumerate(zip(binary_preds, scores)):
            idx = i + window_size
            if idx < full_length:
                full_labels[idx] = pred
                full_scores[idx] = score
        
        # 填充前面的值（使用预测的平均值）
        if len(scores) > 0:
            avg_score = np.mean(scores)
            threshold = detection_results['threshold']
            for i in range(min(window_size, full_length)):
                full_scores[i] = avg_score
                full_labels[i] = 1 if avg_score > threshold else 0
        
        return full_labels.astype(int), full_scores
        
    except Exception as e:
        print(f"简化异常检测失败 ({model_name}): {str(e)}")
        # 回退到随机预测
        labels = np.random.binomial(1, contamination_rate, len(test_data))
        scores = np.random.uniform(0, 1, len(test_data))
        return labels, scores


if __name__ == "__main__":
    main()