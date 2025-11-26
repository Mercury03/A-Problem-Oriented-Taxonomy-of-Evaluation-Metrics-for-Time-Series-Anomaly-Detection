"""
Time Series Anomaly Detection Synthetic Dataset Generator

A synthetic dataset generator designed based on documentation, supporting:
1. Baseline signal generation (trend, multiple cycles, noise)
2. Insertion of five types of anomalies (point anomalies, amplitude drift, change segments, cycle disruption, contextual anomalies)
3. Strict point-level and segment-level annotation
4. Multivariate support
5. Reproducible random seed control

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json
import copy

# 使用新的numpy随机数生成器
rng = np.random.default_rng(42)

class AnomalyType(Enum):
    """异常类型枚举"""
    POINT_SPIKE = "point_spike"        # 孤立点突增/负突变
    LEVEL_SHIFT = "level_shift"        # 幅度/均值漂移
    COLLECTIVE = "collective"          # 突变段/集体异常
    PERIODIC_DISRUPTION = "periodic_disruption"  # 周期破坏
    CONTEXTUAL = "contextual"          # 上下文异常

@dataclass
class AnomalyEvent:
    """异常事件元数据"""
    type: AnomalyType
    start: int
    end: int
    strength: float
    parameters: Dict
    seed: int

class BaselineGenerator:
    """基线信号生成器
    
    生成包含趋势项、多周期成分和噪声的基线时序信号
    """
    
    def __init__(self, length: int, sampling_rate: float = 1.0, random_seed: int = None):
        """
        初始化基线生成器
        
        Args:
            length: 序列长度
            sampling_rate: 采样率（时间单位间隔）
            random_seed: 随机种子
        """
        self.length = length
        self.sampling_rate = sampling_rate
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 时间轴
        self.time = np.arange(length) * sampling_rate
        
    def generate_trend(self, trend_type: str = "linear", **kwargs) -> np.ndarray:
        """
        生成趋势项
        
        Args:
            trend_type: 趋势类型 ("linear", "piecewise_constant", "none")
            **kwargs: 趋势参数
                - linear: slope, intercept
                - piecewise_constant: levels, change_points
        
        Returns:
            趋势信号数组
        """
        if trend_type == "linear":
            slope = kwargs.get("slope", 0.01)
            intercept = kwargs.get("intercept", 0.0)
            return slope * self.time + intercept
            
        elif trend_type == "piecewise_constant":
            levels = kwargs.get("levels", [0, 1, -0.5])
            change_points = kwargs.get("change_points", [self.length//3, 2*self.length//3])
            
            trend = np.zeros(self.length)
            last_point = 0
            
            for i, change_point in enumerate(change_points + [self.length]):
                if i < len(levels):
                    trend[last_point:change_point] = levels[i]
                last_point = change_point
                
            return trend
            
        elif trend_type == "none":
            return np.zeros(self.length)
            
        else:
            raise ValueError(f"Unknown trend type: {trend_type}")
    
    def generate_periodic_components(self, periods: List[float], 
                                   amplitudes: List[float], 
                                   phases: Optional[List[float]] = None) -> np.ndarray:
        """
        生成多周期成分
        
        Args:
            periods: 周期长度列表
            amplitudes: 对应振幅列表
            phases: 对应相位列表（如果None则随机生成）
        
        Returns:
            周期成分的叠加信号
        """
        if len(periods) != len(amplitudes):
            raise ValueError("periods和amplitudes长度必须相同")
            
        if phases is None:
            phases = [rng.uniform(0, 2*np.pi) for _ in periods]
        elif len(phases) != len(periods):
            raise ValueError("phases和periods长度必须相同")
        
        periodic_signal = np.zeros(self.length)
        
        for period, amplitude, phase in zip(periods, amplitudes, phases):
            # 使用离散时间索引而不是连续时间
            component = amplitude * np.sin(2 * np.pi * np.arange(self.length) / period + phase)
            periodic_signal += component
            
        return periodic_signal
    
    def generate_noise(self, noise_type: str = "gaussian", **kwargs) -> np.ndarray:
        """
        生成噪声项
        
        Args:
            noise_type: 噪声类型 ("gaussian", "ar1")
            **kwargs: 噪声参数
                - gaussian: std
                - ar1: std, rho (自回归系数)
        
        Returns:
            噪声信号数组
        """
        if noise_type == "gaussian":
            std = kwargs.get("std", 0.1)
            return rng.normal(0, std, self.length)
            
        elif noise_type == "ar1":
            std = kwargs.get("std", 0.1)
            rho = kwargs.get("rho", 0.5)
            
            # AR(1)过程: η_t = ρ * η_{t-1} + ε_t
            noise = np.zeros(self.length)
            epsilon = rng.normal(0, std, self.length)
            
            for t in range(1, self.length):
                noise[t] = rho * noise[t-1] + epsilon[t]
                
            return noise
            
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
    
    def generate_baseline(self, 
                         trend_config: Dict = None,
                         periodic_config: Dict = None, 
                         noise_config: Dict = None) -> np.ndarray:
        """
        生成完整的基线信号
        
        Args:
            trend_config: 趋势配置字典
            periodic_config: 周期配置字典
            noise_config: 噪声配置字典
        
        Returns:
            基线信号数组
        """
        # 默认配置
        if trend_config is None:
            trend_config = {"trend_type": "linear", "slope": 0.001, "intercept": 0}
            
        if periodic_config is None:
            periodic_config = {
                "periods": [100, 24],  # 主周期和次周期
                "amplitudes": [2.0, 0.5],
                "phases": None
            }
            
        if noise_config is None:
            noise_config = {"noise_type": "gaussian", "std": 0.1}
        
        # 生成各组件
        trend = self.generate_trend(**trend_config)
        periodic = self.generate_periodic_components(**periodic_config)
        noise = self.generate_noise(**noise_config)
        
        # 合成基线信号
        baseline = trend + periodic + noise
        
        return baseline

class AnomalyInjector:
    """异常注入器
    
    实现五种异常类型的注入算法
    """
    
    def __init__(self, random_seed: int = None, baseline_std: float = None):
        """
        初始化异常注入器
        
        Args:
            random_seed: 随机种子
            baseline_std: 原始基线信号的标准差（用于所有异常强度计算）
        """
        self.random_seed = random_seed
        self.baseline_std = baseline_std
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def inject_point_spike(self, signal: np.ndarray, position: int, 
                          strength: float, window: int = 1, 
                          mode: str = "additive", baseline_std: float = None) -> Tuple[np.ndarray, List[int]]:
        """
        注入孤立点突增/负突变
        
        Args:
            signal: 原始信号
            position: 异常位置
            strength: 异常强度（以信号标准差为单位）
            window: 异常窗口大小（1-3点）
            mode: 注入模式 ("additive"加性 或 "multiplicative"乘性)
            baseline_std: 基线标准差（如果提供则使用，否则使用实例的baseline_std或当前信号std）
        
        Returns:
            (修改后的信号, 异常点索引列表)
        """
        signal_copy = signal.copy()
        # 使用固定的基线标准差，避免异常叠加导致强度放大
        if baseline_std is not None:
            signal_std = baseline_std
        elif self.baseline_std is not None:
            signal_std = self.baseline_std
        else:
            signal_std = np.std(signal)
        
        affected_indices = []
        
        # 确保窗口不超出边界
        start_pos = max(0, position)
        end_pos = min(len(signal), position + window)
        
        for i in range(start_pos, end_pos):
            if mode == "additive":
                signal_copy[i] += strength * signal_std
            elif mode == "multiplicative":
                signal_copy[i] *= max(strength - 1.2, 1.5)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            affected_indices.append(i)
        
        return signal_copy, affected_indices
    
    def inject_level_shift(self, signal: np.ndarray, start: int, end: int,
                          shift_type: str = "mean", strength: float = 1.0, 
                          baseline_std: float = None) -> Tuple[np.ndarray, List[int]]:
        """
        注入幅度/均值漂移
        
        Args:
            signal: 原始信号
            start: 漂移开始位置
            end: 漂移结束位置
            shift_type: 漂移类型 ("mean"均值漂移 或 "amplitude"振幅漂移)
            strength: 漂移强度
            baseline_std: 基线标准差（如果提供则使用，否则使用实例的baseline_std或当前信号std）
        
        Returns:
            (修改后的信号, 异常点索引列表)
        """
        signal_copy = signal.copy()
        # 使用固定的基线标准差，避免异常叠加导致强度放大
        if baseline_std is not None:
            signal_std = baseline_std
        elif self.baseline_std is not None:
            signal_std = self.baseline_std
        else:
            signal_std = np.std(signal)
        
        affected_indices = list(range(start, min(end, len(signal))))
        
        if shift_type == "mean":
            # 均值漂移：加性偏移
            signal_copy[start:end] += strength * signal_std
        elif shift_type == "amplitude":
            # 振幅漂移：乘性缩放
            signal_copy[start:end] *= max(strength - 3.5, 1.2)
        else:
            raise ValueError(f"Unknown shift_type: {shift_type}")
        
        return signal_copy, affected_indices
    
    def inject_collective_anomaly(self, signal: np.ndarray, start: int, end: int,
                                 anomaly_subtype: str = "frequency_change",
                                 baseline_std: float = None, **kwargs) -> Tuple[np.ndarray, List[int]]:
        """
        注入突变段/集体异常
        
        Args:
            signal: 原始信号
            start: 异常开始位置
            end: 异常结束位置
            anomaly_subtype: 异常子类型
                - "frequency_change": 频率改变
                - "noise_increase": 噪声增强
                - "pattern_change": 模式改变
            baseline_std: 基线标准差（如果提供则使用，否则使用实例的baseline_std或当前信号std）
            **kwargs: 子类型特定参数
        
        Returns:
            (修改后的信号, 异常点索引列表)
        """
        signal_copy = signal.copy()
        affected_indices = list(range(start, min(end, len(signal))))
        segment_length = end - start
        
        # 使用固定的基线标准差，避免异常叠加导致强度放大
        if baseline_std is not None:
            signal_std = baseline_std
        elif self.baseline_std is not None:
            signal_std = self.baseline_std
        else:
            signal_std = np.std(signal)
        
        if anomaly_subtype == "frequency_change":
            # 频率改变：用不同周期的信号替换该段
            new_period = kwargs.get("new_period", 10)
            amplitude = kwargs.get("amplitude", np.std(signal[start:end]))
            phase = kwargs.get("phase", 0)
            
            t = np.arange(segment_length)
            new_segment = amplitude * np.sin(2 * np.pi * t / new_period + phase)
            signal_copy[start:end] = new_segment
            
        elif anomaly_subtype == "noise_increase":
            # 噪声增强：增加该段的噪声水平
            noise_multiplier = kwargs.get("noise_multiplier", 5.0)
            noise_std = kwargs.get("noise_std", signal_std * 0.1)
            
            enhanced_noise = rng.normal(0, noise_std * noise_multiplier, segment_length)
            signal_copy[start:end] += enhanced_noise
            
        elif anomaly_subtype == "pattern_change":
            # 模式改变：替换为完全不同的模式（如白噪声）
            pattern_std = kwargs.get("pattern_std", signal_std)
            new_pattern = rng.normal(np.mean(signal[start:end]), 
                                   pattern_std, segment_length)
            signal_copy[start:end] = new_pattern
            
        else:
            raise ValueError(f"Unknown anomaly_subtype: {anomaly_subtype}")
        
        return signal_copy, affected_indices
    
    def inject_periodic_disruption(self, signal: np.ndarray, start: int, end: int,
                                  disruption_type: str = "flatten",
                                  period: int = 100, **kwargs) -> Tuple[np.ndarray, List[int]]:
        """
        注入周期破坏
        
        Args:
            signal: 原始信号
            start: 破坏开始位置
            end: 破坏结束位置
            disruption_type: 破坏类型
                - "flatten": 失去周期（用均值替代）
                - "phase_shift": 相位错位
                - "amplitude_reduce": 振幅减少
            period: 基础周期长度
            **kwargs: 破坏特定参数
        
        Returns:
            (修改后的信号, 异常点索引列表)
        """
        signal_copy = signal.copy()
        affected_indices = list(range(start, min(end, len(signal))))
        
        if disruption_type == "flatten":
            # 失去周期：用段内均值替代，保持少量噪声
            segment_mean = np.mean(signal[start:end])
            flatten_ratio = kwargs.get("flatten_ratio", 0.1)  # 保留10%的原始波动
            
            original_segment = signal[start:end]
            flattened = segment_mean + (original_segment - segment_mean) * flatten_ratio
            signal_copy[start:end] = flattened
            
        elif disruption_type == "phase_shift":
            # 相位错位：整体移动
            shift_amount = kwargs.get("shift_amount", period // 4)
            segment = signal[start:end]
            
            # 循环移位
            shifted_segment = np.concatenate([segment[shift_amount:], segment[:shift_amount]])
            
            signal_copy[start:end] = shifted_segment
            
        elif disruption_type == "amplitude_reduce":
            # 振幅减少
            amplitude_ratio = kwargs.get("amplitude_ratio", 0.2)
            segment_mean = np.mean(signal[start:end])
            
            # 保持均值，减少振幅
            signal_copy[start:end] = segment_mean + (signal[start:end] - segment_mean) * amplitude_ratio
            
        else:
            raise ValueError(f"Unknown disruption_type: {disruption_type}")
        
        return signal_copy, affected_indices
    
    def inject_contextual_anomaly(self, signal: np.ndarray, start: int, end: int,
                                 context_period: int = 100, 
                                 context_shift: str = "temporal",
                                 baseline_std: float = None) -> Tuple[np.ndarray, List[int]]:
        """
        注入上下文异常
        
        Args:
            signal: 原始信号
            start: 异常开始位置
            end: 异常结束位置
            context_period: 上下文周期长度
            context_shift: 上下文偏移类型
                - "temporal": 时间上下文错位
                - "amplitude": 幅度上下文异常
            baseline_std: 基线标准差（如果提供则使用，否则使用实例的baseline_std或当前信号std）
        
        Returns:
            (修改后的信号, 异常点索引列表)
        """
        signal_copy = signal.copy()
        affected_indices = list(range(start, min(end, len(signal))))
        
        # 使用固定的基线标准差，避免异常叠加导致强度放大
        if baseline_std is not None:
            signal_std = baseline_std
        elif self.baseline_std is not None:
            signal_std = self.baseline_std
        else:
            signal_std = np.std(signal)
        
        if context_shift == "temporal":
            # 时间上下文错位：使用其他时间段的正常模式
            # 例如：用"白天"模式替换"夜晚"时段
            context_offset = context_period // 2
            
            # 寻找上下文来源段
            if start - context_offset >= 0:
                source_start = start - context_offset
                source_end = source_start + (end - start)
            elif start + len(signal) // 2 + (end - start) < len(signal):
                source_start = start + len(signal) // 2
                source_end = source_start + (end - start)
            else:
                # 如果找不到合适的来源段，使用噪声替换
                signal_copy[start:end] = rng.normal(
                    np.mean(signal), signal_std, end - start
                )
                return signal_copy, affected_indices
            
            # 替换为上下文来源段
            signal_copy[start:end] = signal[source_start:source_end]
            
        elif context_shift == "amplitude":
            # 幅度上下文异常：在当前上下文中使用不合适的幅度
            local_mean = np.mean(signal[max(0, start-50):min(len(signal), end+50)])
            global_mean = np.mean(signal)
            
            # 使用全局特征替换局部特征
            adjustment = global_mean - local_mean
            signal_copy[start:end] += adjustment
            
        else:
            raise ValueError(f"Unknown context_shift: {context_shift}")
        
        return signal_copy, affected_indices


class AnomalyInsertionStrategy:
    """异常插入策略管理器
    
    负责管理异常类型分配、位置选择、重叠防止等策略
    """
    
    def __init__(self, length: int, contamination_rate: float = 0.05, 
                 min_gap: int = 10, random_seed: int = None):
        """
        初始化插入策略管理器
        
        Args:
            length: 序列长度
            contamination_rate: 污染率（异常占总序列的比例）
            min_gap: 异常事件之间的最小间隔
            random_seed: 随机种子
        """
        self.length = length
        self.contamination_rate = contamination_rate
        self.min_gap = min_gap
        self.random_seed = random_seed
        
        if random_seed is not None:
            global rng
            rng = np.random.default_rng(random_seed)
        
        # 异常类型默认分配比例
        self.default_type_distribution = {
            AnomalyType.POINT_SPIKE: 0.025,
            AnomalyType.LEVEL_SHIFT: 0.35,
            AnomalyType.COLLECTIVE: 0.25,
            AnomalyType.PERIODIC_DISRUPTION: 0.25,
            AnomalyType.CONTEXTUAL: 0.125
        }
        
        # 占用表：记录哪些时间点已被占用
        self.occupancy = np.zeros(length, dtype=bool)
        
    def calculate_anomaly_budget(self) -> int:
        """计算异常总预算（总异常点数）"""
        return int(self.length * self.contamination_rate)
    
    def allocate_anomaly_points(self, type_distribution: Dict[AnomalyType, float] = None) -> Dict[AnomalyType, int]:
        """
        为每种异常类型分配点数配额（不是事件数量）
        
        Args:
            type_distribution: 异常类型分布比例
        
        Returns:
            每种异常类型的点数配额
        """
        if type_distribution is None:
            type_distribution = self.default_type_distribution
        
        # 基于污染率精确控制异常点数量
        total_anomaly_points_budget = int(self.length * self.contamination_rate)
        
        allocation = {}
        
        # 按比例分配点数配额
        remaining_points = total_anomaly_points_budget
        for anomaly_type, ratio in type_distribution.items():
            if ratio > 0:
                allocated_points = int(total_anomaly_points_budget * ratio)
                allocation[anomaly_type] = allocated_points
                remaining_points -= allocated_points
        
        # 将剩余点数分配给第一个非零类型
        if remaining_points > 0:
            for anomaly_type in type_distribution.keys():
                if type_distribution[anomaly_type] > 0:
                    allocation[anomaly_type] += remaining_points
                    break
        
        return allocation
    
    def generate_anomaly_events_by_points(self, type_allocation: Dict[AnomalyType, int]) -> List[AnomalyEvent]:
        """
        基于点数配额生成异常事件（实现用户建议的动态长度分配策略）
        
        Args:
            type_allocation: 每种异常类型的点数配额
        
        Returns:
            异常事件列表
        """
        events = []
        
        # 按异常类型处理顺序：长异常优先，点异常最后
        processing_order = [
            AnomalyType.LEVEL_SHIFT,
            AnomalyType.PERIODIC_DISRUPTION, 
            AnomalyType.COLLECTIVE,
            AnomalyType.CONTEXTUAL,
            AnomalyType.POINT_SPIKE  # 点异常最后处理，填补剩余空间
        ]
        
        for anomaly_type in processing_order:
            points_quota = type_allocation.get(anomaly_type, 0)
            if points_quota <= 0:
                continue
                
            events.extend(self._generate_events_for_type(anomaly_type, points_quota))
        
        return events
    
    def _generate_events_for_type(self, anomaly_type: AnomalyType, points_quota: int) -> List[AnomalyEvent]:
        """
        为特定异常类型生成事件，使用动态长度调整策略
        
        Args:
            anomaly_type: 异常类型
            points_quota: 该类型的点数配额
        
        Returns:
            该类型的异常事件列表
        """
        events = []
        remaining_quota = points_quota
        max_attempts = 100
        
        # 根据异常类型设定基础长度范围
        if anomaly_type == AnomalyType.POINT_SPIKE:
            min_len, max_len = 1, 3
        elif anomaly_type == AnomalyType.LEVEL_SHIFT:
            min_len, max_len = 10, min(200, points_quota)
        elif anomaly_type == AnomalyType.COLLECTIVE:
            min_len, max_len = 8, min(150, points_quota)
        elif anomaly_type == AnomalyType.PERIODIC_DISRUPTION:
            min_len, max_len = 15, min(250, points_quota)
        elif anomaly_type == AnomalyType.CONTEXTUAL:
            min_len, max_len = 5, min(100, points_quota)
        else:
            min_len, max_len = 5, min(50, points_quota)
        
        while remaining_quota > 0:
            # 动态调整最大长度：不能超过剩余配额
            current_max_len = min(max_len, remaining_quota)
            
            if current_max_len < min_len:
                # 如果剩余配额不足以创建最小长度的异常，则停止
                break
            
            # 随机选择这次异常的长度
            if anomaly_type == AnomalyType.POINT_SPIKE:
                # 点异常长度固定为1-3
                length = min(rng.integers(1, 4), remaining_quota)
            else:
                # 其他异常类型使用动态长度
                length = rng.integers(min_len, current_max_len + 1)
            
            # 寻找可用位置
            position = self.find_available_position(length, length, max_attempts)
            if position is None:
                # 如果找不到位置，尝试更短的长度
                if anomaly_type != AnomalyType.POINT_SPIKE and length > min_len:
                    length = max(min_len, length // 2)
                    position = self.find_available_position(length, length, max_attempts)
                
                if position is None:
                    # 仍然找不到位置，结束当前类型的处理
                    break
            
            start, end = position
            actual_length = end - start
            self.mark_occupied(start, end)
            
            # 生成异常强度和参数
            # 异常强度：以基线标准差的倍数表示
            # 范围：2.5-3.5σ，提供从轻微到显著的异常强度
            strength = rng.uniform(3.0, 3.5)
            
            # 创建异常事件
            event = AnomalyEvent(
                type=anomaly_type,
                start=start,
                end=end,
                strength=strength,
                parameters=self._generate_anomaly_parameters(anomaly_type, actual_length),
                seed=rng.integers(0, 2**31)
            )
            events.append(event)
            
            # 更新剩余配额
            remaining_quota -= actual_length
            
            # 安全检查：避免无限循环
            max_attempts -= 1
            if max_attempts <= 0:
                break
        
        return events

    def find_available_position(self, min_length: int, max_length: int, 
                              max_attempts: int = 100) -> Optional[Tuple[int, int]]:
        """
        寻找可用的插入位置
        
        Args:
            min_length: 最小长度需求
            max_length: 最大长度需求
            max_attempts: 最大尝试次数
        
        Returns:
            (start, end) 位置或 None（如果找不到）
        """
        for _ in range(max_attempts):
            # 随机选择长度
            length = rng.integers(min_length, max_length + 1)
            
            # 随机选择起始位置
            max_start = self.length - length
            if max_start <= 0:
                continue
                
            start = rng.integers(0, max_start)
            end = start + length
            
            # 检查是否与已占用区域冲突（包括min_gap缓冲）
            check_start = max(0, start - self.min_gap)
            check_end = min(self.length, end + self.min_gap)
            
            if not np.any(self.occupancy[check_start:check_end]):
                return start, end
        
        return None
    
    def mark_occupied(self, start: int, end: int):
        """标记时间段为已占用"""
        self.occupancy[start:end] = True
    
    def _generate_anomaly_parameters(self, anomaly_type: AnomalyType, length: int) -> Dict:
        """为特定异常类型生成参数"""
        if anomaly_type == AnomalyType.POINT_SPIKE:
            return {
                "window": min(3, length),
                "mode": rng.choice(["additive", "multiplicative"])
            }
        elif anomaly_type == AnomalyType.LEVEL_SHIFT:
            return {
                "shift_type": rng.choice(["mean", "amplitude"])
            }
        elif anomaly_type == AnomalyType.COLLECTIVE:
            return {
                "anomaly_subtype": rng.choice(["frequency_change", "noise_increase", "pattern_change"]),
                "new_period": rng.integers(5, 50),
                "noise_multiplier": rng.uniform(2, 10)
            }
        elif anomaly_type == AnomalyType.PERIODIC_DISRUPTION:
            return {
                "disruption_type": rng.choice(["flatten", "phase_shift", "amplitude_reduce"]),
                "period": rng.integers(20, 100),
                "flatten_ratio": rng.uniform(0.05, 0.3)
            }
        elif anomaly_type == AnomalyType.CONTEXTUAL:
            return {
                "context_period": rng.integers(50, 200),
                "context_shift": rng.choice(["temporal", "amplitude"])
            }
        else:
            return {}


class AnnotationSystem:
    """标注系统
    
    负责生成点级二值标签和段级元数据
    """
    
    def __init__(self, length: int):
        """
        初始化标注系统
        
        Args:
            length: 序列长度
        """
        self.length = length
        self.point_labels = np.zeros(length, dtype=int)  # 点级标签
        self.segment_metadata = []  # 段级元数据
    
    def annotate_events(self, events: List[AnomalyEvent]):
        """
        为异常事件生成标注
        
        Args:
            events: 异常事件列表
        """
        for event in events:
            # 点级标注
            self.point_labels[event.start:event.end] = 1
            
            # 段级元数据
            segment_info = {
                "type": event.type.value,
                "start": event.start,
                "end": event.end,
                "length": event.end - event.start,
                "strength": event.strength,
                "parameters": event.parameters,
                "seed": event.seed
            }
            self.segment_metadata.append(segment_info)
    
    def get_point_labels(self) -> np.ndarray:
        """获取点级标签"""
        return self.point_labels.copy()
    
    def get_segment_metadata(self) -> List[Dict]:
        """获取段级元数据"""
        return copy.deepcopy(self.segment_metadata)
    
    def get_anomaly_statistics(self) -> Dict:
        """获取异常统计信息"""
        total_anomaly_points = np.sum(self.point_labels)
        contamination_rate = total_anomaly_points / self.length
        
        type_counts = {}
        for segment in self.segment_metadata:
            anomaly_type = segment["type"]
            type_counts[anomaly_type] = type_counts.get(anomaly_type, 0) + 1
        
        return {
            "total_points": self.length,
            "anomaly_points": int(total_anomaly_points),
            "contamination_rate": float(contamination_rate),
            "num_segments": len(self.segment_metadata),
            "type_distribution": type_counts
        }


class SyntheticAnomalyDataset:
    """合成异常数据集生成器
    
    整合所有组件，提供完整的数据集生成功能
    """
    
    def __init__(self, random_seed: int = None):
        """
        初始化数据集生成器
        
        Args:
            random_seed: 随机种子
        """
        self.random_seed = random_seed
        if random_seed is not None:
            global rng
            rng = np.random.default_rng(random_seed)
    
    def generate_dataset(self, 
                        length: int = 5000,
                        contamination_rate: float = 0.05,
                        baseline_config: Dict = None,
                        anomaly_type_distribution: Dict[AnomalyType, float] = None,
                        min_gap: int = 10,
                        multivariate: bool = False,
                        num_channels: int = 1) -> Dict:
        """
        生成完整的合成异常数据集
        
        Args:
            length: 序列长度
            contamination_rate: 污染率
            baseline_config: 基线配置
            anomaly_type_distribution: 异常类型分布
            min_gap: 异常间最小间隔
            multivariate: 是否多变量
            num_channels: 通道数（多变量时）
        
        Returns:
            包含数据、标签和元数据的字典
        """
        # 默认基线配置
        if baseline_config is None:
            baseline_config = {
                "trend_config": {"trend_type": "linear", "slope": 0.001, "intercept": 0},
                "periodic_config": {"periods": [100, 24], "amplitudes": [2.0, 0.5], "phases": None},
                "noise_config": {"noise_type": "gaussian", "std": 0.1}
            }
        
        # 生成基线信号
        baseline_generator = BaselineGenerator(length, random_seed=self.random_seed)
        
        if multivariate:
            # 多变量情况
            signals = []
            for _ in range(num_channels):
                # 为每个通道生成略有不同的基线
                channel_config = copy.deepcopy(baseline_config)
                
                # 添加通道间的差异
                if "periodic_config" in channel_config:
                    amplitudes = channel_config["periodic_config"]["amplitudes"]
                    channel_config["periodic_config"]["amplitudes"] = [
                        a * rng.uniform(0.8, 1.2) for a in amplitudes
                    ]
                
                signal = baseline_generator.generate_baseline(**channel_config)
                signals.append(signal)
            
            baseline_signals = np.array(signals).T  # shape: (length, num_channels)
        else:
            # 单变量情况
            baseline_signals = baseline_generator.generate_baseline(**baseline_config)
        
        # 初始化插入策略
        strategy = AnomalyInsertionStrategy(
            length=length, 
            contamination_rate=contamination_rate,
            min_gap=min_gap,
            random_seed=self.random_seed
        )
        
        # 分配异常点数配额
        type_allocation = strategy.allocate_anomaly_points(anomaly_type_distribution)
        
        # 生成异常事件
        events = strategy.generate_anomaly_events_by_points(type_allocation)
        
        # 计算原始基线信号的标准差（用于所有异常注入的强度基准）
        if multivariate:
            # 多变量情况：使用所有通道的平均标准差
            baseline_std = np.mean([np.std(baseline_signals[:, i]) for i in range(num_channels)])
        else:
            # 单变量情况：直接计算标准差
            baseline_std = np.std(baseline_signals)
        
        # 注入异常
        injector = AnomalyInjector(random_seed=self.random_seed, baseline_std=baseline_std)
        
        if multivariate:
            anomalous_signals = baseline_signals.copy()
            for event in events:
                # 决定影响哪些通道（可以是同步或异步）
                if rng.uniform() < 0.7:  # 70%概率影响所有通道（同步异常）
                    affected_channels = list(range(num_channels))
                else:  # 30%概率只影响部分通道（异步异常）
                    num_affected = rng.integers(1, max(2, num_channels))
                    affected_channels = rng.choice(num_channels, num_affected, replace=False)
                
                for channel in affected_channels:
                    signal = anomalous_signals[:, channel]
                    modified_signal, _ = self._inject_single_anomaly(injector, signal, event)
                    anomalous_signals[:, channel] = modified_signal
        else:
            anomalous_signals = baseline_signals.copy()
            for event in events:
                anomalous_signals, _ = self._inject_single_anomaly(injector, anomalous_signals, event)
        
        # 生成标注
        annotation_system = AnnotationSystem(length)
        annotation_system.annotate_events(events)
        
        # 编译结果
        result = {
            "data": anomalous_signals,
            "baseline": baseline_signals,
            "labels": annotation_system.get_point_labels(),
            "segments": annotation_system.get_segment_metadata(),
            "statistics": annotation_system.get_anomaly_statistics(),
            "config": {
                "length": length,
                "contamination_rate": contamination_rate,
                "baseline_config": baseline_config,
                "anomaly_type_distribution": anomaly_type_distribution or strategy.default_type_distribution,
                "min_gap": min_gap,
                "multivariate": multivariate,
                "num_channels": num_channels,
                "random_seed": self.random_seed
            }
        }
        
        return result
    
    def _inject_single_anomaly(self, injector: AnomalyInjector, 
                              signal: np.ndarray, event: AnomalyEvent) -> Tuple[np.ndarray, List[int]]:
        """注入单个异常事件"""
        if event.type == AnomalyType.POINT_SPIKE:
            return injector.inject_point_spike(
                signal, event.start, event.strength, 
                baseline_std=injector.baseline_std, **event.parameters
            )
        elif event.type == AnomalyType.LEVEL_SHIFT:
            return injector.inject_level_shift(
                signal, event.start, event.end, strength=event.strength, 
                baseline_std=injector.baseline_std, **event.parameters
            )
        elif event.type == AnomalyType.COLLECTIVE:
            return injector.inject_collective_anomaly(
                signal, event.start, event.end, 
                baseline_std=injector.baseline_std, **event.parameters
            )
        elif event.type == AnomalyType.PERIODIC_DISRUPTION:
            return injector.inject_periodic_disruption(
                signal, event.start, event.end, **event.parameters
            )
        elif event.type == AnomalyType.CONTEXTUAL:
            return injector.inject_contextual_anomaly(
                signal, event.start, event.end, 
                baseline_std=injector.baseline_std, **event.parameters
            )
        else:
            raise ValueError(f"Unknown anomaly type: {event.type}")


# 可视化和实用函数
def plot_synthetic_dataset(dataset: Dict, save_path: str = None, figsize: Tuple[int, int] = (15, 8)):
    """
    可视化合成数据集
    
    Args:
        dataset: 生成的数据集字典
        save_path: 保存路径（可选）
        figsize: 图形大小
    """
    data = dataset["data"]
    baseline = dataset["baseline"]
    labels = dataset["labels"]
    segments = dataset["segments"]
    
    # 判断是否为多变量
    is_multivariate = len(data.shape) > 1 and data.shape[1] > 1
    
    if is_multivariate:
        num_channels = data.shape[1]
        _, axes = plt.subplots(num_channels, 1, figsize=figsize, sharex=True)
        if num_channels == 1:
            axes = [axes]
        
        for i in range(num_channels):
            ax = axes[i]
            
            # 绘制基线和异常信号
            ax.plot(baseline[:, i], label=f"Baseline Channel {i+1}", alpha=0.7, color="blue")
            ax.plot(data[:, i], label=f"With Anomalies Channel {i+1}", alpha=0.8, color="red")
            
            # 标记异常区域
            anomaly_regions = np.nonzero(labels)[0]
            if len(anomaly_regions) > 0:
                ax.scatter(anomaly_regions, data[anomaly_regions, i], 
                          c="red", s=2, alpha=0.6, label="Anomaly Points")
            
            ax.set_ylabel(f"Channel {i+1}")
            ax.legend()
            ax.grid(True, alpha=0.3)
    else:
        _, ax = plt.subplots(1, 1, figsize=figsize)
        
        # 绘制基线和异常信号
        ax.plot(baseline, label="Baseline", alpha=0.7, color="blue")
        ax.plot(data, label="With Anomalies", alpha=0.8, color="red")
        
        # 标记异常区域
        anomaly_regions = np.nonzero(labels)[0]
        if len(anomaly_regions) > 0:
            ax.scatter(anomaly_regions, data[anomaly_regions], 
                      c="red", s=2, alpha=0.6, label="Anomaly Points")
        
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 添加段级异常标记
    for segment in segments:
        start, end = segment["start"], segment["end"]
        anomaly_type = segment["type"]
        
        if is_multivariate:
            for ax in axes:
                ax.axvspan(start, end, alpha=0.2, color="orange", label=f"{anomaly_type}" if ax == axes[0] else "")
        else:
            ax.axvspan(start, end, alpha=0.2, color="orange", label=f"{anomaly_type}")
    
    plt.xlabel("Time")
    plt.title(f"Synthetic Anomaly Dataset (Contamination: {dataset['statistics']['contamination_rate']:.2%})")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def export_dataset(dataset: Dict, export_path: str, format: str = "npz"):
    """
    导出数据集
    
    Args:
        dataset: 数据集字典
        export_path: 导出路径
        format: 导出格式 ("npz", "csv", "json")
    """
    if format == "npz":
        np.savez_compressed(
            export_path,
            data=dataset["data"],
            baseline=dataset["baseline"],
            labels=dataset["labels"],
            config=json.dumps(dataset["config"], default=str),
            statistics=json.dumps(dataset["statistics"]),
            segments=json.dumps(dataset["segments"], default=str)
        )
    elif format == "csv":
        # 导出为CSV格式
        df = pd.DataFrame({
            "data": dataset["data"].flatten() if len(dataset["data"].shape) == 1 else dataset["data"][:, 0],
            "baseline": dataset["baseline"].flatten() if len(dataset["baseline"].shape) == 1 else dataset["baseline"][:, 0],
            "label": dataset["labels"]
        })
        df.to_csv(export_path, index=False)
    elif format == "json":
        # 导出为JSON格式
        export_data = {
            "data": dataset["data"].tolist(),
            "baseline": dataset["baseline"].tolist(),
            "labels": dataset["labels"].tolist(),
            "segments": dataset["segments"],
            "statistics": dataset["statistics"],
            "config": dataset["config"]
        }
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
    else:
        raise ValueError(f"Unsupported format: {format}")


# 批量生成实验
def generate_experimental_datasets(base_config: Dict, 
                                 variations: List[Dict],
                                 n_repeats: int = 10,
                                 output_dir: str = "./synthetic_datasets") -> List[Dict]:
    """
    生成实验数据集批次
    
    Args:
        base_config: 基础配置
        variations: 配置变化列表
        n_repeats: 每个配置的重复次数
        output_dir: 输出目录
    
    Returns:
        生成的数据集列表
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = []
    dataset_id = 0
    
    for variation in variations:
        # 合并基础配置和变化配置
        config = {**base_config, **variation}
        
        for repeat in range(n_repeats):
            # 使用不同的随机种子
            config["random_seed"] = base_config.get("random_seed", 42) + dataset_id
            
            # 生成数据集
            generator = SyntheticAnomalyDataset(config["random_seed"])
            dataset = generator.generate_dataset(**config)
            
            # 添加ID和重复信息
            dataset["id"] = dataset_id
            dataset["variation"] = variation
            dataset["repeat"] = repeat
            
            # 保存数据集
            filename = f"dataset_{dataset_id:04d}_repeat_{repeat:02d}.npz"
            export_dataset(dataset, os.path.join(output_dir, filename))
            
            datasets.append(dataset)
            dataset_id += 1
    
    return datasets
