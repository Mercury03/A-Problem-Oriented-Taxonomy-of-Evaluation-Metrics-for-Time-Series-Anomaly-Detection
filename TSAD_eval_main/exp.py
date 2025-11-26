"""
Time Series Anomaly Detection Metric Robustness Experiment Framework

This experiment tests whether 20+ time series anomaly detection metrics can resist
random prediction scores through rigorous statistical analysis to evaluate the
reliability and discriminative ability of each metric.

Experiment Objectives:
1. Quantify the performance distribution of each metric under various data scenarios and prediction strategies
2. Test whether metrics can significantly and stably distinguish "informative detectors" from "random/uninformative predictions"

"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to save memory
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import itertools
from joblib import Parallel, delayed
import pickle
import gc  # Garbage collection

# Statistical analysis
from scipy import stats
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.utils import resample
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

# Set warning filters
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Import existing components
import sys
import os
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

from synthetic_anomaly_generator import (
    SyntheticAnomalyDataset, AnomalyType, AnomalyEvent
)
from random_prediction_constructor import PredictionTestSuite
from new_metric import FB_TDF_Metric

# Import real model benchmark
try:
    from fast_anomaly_detection_benchmark import (
        FastModelTrainer, AnomalyDetector, ModelConfig, 
        TimeSeriesDataProcessor, create_fedformer
    )
    from models.DLinear.DLinear import DLinear
    import torch
    REAL_MODELS_AVAILABLE = True
    print("✓ Real models loaded")
except ImportError as e:
    print(f"Warning: Real models unavailable - {e}")
    REAL_MODELS_AVAILABLE = False

# Try to import complex metrics (optional)
COMPLEX_METRICS_AVAILABLE = False
NAB_SCORE_AVAILABLE = False
try:
    from metrics import (
        Pointwise_metrics, DelayThresholdedPointAdjust, PointAdjust, PointAdjustKPercent,
        LatencySparsityAware, Segmentwise_metrics, Composite_f, Affiliation, Range_PR,
        TaF, eTaF, Time_Tolerant, Temporal_Distance, Best_threshold_pw,
        AUC_ROC, AUC_PR_pw, VUS_ROC, VUS_PR, PatK_pw
    )
    COMPLEX_METRICS_AVAILABLE = True
    print("✓ Complex metrics loaded")
    
    # Separately detect NAB_score dependency
    try:
        from metrics import NAB_score
        
        # Further test if NAB_score is truly available
        test_nab = NAB_score(10, [3, 4], [3, 4])
        test_score = test_nab.get_score()
        if not np.isnan(test_score):
            NAB_SCORE_AVAILABLE = True
            print(f"✓ NAB_score metric loaded and tested successfully (test score: {test_score:.3f})")
        else:
            print("Warning: NAB_score imported successfully but test returned NaN, using fallback version")
        
    except ImportError as nab_e:
        print(f"Warning: NAB_score unavailable - {nab_e}")
        print("  Hint: Install nabscore library: pip install nabscore")
    except Exception as test_e:
        print(f"Warning: NAB_score test failed - {test_e}")
        print("  Will use F1-based fallback version")
        
except ImportError as e:
    print(f"Warning: Complex metrics unavailable - {e}")

# Try to import PATE (optional)
PATE_AVAILABLE = False
try:
    # Add PATE module path
    import sys
    import os
    pate_path = os.path.join(os.path.dirname(__file__), 'PATE-main')
    if pate_path not in sys.path:
        sys.path.append(pate_path)
    
    from pate.PATE_metric import PATE
    PATE_AVAILABLE = True
    print("✓ PATE metric loaded")
except ImportError as e:
    print(f"Warning: PATE metric not available - {e}")

class RealModelDetector:
    """Real anomaly detection model wrapper - uses simplified interface"""
    
    def __init__(self, model_name="DLinear"):
        self.model_name = model_name
        self.is_ready = REAL_MODELS_AVAILABLE
        
    def train_and_predict(self, train_data, test_data, contamination_rate=0.1):
        """Train model and make predictions - directly call simplified interface"""
        if not self.is_ready:
            print(f"  Real model unavailable, using random predictions")
            labels = np.random.binomial(1, contamination_rate, len(test_data))
            scores = np.random.uniform(0, 1, len(test_data))
            return labels, scores
        
        try:
            print(f"  Using real model: {self.model_name}")
            
            # Directly call simplified interface
            from fast_anomaly_detection_benchmark import simple_anomaly_detection
            
            labels, scores = simple_anomaly_detection(
                train_data=train_data,
                test_data=test_data, 
                true_labels=np.ones(len(test_data)),  # Placeholder, won't be used
                model_name=self.model_name,
                contamination_rate=contamination_rate,
                window_size=50
            )
            
            print(f"  ✓ Real model prediction completed")
            return labels, scores
            
        except Exception as e:
            print(f"  ✗ Real model failed: {str(e)}")
            # Fallback to random predictions
            labels = np.random.binomial(1, contamination_rate, len(test_data))
            scores = np.random.uniform(0, 1, len(test_data))
            return labels, scores

@dataclass
class ExperimentConfig:
    """Experiment configuration class - optimized version"""
    
    # Dataset generation configuration
    n_datasets: int = 30  # Number of datasets to generate
    dataset_lengths: List[int] = None  # List of sequence lengths
    anomaly_ratios: List[float] = None  # List of anomaly ratios
    
    # Prediction generation configuration
    n_random_trials: int = 20  # Number of random scores
    random_strategies: List[str] = None  # List of random strategies
    
    # Statistical analysis configuration
    alpha: float = 0.05  # Significance level
    effect_size_threshold: float = 0.3  # Effect size threshold
    auc_threshold: float = 0.7  # AUC threshold
    false_positive_threshold: float = 0.1  # False positive threshold
    spearman_threshold: float = 0.5  # Monotonicity threshold
    
    # Output configuration
    output_dir: str = "./experiment_results"
    save_plots: bool = True
    save_data: bool = True
    
    # Performance optimization configuration
    enable_parallel: bool = True  # Enable parallel processing
    max_parallel_jobs: int = 4   # Maximum parallel tasks
    memory_limit_mb: int = 4000  # Memory limit (MB)
    enable_caching: bool = True  # Enable caching
    
    # Multiple experiment configuration (new)
    n_experiment_runs: int = 1  # Number of independent experiment runs (recommended 1-5)
    parallel_model_training: bool = False  # Whether to parallelize real model training (requires multiple GPUs)
    gpu_devices: List[int] = None  # GPU device list (e.g., [0,1,2,3])
    aggregate_runs: str = "mean"  # Aggregation method for multiple runs: "mean", "median", "all"
    
    # Metric selection configuration
    metric_categories: List[str] = None  # Metric category filter 
    skip_slow_metrics: bool = False      # Skip slow-to-compute metrics
    disable_metrics: List[str] = None    # Explicitly disabled metrics list (highest priority)
    
    def __post_init__(self):
        """Set default values"""
        if self.dataset_lengths is None:
            self.dataset_lengths = [5000, 10000, 50000]
        
        if self.anomaly_ratios is None:
            self.anomaly_ratios = [0.05, 0.10, 0.15, 0.20]
        
        if self.random_strategies is None:
            self.random_strategies = [
                "uniform", "normal", "beta", "walk", 
                "periodic", "structured"
            ]
        
        if self.metric_categories is None:
            self.metric_categories = ["all"]  # Default to using all metrics
        if self.disable_metrics is None:
            self.disable_metrics = []
        if self.gpu_devices is None:
            self.gpu_devices = [0]  # Default to using GPU 0
            
        # Optimize configuration validation
        # Only automatically skip slow metrics in large-scale experiments when user hasn't explicitly requested to keep them
        if (self.n_datasets > 10 or max(self.dataset_lengths) > 10000) and self.skip_slow_metrics == False:
            print("  Large-scale experiment detected: enabling performance optimization")
            self.enable_caching = True
            # Note: no longer automatically set skip_slow_metrics here, respect user configuration
        
        # Multiple experiment configuration validation
        if self.n_experiment_runs > 1:
            print(f"  ⚠️  Multiple experiment mode: will run {self.n_experiment_runs} independent experiments")
            if self.parallel_model_training and len(self.gpu_devices) > 1:
                print(f"  ✓ Parallel model training enabled, using GPUs: {self.gpu_devices}")
            elif self.n_experiment_runs > 3:
                print(f"  ⚠️  Warning: running {self.n_experiment_runs} experiments may take a long time")
    
    def get_performance_mode(self) -> str:
        """Get performance mode"""
        if self.n_datasets <= 2 and self.n_random_trials <= 10 and max(self.dataset_lengths) <= 5000:
            return "test"  # Test mode: small scale but complete testing
        elif self.n_datasets <= 5 and max(self.dataset_lengths) <= 5000:
            return "standard"  # Standard mode
        else:
            return "large_scale"  # Large scale mode

@dataclass
class MetricResult:
    """Experiment results for a single metric"""
    metric_name: str
    genuine_scores: List[float]  # Genuine detector scores
    random_scores: List[float]   # Random prediction scores
    oracle_scores: List[float]   # Oracle attack scores
    
    # Statistical test results
    mannwhitney_statistic: float
    mannwhitney_pvalue: float
    effect_size: float
    auc_score: float
    spearman_rho: float
    spearman_pvalue: float
    
    # Descriptive statistics
    random_mean: float
    random_std: float
    genuine_mean: float
    genuine_std: float
    
    # Verdict results
    pass_verdict: str  # "pass", "weak", "fail" - aggregated verdict based on all random trials
    failure_reasons: List[str]
    
    # Per-trial independent verdicts (new)
    per_trial_verdicts: List[str] = None  # Verdict for each random trial ["pass", "weak", "fail", ...]
    per_trial_effect_sizes: List[float] = None  # Effect size for each random trial
    per_trial_aucs: List[float] = None  # AUC for each random trial

@dataclass 
class DatasetResult:
    """Experiment results for a single dataset"""
    dataset_id: str
    dataset_config: Dict[str, Any]
    metric_results: Dict[str, MetricResult]

class MetricRegistry:
    """Metric registry - supports complex metrics"""
    
    def __init__(self):
        self.metrics = {}
        # Binary threshold strategies:
        # 'respect_predictions'  : Directly use incoming predictions (if 0/1), ignore scores
        # 'gt_ratio_topk'        : Use ground truth label ratio for top-k on scores
        # 'fixed_0.5'            : Use 0.5 threshold to binarize scores
        # 'score_median'         : Use median of scores as threshold
        self.binary_threshold_mode = 'respect_predictions'
        
        # Metric name abbreviation mappings (LaTeX format - italic version, for titles etc.)
        self.metric_abbreviations = {
            'pointwise_f1_complex': r'$PwF_{\beta}$',
            'point_adjust': r'$PAF_{\beta}$',
            'segmentwise_f1': r'${S}F_{\beta}$',
            'composite_f': r'$CF_\beta$',
            'delay_threshold_adjust_5': r'${dT\text{-}PAF}^k_{\boldsymbol{\beta}}$',
            'nab_score': r'$NAB$',
            'range_pr_flat': r'$RF^{\alpha}_{\beta}$ (flat)',
            'range_pr_front': r'$RF^{\alpha}_{\beta}$ (front)',
            'time_tolerant_5': r'$TF_\beta^\tau$',
            'time_tolerant_2': r'$TF_\beta^\tau$ (t=2)',
            'temporal_distance': r'$TD$',
            'affiliation': r'$AF_\beta$',
            'taf_default': r'$TaF_\beta^\delta$',
            'etaf_default': r'$eTaF_\beta$',
            'vus_roc_4': r'$VUS^l\text{-}ROC$',
            'vus_pr_4': r'$VUS^l\text{-}PR$',
            'pate_auc_default': r'$PATE$ (AUC)',
            'pate_auc_small': r'$PATE$ (AUC-small)',
            'pate_f1_default': r'$PATE$ (F1)',
            'latency_sparsity_aware_2': r'$\mathit{LSF}_w^{\beta}$',
            'patk': r'$P@K$',
            'patk_5': r'$P@K$ (k=5)',
            'point_adjust_k20': r'$\mathit{K\%\text{-}PAF^k_{\beta}}$',
            'best_threshold_f1': r'$\mathit{Best\text{-}PwF}$',
            'auc_roc': r'$AUC\text{-}ROC$',
            'auc_pr': r'$AUC\text{-}PR$',
            'fb_tdf1': r'$FB\text{-}TDF_1$',
        }
        
        # Metric name abbreviation mappings (upright version, for rotated axis labels)
        # Use \mathrm{} to make letters upright, avoiding visual conflicts with rotated labels
        self.metric_abbreviations_upright = {
            'pointwise_f1_complex': r'$\mathrm{PwF}_{\beta}$',
            'point_adjust': r'$\mathrm{PAF}_{\beta}$',
            'segmentwise_f1': r'$\mathrm{SF}_{\beta}$',
            'composite_f': r'$\mathrm{CF}_\beta$',
            'delay_threshold_adjust_5': r'$\mathrm{dT\text{-}PAF}^k_{\boldsymbol{\beta}}$',
            'nab_score': r'$\mathrm{NAB}$',
            'range_pr_flat': r'$\mathrm{RF}^{\alpha}_{\beta}$ (flat)',
            'range_pr_front': r'$\mathrm{RF}^{\alpha}_{\beta}$ (front)',
            'time_tolerant_5': r'$\mathrm{TF}_\beta^\tau$',
            'time_tolerant_2': r'$\mathrm{TF}_\beta^\tau$ (t=2)',
            'temporal_distance': r'$\mathrm{TD}$',
            'affiliation': r'$\mathrm{AF}_\beta$',
            'taf_default': r'$\mathrm{TaF}_\beta^\delta$',
            'etaf_default': r'$\mathrm{eTaF}_\beta$',
            'vus_roc_4': r'$\mathrm{VUS}^l\text{-}\mathrm{ROC}$',
            'vus_pr_4': r'$\mathrm{VUS}^l\text{-}\mathrm{PR}$',
            'pate_auc_default': r'$\mathrm{PATE}$ (AUC)',
            'pate_auc_small': r'$\mathrm{PATE}$ (AUC-small)',
            'pate_f1_default': r'$\mathrm{PATE}$ (F1)',
            'latency_sparsity_aware_2': r'$\mathrm{LSF}_w^{\beta}$',
            'patk': r'$\mathrm{P@K}$',
            'patk_5': r'$\mathrm{P@K}$ (k=5)',
            'point_adjust_k20': r'$\mathrm{K\%\text{-}PAF}^k_{\beta}$',
            'best_threshold_f1': r'$\mathrm{Best\text{-}PwF}$',
            'auc_roc': r'$\mathrm{AUC\text{-}ROC}$',
            'auc_pr': r'$\mathrm{AUC\text{-}PR}$',
            'fb_tdf1': r'$\mathrm{FB\text{-}TDF}_1$',
        }
        
        self._register_all_metrics()
    
    @staticmethod
    def interpret_nab_score(score: float) -> str:
        """Interpret the meaning of NAB score"""
        if np.isnan(score):
            return "Invalid"
        elif score < 0:
            return "Harmful"
        elif score < 50:
            return "Below Baseline"
        elif score < 70:
            return "Good"
        else:
            return "Excellent"
    
    def get_display_name(self, metric_name: str, use_abbreviation: bool = False, 
                        use_upright: bool = False) -> str:

        if use_abbreviation:
            if use_upright and metric_name in self.metric_abbreviations_upright:
                return self.metric_abbreviations_upright[metric_name]
            elif metric_name in self.metric_abbreviations:
                return self.metric_abbreviations[metric_name]
        
        if metric_name == 'pointwise_f1_complex':
            return 'pointwise_f1'
        
        return metric_name


    def set_binary_threshold_mode(self, mode: str):
        allowed = {'respect_predictions', 'gt_ratio_topk', 'fixed_0.5', 'score_median'}
        if mode not in allowed:
            raise ValueError(f"Unsupported binary threshold mode: {mode}. Allowed: {allowed}")
        self.binary_threshold_mode = mode
    
    def _register_all_metrics(self):
        

        def _pointwise_f1_topk(y_true: np.ndarray, y_score: np.ndarray) -> float:
            """Point-wise F1 (fair version):
            Use the anomaly ratio of true labels to perform top-k thresholding on continuous scores, then calculate F1.
            If there are no true anomalies, return 0.
            """
            if y_true is None or y_score is None:
                return 0.0
            y_true = np.asarray(y_true).astype(int)
            y_score = np.asarray(y_score).astype(float)
            n = len(y_true)
            if n == 0:
                return 0.0
            anomaly_ratio = float(np.mean(y_true))
            if anomaly_ratio <= 0:
                return 0.0
            k = int(round(n * anomaly_ratio))
            if k <= 0:
                return 0.0
            if k >= n:
                pred = np.ones(n, dtype=int)
            else:
                # Handle NaN/Inf
                finite_mask = np.isfinite(y_score)
                if not np.all(finite_mask):
                    if np.any(finite_mask):
                        min_valid = np.min(y_score[finite_mask])
                        y_score = np.where(finite_mask, y_score, min_valid - 1e-9)
                    else:
                        return 0.0
                kth = np.partition(y_score, -k)[-k]
                pred = (y_score >= kth).astype(int)
            return f1_score(y_true, pred, zero_division=0)

        self.metrics["pointwise_f1"] = {
            "func": _pointwise_f1_topk,
            "type": "binary_detection",
            "description": "Point-wise F1 score (top-k by GT anomaly ratio)"
        }
        
        '''        self.metrics["pointwise_precision"] = {
            "func": lambda y_true, y_score: precision_score(y_true, (y_score > 0.5).astype(int), zero_division=0),
            "type": "binary_detection",
            "description": "Point-wise Precision (simplified)"
        }
        
        self.metrics["pointwise_recall"] = {
            "func": lambda y_true, y_score: recall_score(y_true, (y_score > 0.5).astype(int), zero_division=0),
            "type": "binary_detection",
            "description": "Point-wise Recall (simplified)"
        }'''
        

        self._register_fb_tdf1_metric()
        

        if COMPLEX_METRICS_AVAILABLE:
            self._register_complex_metrics()

        if PATE_AVAILABLE:
            self._register_pate_metric()
    
    def _register_fb_tdf1_metric(self):

        
        def _fb_tdf1_wrapper(y_true: np.ndarray, y_score: np.ndarray) -> float:
            """
            FB-TDF1

            """
            try:
                y_true = np.asarray(y_true).astype(int)
                y_score = np.asarray(y_score).astype(float)
                
                anomaly_ratio = np.sum(y_true) / len(y_true)
                
                if anomaly_ratio == 0:
                    return 0.0
                

                k = int(np.ceil(anomaly_ratio * len(y_score)))
                k = max(1, min(k, len(y_score)))  
                

                threshold = np.partition(y_score, -k)[-k]
                

                y_pred = (y_score >= threshold).astype(int)
                
                fb_metric = FB_TDF_Metric(
                    tolerance=5,
                    mode='balanced',
                    enable_timeliness=False,
                    timeliness_decay=0.5,
                    dispersed_decay=1.0
                )
                
                metrics = fb_metric.compute(y_true, y_pred)
                
                return metrics['f1_score']
                
            except Exception as e:
                warnings.warn(f"FB-TDF1 calculation failed: {e}")
                return np.nan
        
        self.metrics["fb_tdf1"] = {
            "func": _fb_tdf1_wrapper,
            "type": "binary_detection",
            "description": "Fuzzy Boundary Time-aware Detection F1 (FB-TDF1)"
        }
    
    def _register_complex_metrics(self):

        
        self.metrics["pointwise_f1_complex"] = {
            "class": Pointwise_metrics,
            "type": "binary_detection",
            "description": "Point-wise F1 score (complex)"
        }
        

        self.metrics["point_adjust"] = {
            "class": PointAdjust,
            "type": "binary_detection",
            "description": "Point-Adjust F1"
        }
        
        self.metrics["delay_threshold_adjust_5"] = {
            "class": DelayThresholdedPointAdjust,
            "type": "binary_detection", 
            "params": {"k": 5},
            "description": "Delay Thresholded Point Adjust (k=5)"
        }
        
        self.metrics["point_adjust_k20"] = {
            "class": PointAdjustKPercent,
            "type": "binary_detection",
            "params": {"k": 0.2},
            "description": "Point-Adjust K% F1 (k=20%)"
        }
        

        self.metrics["latency_sparsity_aware_2"] = {
            "class": LatencySparsityAware,
            "type": "binary_detection",
            "params": {"tw": 2},
            "description": "Latency Sparsity Aware F1 (tw=2)"
        }
        

        self.metrics["segmentwise_f1"] = {
            "class": Segmentwise_metrics,
            "type": "binary_detection",
            "description": "Segment-wise F1"
        }
        

        self.metrics["composite_f"] = {
            "class": Composite_f,
            "type": "binary_detection",
            "description": "Composite F1 score"
        }
        
        self.metrics["affiliation"] = {
            "class": Affiliation,
            "type": "binary_detection",
            "description": "Affiliation-based metrics"
        }
        
        self.metrics["range_pr_flat"] = {
            "class": Range_PR,
            "type": "binary_detection",
            "params": {"alpha": 0.2, "bias": "flat"},
            "description": "Range-based PR (flat, alpha=0.2)"
        }
        
        self.metrics["range_pr_front"] = {
            "class": Range_PR,
            "type": "binary_detection",
            "params": {"alpha": 0.2, "bias": "front"},
            "description": "Range-based PR (front, alpha=0.2)"
        }
        

        self.metrics["taf_default"] = {
            "class": TaF,
            "type": "binary_detection",
            "params": {"theta": 0.5, "alpha": 0.5, "delta": 0},
            "description": "Time-aware F1 (default params)"
        }
        
        self.metrics["etaf_default"] = {
            "class": eTaF,
            "type": "binary_detection",
            "params": {"theta_p": 0.5, "theta_r": 0.1},
            "description": "Enhanced Time-aware F1 (default params)"
        }
        
        
        self.metrics["time_tolerant_5"] = {
            "class": Time_Tolerant,
            "type": "binary_detection",
            "params": {"d": 5},
            "description": "Time Tolerant (d=5)"
        }
        
        # 7. 距离指标（越小越好）
        self.metrics["temporal_distance"] = {
            "class": Temporal_Distance,
            "type": "binary_detection",
            "description": "Temporal Distance",
            "lower_is_better": True  # 标记为越小越好的指标
        }
        
        # 8. NAB评分
        if NAB_SCORE_AVAILABLE:
            self.metrics["nab_score"] = {
                "class": NAB_score,
                "type": "binary_detection", 
                "description": "NAB Score (Numenta Anomaly Benchmark)"
            }
        else:
            # 简单占位符
            self.metrics["nab_score"] = {
                "func": lambda y_true, y_score: 0.0,  # 直接返回0
                "type": "binary_detection",
                "description": "NAB Score (Unavailable)"
            }
        
        # 9. 基于阈值的最优指标 (需要连续分数)
        self.metrics["best_threshold_f1"] = {
            "class": Best_threshold_pw,
            "type": "nonbinary_detection",
            "description": "Best threshold F1"
        }
        
        # 10. AUC指标 (需要连续分数)
        self.metrics["auc_roc"] = {
            "class": AUC_ROC,
            "type": "nonbinary_detection",
            "description": "AUC ROC"
        }
        
        self.metrics["auc_pr"] = {
            "class": AUC_PR_pw,
            "type": "nonbinary_detection",
            "description": "AUC PR"
        }
        
        # 11. VUS指标 (需要连续分数)
        try:
            self.metrics["vus_roc_4"] = {
                "class": VUS_ROC,
                "type": "nonbinary_detection",
                "params": {"max_window": 4},
                "description": "VUS ROC (window=4)"
            }
            
            self.metrics["vus_pr_4"] = {
                "class": VUS_PR,
                "type": "nonbinary_detection",
                "params": {"max_window": 4},
                "description": "VUS PR (window=4)"
            }
        except:
            print("警告: VUS指标不可用")
        
        # 12. PatK指标 (需要连续分数)
        try:
            self.metrics["patk"] = {
                "class": PatK_pw,
                "type": "nonbinary_detection",
                "description": "Precision@K"
            }
        except:
            print("警告: PatK指标不可用")
    
    def _register_pate_metric(self):
        """Register PATE metrics"""
        # PATE - 连续分数模式 (AUC-PR)
        self.metrics["pate_auc_default"] = {
            "func": lambda y_true, y_score: PATE(y_true, y_score, binary_scores=False, e_buffer=100, d_buffer=100),
            "type": "nonbinary_detection",
            "description": "PATE AUC-PR (default buffers: 100/100)"
        }
        
        self.metrics["pate_auc_small"] = {
            "func": lambda y_true, y_score: PATE(y_true, y_score, binary_scores=False, e_buffer=50, d_buffer=50),
            "type": "nonbinary_detection", 
            "description": "PATE AUC-PR (small buffers: 50/50)"
        }
        
        # PATE-F1 - 二元分数模式 (F1分数)
        def pate_f1_metric(y_true, y_score):
            """PATE F1 metric wrapper"""
            # 将连续分数转换为二元预测 (阈值0.5)
            y_binary = (y_score > 0.5).astype(int)
            return PATE(y_true, y_binary, binary_scores=True, e_buffer=100, d_buffer=100)
        
        self.metrics["pate_f1_default"] = {
            "func": pate_f1_metric,
            "type": "nonbinary_detection",
            "description": "PATE F1 score (binary mode, buffers: 100/100)"
        }
        
        
    def get_metric_names(self, performance_mode: str = "standard") -> List[str]:
        """Get all metric names - support performance mode filtering"""
        all_metrics = list(self.metrics.keys())
        # 先应用显式禁用
        if hasattr(self, 'config') and getattr(self, 'config') is not None:
            disabled = set(getattr(self.config, 'disable_metrics', []) or [])
            if disabled:
                all_metrics = [m for m in all_metrics if m not in disabled]
        # If explicitly requested not to skip slow metrics, return directly
        # debug/test 模式下 ExperimentConfig 会把 n_datasets 设置得很小，此时需要全部指标
        if hasattr(self, 'config'):
            cfg = getattr(self, 'config')
            # 如果调用者把 registry 挂在 experiment 上，可通过外部设置 skip_slow_metrics
            skip = getattr(cfg, 'skip_slow_metrics', False)
            if not skip:
                return all_metrics
        
        if performance_mode == "large_scale":
            slow_metrics = {
                "vus_roc_4", "vus_pr_4", "pate_auc_default", "pate_auc_small",
                "segmentwise_f1", "composite_f", "affiliation"
            }
            return [m for m in all_metrics if m not in slow_metrics]
        return all_metrics
    
    def get_metric_priority(self, metric_name: str) -> int:
        """Get metric priority (for optimizing processing order)"""
        # 优先级：1=最高（最快），5=最低（最慢）
        priority_map = {
            "pointwise_f1": 1,
            "auc_roc": 2,
            "auc_pr": 2,
            "point_adjust": 2,
            "time_tolerant_2": 3,
            "best_threshold_f1": 3,
            "nab_score": 3,
            "pate_auc_default": 4,
            "segmentwise_f1": 4,
            "vus_roc_4": 5,
            "vus_pr_4": 5,
        }
        return priority_map.get(metric_name, 3)  # 默认中等优先级
    
    def compute_metric(self, metric_name: str, gt_labels: np.ndarray, 
                      predictions: np.ndarray, scores: Optional[np.ndarray] = None) -> float:
        """Calculate scores for specified metrics - optimized version"""
        
        if metric_name not in self.metrics:
            raise ValueError(f"Unknown metric: {metric_name}")
        
        # 输入验证和快速退出
        if len(gt_labels) == 0:
            return 0.0
        
        # 如果所有标签都是0或都是1，某些指标会返回固定值
        if len(np.unique(gt_labels)) == 1:
            if metric_name == "auc_roc":
                return 0.5  # 无区分度时AUC为0.5
            elif "f1" in metric_name.lower():
                return 0.0  # F1分数为0
        
        metric_info = self.metrics[metric_name]
        
        try:
            if "func" in metric_info:
                # Use function to calculate metrics
                if scores is None:
                    scores = predictions.astype(float)
                return metric_info["func"](gt_labels, scores)
            
            elif "class" in metric_info:
                # Use class to calculate metrics - optimized version
                metric_class = metric_info["class"]
                metric_type = metric_info["type"]
                params = metric_info.get("params", {})
                
                if metric_type == "binary_detection":
                    # 二元检测指标：优化版
                    length = len(gt_labels)
                    
                    # Get continuous scores
                    if scores is None:
                        scores = predictions.astype(float)
                    else:
                        scores = scores.astype(float)

                    # 快速处理异常值
                    finite_mask = np.isfinite(scores)
                    if not np.all(finite_mask):
                        if np.any(finite_mask):
                            min_valid = np.min(scores[finite_mask])
                            scores = np.where(finite_mask, scores, min_valid - 1e-9)
                        else:
                            return np.nan

                    # 优化的top-k阈值化
                    anomaly_ratio = float(np.mean(gt_labels))
                    if anomaly_ratio <= 0 or length == 0:
                        pred_labels = np.zeros(length, dtype=int)
                    else:
                        k = int(round(length * anomaly_ratio))
                        if k <= 0:
                            pred_labels = np.zeros(length, dtype=int)
                        elif k >= length:
                            pred_labels = np.ones(length, dtype=int)
                        else:
                            # Use numpy's fast partition algorithm
                            kth_value = np.partition(scores, -k)[-k]
                            pred_labels = (scores >= kth_value).astype(int)
                    
                    # 快速转换为点级表示
                    gt_anomalies = np.where(gt_labels == 1)[0]
                    pred_anomalies = np.where(pred_labels == 1)[0]
                    
                    # NAB_score特殊处理 - 关键修复：NAB需要完整的异常分数序列
                    if metric_name == "nab_score":
                        # Check key prerequisites
                        if len(gt_anomalies) == 0:
                            score = np.nan
                        else:
                            try:
                                # Critical fix: create complete anomaly score sequence, not just using point-level indices
                                if scores is None:
                                    # Create score sequence from binary predictions
                                    full_pred_scores = pred_labels.astype(float)
                                else:
                                    full_pred_scores = scores.copy()
                                
                                # 确保分数在合理范围内
                                full_pred_scores = np.clip(full_pred_scores, 0.0, 1.0)
                                     
                                # 重要修复：将点级异常转换为段格式
                                def points_to_segments(points):
                                    # 处理numpy数组和列表
                                    if isinstance(points, np.ndarray):
                                        if len(points) == 0:
                                            return []
                                        points = points.tolist()
                                    elif not points:  # 空列表
                                        return []
                                    
                                    segments = []
                                    start = points[0]
                                    end = points[0]
                                    for i in range(1, len(points)):
                                        if points[i] == end + 1:
                                            end = points[i]
                                        else:
                                            segments.append((start, end))
                                            start = points[i]
                                            end = points[i]
                                    segments.append((start, end))
                                    
                                    # NAB算法修复：确保每个段至少有2个点宽度，避免除零错误
                                    fixed_segments = []
                                    for start, end in segments:
                                        if start == end:  # 单点段
                                            # 扩展为至少2个点宽的段
                                            if start > 0:
                                                fixed_segments.append((start-1, end))
                                            else:
                                                fixed_segments.append((start, end+1))
                                        else:
                                            fixed_segments.append((start, end))
                                    
                                    return fixed_segments
                                
                                window_limits = points_to_segments(gt_anomalies)
                                timestamps = np.arange(length)
                                
                                
                                # 手动调用Sweeper来调试
                                from nabscore import Sweeper
                                sweeper = Sweeper(
                                    probationPercent=0.1, 
                                    costMatrix={
                                        "fnWeight": 1.0, 
                                        "fpWeight": 1.0, 
                                        "tnWeight": 1.0, 
                                        "tpWeight": 1.0
                                    }
                                )
                                
                                # Call calcSweepScore - use complete score sequence
                                anomaly_list = sweeper.calcSweepScore(timestamps, full_pred_scores, window_limits, "debug_dataset")
                                
                                # Check AnomalyPoint results
                                non_zero_sweep = [p for p in anomaly_list if abs(p.sweepScore) > 1e-10]
                                # Calculate threshold score
                                scores_by_threshold = sweeper.calcScoreByThreshold(anomaly_list)
                                
                                if len(scores_by_threshold) >= 2:
                                    null_score = scores_by_threshold[0].score
                                    raw_score = scores_by_threshold[1].score

                                    # Calculate perfect prediction score
                                    perfect_scores = np.zeros(length)
                                    perfect_scores[gt_anomalies] = 1.0
                                    perfect_anomaly_list = sweeper.calcSweepScore(timestamps, perfect_scores, window_limits, "perfect_dataset")
                                    perfect_scores_by_threshold = sweeper.calcScoreByThreshold(perfect_anomaly_list)
                                    
                                    if len(perfect_scores_by_threshold) >= 2:
                                        perfect_null = perfect_scores_by_threshold[0].score
                                        perfect_score = perfect_scores_by_threshold[1].score
                                        
                                        
                                        # Calculate final NAB score
                                        # NAB分数解释：
                                        # - 100分: 完美预测
                                        # - 0分: 等同于不预测（baseline）
                                        # - 负分: 比不预测还差（有害预测，通常因误报过多）
                                        # - 分数范围: (-∞, 100]，负值是正常的数学结果
                                        if abs(perfect_score - null_score) > 1e-10:
                                            score = (raw_score - null_score) / (perfect_score - null_score) * 100
                                            
                                        else:
                                            score = 0.0
                                    else:
                                        score = 0.0
                                else:
                                    score = 0.0
                                
                            except Exception as e:
                                print(f"    NAB: {str(e)}")
                                import traceback
                                traceback.print_exc()
                                score = 0.0
                        
                    else:
                        # 其他指标的标准处理
                        if params:
                            metric_obj = metric_class(length, gt_anomalies, pred_anomalies, **params)
                        else:
                            metric_obj = metric_class(length, gt_anomalies, pred_anomalies)
                        
                        score = metric_obj.get_score()
                    
                    # NAB_score最终调试输出
                    if metric_name == "nab_score":
                        interpretation = MetricRegistry.interpret_nab_score(score)
                        
                    
                    return score
                    
                elif metric_type == "nonbinary_detection":
                    # 非二元检测指标：优化版
                    if scores is None:
                        scores = predictions.astype(float)
                    
                    # 快速转换为点级表示
                    gt_anomalies = np.where(gt_labels == 1)[0]
                    
                    # Create metric object
                    if params:
                        metric_obj = metric_class(gt_anomalies, scores, **params)
                    else:
                        metric_obj = metric_class(gt_anomalies, scores)
                    
                    return metric_obj.get_score()
                
                else:
                    raise ValueError(f"Unknown metric type: {metric_type}")
            
            else:
                return np.nan
                
        except Exception as e:
            # 优化的错误处理：对于已知问题直接返回默认值
            error_msg = str(e).lower()
            
            # NAB_score特殊处理
            if metric_name == "nab_score":
                # 尝试简单回退
                if len(np.unique(gt_labels)) > 1 and scores is not None:
                    try:
                        from sklearn.metrics import f1_score
                        pred_binary = (scores > 0.5).astype(int)
                        fallback_score = f1_score(gt_labels, pred_binary, zero_division=0) * 100  # NAB风格评分
                        return fallback_score
                    except:
                        return 0.0
                else:
                    return 0.0
            
            # 其他复杂指标的处理
            elif "complex" in metric_name or metric_name in ["segmentwise_f1", "composite_f", "affiliation"]:
                try:
                    # 快速回退到F1分数
                    if scores is not None:
                        pred_binary = (scores > 0.5).astype(int)
                    else:
                        pred_binary = predictions.astype(int)
                    return f1_score(gt_labels, pred_binary, zero_division=0)
                except:
                    return 0.0
            
            # 对于其他错误，根据指标类型返回合理的默认值
            if "auc" in metric_name.lower():
                return 0.5
            elif "f1" in metric_name.lower() or "precision" in metric_name.lower() or "recall" in metric_name.lower():
                return 0.0
            else:
                return np.nan

class MetricRobustnessExperiment:
    """Metric robustness experiment main class - optimized version"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.metric_registry = MetricRegistry()
        # 关联配置到注册表，便于在大规模模式下根据 skip_slow_metrics 决定是否过滤慢指标
        self.metric_registry.config = config
        
        # Memory usage monitoring
        self._monitor_memory = True
        self._max_memory_mb = 4000  # Maximum memory usage limit (MB)
        
        # Determine the number of models to use based on configuration (test mode vs normal mode)
        if config.n_datasets == 1 and config.n_random_trials <= 5:
            # Test mode: only use the fastest model

            self.real_models = {
                'DLinear': RealModelDetector(model_name="DLinear"),
            }
        else:
            # Normal mode: use complete model collection

            self.real_models = {
                'DLinear': RealModelDetector(model_name="DLinear"),
                'DeepAR': RealModelDetector(model_name="DeepAR"), 
                'Informer': RealModelDetector(model_name="Informer"),
            }
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Experiment results storage
        self.results: Dict[str, DatasetResult] = {}
        self.summary_stats: Dict[str, Any] = {}
        
        # Cache initialization
        self._random_predictions_cache = {}
        self._quality_gradient_cache = {}
        
    def _check_memory_usage(self):
        """Check memory usage"""
        if not self._monitor_memory:
            return
        
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self._max_memory_mb:

                self._cleanup_caches()
                gc.collect()  # 强制垃圾回收
                
                # Check again
                memory_mb = process.memory_info().rss / 1024 / 1024

                
        except ImportError:
            # If psutil is not available, use simple cache cleanup strategy
            if len(self._random_predictions_cache) > 5:
                self._cleanup_caches()
    
    def _optimize_for_memory(self):
        """Optimize configuration for memory usage"""
        # Limit cache size
        self._max_cache_size = 5
        
        # Disable some memory-intensive visualizations
        if hasattr(self.config, 'save_plots') and len(self.results) > 10:

            self.config.save_plots = False
        
        # Reduce quality gradient points
        if not hasattr(self, '_quality_levels_reduced'):
            self._quality_levels_reduced = True

        
    def _create_dataset_for_constructor(self, dataset_dict: Dict) -> Dict:
        """Prepare dataset format for PredictionTestSuite"""
        # dataset_dict已经是正确格式，只需要确保有segments字段
        if "segments" not in dataset_dict:
            # 从labels中提取segments
            labels = dataset_dict["labels"]
            segments = []
            in_segment = False
            start = 0
            
            for i, label in enumerate(labels):
                if label == 1 and not in_segment:
                    start = i
                    in_segment = True
                elif label == 0 and in_segment:
                    segments.append({"start": start, "end": i})
                    in_segment = False
            
            if in_segment:
                segments.append({"start": start, "end": len(labels)})
            
            dataset_dict["segments"] = segments
        
        return dataset_dict
        
    def generate_experimental_datasets(self) -> List[Tuple[str, Dict]]:
        """Generate experimental datasets
        
        Note: SyntheticAnomalyDataset has automatically generated through default_type_distribution
        Contains mixed datasets of multiple anomaly types, no need to loop through anomaly types
        """
        datasets = []
        
        dataset_id = 0
        for length in self.config.dataset_lengths:
            for anomaly_ratio in self.config.anomaly_ratios:
                    # Generate datasets containing mixed anomaly types
                    dataset = SyntheticAnomalyDataset(random_seed=42 + dataset_id)
                    
                    # SyntheticAnomalyDataset.generate_dataset() automatically includes multiple anomaly types
                    # Allocated through default_type_distribution:
                    # POINT_SPIKE: 5%, LEVEL_SHIFT: 35%, COLLECTIVE: 25%, 
                    # PERIODIC_DISRUPTION: 25%, CONTEXTUAL: 10%
                    dataset_result = dataset.generate_dataset(
                        length=length,
                        contamination_rate=anomaly_ratio
                    )
                    
                    dataset_id_str = f"ds_{dataset_id:03d}_mixed_anomalies_{length}_{anomaly_ratio}"
                    datasets.append((dataset_id_str, dataset_result))
                    dataset_id += 1
                    
                    if len(datasets) >= self.config.n_datasets:
                        return datasets
        
        return datasets
    
    def _generate_controlled_random_predictions(self, metric_name: str, test_suite: PredictionTestSuite, 
                                               gt_labels: np.ndarray) -> List[float]:
        """Generate streamlined but powerful random prediction control group (optimized version)"""
        
        # Use cache key to avoid recalculating the same random predictions
        cache_key = (len(gt_labels), float(np.mean(gt_labels)), self.config.n_random_trials)
        if not hasattr(self, '_random_predictions_cache'):
            self._random_predictions_cache = {}
        
        if cache_key in self._random_predictions_cache:
            cached_predictions = self._random_predictions_cache[cache_key]
            # Calculate current metric for cached predictions
            random_scores = []
            for pred_labels, pred_scores in cached_predictions:
                score = self.metric_registry.compute_metric(metric_name, gt_labels, pred_labels, pred_scores)
                if not np.isnan(score):
                    random_scores.append(score)
            print(f"      Using cached {len(random_scores)} random prediction scores")
            return random_scores
        
        true_anomaly_rate = np.mean(gt_labels)
        length = len(gt_labels)
        random_scores = []
        all_predictions = []
        
        # Streamlined to 3 core random strategies
        n_pure = self.config.n_random_trials // 2
        n_cluster = self.config.n_random_trials // 4
        n_uniform = self.config.n_random_trials - n_pure - n_cluster

        
        # Strategy 1: Uniform score prediction (representing "position random with fixed anomaly rate")
        for _ in range(n_uniform):
            pred = test_suite.random_gen.uniform_score_prediction(length, top_k_ratio=true_anomaly_rate)
            all_predictions.append((pred.labels.copy(), pred.scores.copy()))
        
        # Strategy 2: Clustered random prediction (test structured randomness, enhance challenge)
        for _ in range(n_cluster):
            try:
                cluster_size = np.random.randint(5, 15)
                avg_cluster_size = cluster_size * 1.5
                cluster_rate = (true_anomaly_rate / avg_cluster_size) * np.random.uniform(0.8, 1.2)
                cluster_rate = np.clip(cluster_rate, 0.01, 0.15)
                
                pred = test_suite.random_gen.clustered_random_prediction(
                    length, cluster_rate=cluster_rate, 
                    cluster_size_range=(cluster_size, cluster_size*2)
                )
                
                # Debug info (optional)
                actual_rate = np.mean(pred.labels)
                if actual_rate > true_anomaly_rate * 2.5:
                    print(f"      Warning: Cluster prediction anomaly rate too high ({actual_rate:.2%} vs {true_anomaly_rate:.2%})")
                
                all_predictions.append((pred.labels.copy(), pred.scores.copy()))
            except Exception as e:
                pred = test_suite.random_gen.uniform_score_prediction(length, top_k_ratio=true_anomaly_rate)
                all_predictions.append((pred.labels.copy(), pred.scores.copy()))

        # Strategy 3: Pure random prediction (absolute baseline, anomaly rate also random)
        for _ in range(n_pure):
            random_anomaly_rate = np.random.uniform(0.01, true_anomaly_rate)
            pred = test_suite.random_gen.bernoulli_prediction(length, p=random_anomaly_rate)
            all_predictions.append((pred.labels.copy(), pred.scores.copy()))
        
        # Cache prediction results
        self._random_predictions_cache[cache_key] = all_predictions
        
        # Calculate metric scores
        for pred_labels, pred_scores in all_predictions:
            score = self.metric_registry.compute_metric(metric_name, gt_labels, pred_labels, pred_scores)
            if not np.isnan(score):
                random_scores.append(score)
        
        print(f"      Generated {len(random_scores)} valid random prediction scores (uniform:{n_uniform}, cluster:{n_cluster}, pure random:{n_pure})")
        return random_scores
    
    def _generate_quality_gradient_predictions(self, metric_name: str, gt_labels: np.ndarray) -> List[float]:
        """Generate controlled quality gradient prediction family (from perfect to very poor controllable variants) - optimized version"""
        
        length = len(gt_labels)
        
        # Use cache to avoid recalculating the same quality gradient
        cache_key = (length, tuple(gt_labels), metric_name)
        if not hasattr(self, '_quality_gradient_cache'):
            self._quality_gradient_cache = {}
        
        if cache_key in self._quality_gradient_cache:
            cached_scores = self._quality_gradient_cache[cache_key]
            print(f"      Using cached {len(cached_scores)} quality gradient scores")
            return cached_scores
        
        quality_scores = []
        
        # Reduce quality levels to improve speed (from 19 to 10)
        quality_levels = np.linspace(0.9, 0.1, 10)  # From high quality to low quality
        
        print(f"      Quickly generate quality gradient (optimized version)...")
        
        # Pre-calculate some common values
        gt_mean = np.mean(gt_labels)
        indices = np.arange(length)
        
        for quality in quality_levels:
            try:
                if quality > 0.7:
                    # High quality: small amount of noise
                    pred_labels = gt_labels.copy()
                    n_flip = int(length * (1 - quality) * 0.1)
                    if n_flip > 0:
                        flip_indices = np.random.choice(indices, n_flip, replace=False)
                        pred_labels[flip_indices] = 1 - pred_labels[flip_indices]
                    pred_scores = pred_labels.astype(float) + np.random.normal(0, 0.1, length)
                    
                elif quality > 0.4:
                    # Medium quality: partially correct
                    pred_labels = gt_labels.copy()
                    n_flip = int(length * (1 - quality))
                    if n_flip > 0 and n_flip < length:
                        flip_indices = np.random.choice(indices, n_flip, replace=False)
                        pred_labels[flip_indices] = 1 - pred_labels[flip_indices]
                    pred_scores = pred_labels.astype(float) + np.random.normal(0, 0.2, length)
                    
                elif quality > 0.2:
                    # Low quality: mostly random, retain small amount of real signal
                    pred_labels = np.random.binomial(1, gt_mean, length)
                    n_keep = int(length * quality)
                    if n_keep > 0 and n_keep < length:
                        keep_indices = np.random.choice(indices, n_keep, replace=False)
                        pred_labels[keep_indices] = gt_labels[keep_indices]
                    pred_scores = pred_labels.astype(float) + np.random.normal(0, 0.3, length)
                    
                else:
                    # Very poor quality: basically random
                    pred_labels = np.random.binomial(1, gt_mean, length)
                    pred_scores = np.random.uniform(0, 1, length)
                
                # Calculate metric score
                score = self.metric_registry.compute_metric(metric_name, gt_labels, pred_labels, pred_scores)
                
                # NAB_score special debugging
                if metric_name == "nab_score" and (score == 0.0 or np.isnan(score)):
                    print(f"      NAB quality gradient debugging (quality={quality:.2f}): score={score}")
                    print(f"        GT anomaly count: {np.sum(gt_labels)}, predicted anomaly count: {np.sum(pred_labels)}")
                    print(f"        Predicted score range: [{np.min(pred_scores):.3f}, {np.max(pred_scores):.3f}]")
                
                if not np.isnan(score):
                    quality_scores.append(score)
                    
            except Exception as e:
                # If generation fails, use simple random prediction
                pred_labels = np.random.binomial(1, gt_mean, length)
                pred_scores = np.random.uniform(0, 1, length)
                try:
                    score = self.metric_registry.compute_metric(metric_name, gt_labels, pred_labels, pred_scores)
                    if not np.isnan(score):
                        quality_scores.append(score)
                except:
                    continue
        
        # Cache result
        self._quality_gradient_cache[cache_key] = quality_scores
        
        print(f"      Generated {len(quality_scores)} quality gradient scores")
        return quality_scores
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run complete experiment - optimized version, supports multiple independent runs"""
        print("=== Start metric robustness experiment ===")
        print(f"Configuration: {self.config.n_datasets} datasets, {self.config.n_random_trials} random trials")
        
        if self.config.n_experiment_runs > 1:
            print(f"⚙️  Multiple experiment mode: will run {self.config.n_experiment_runs} independent experiments")
            print(f"   Aggregation method: {self.config.aggregate_runs}")
            if self.config.parallel_model_training:
                print(f"   Parallel training: using GPUs {self.config.gpu_devices}")
        
        start_time = time.time()
        
        # Multiple experiment runs
        all_run_results = []
        for run_id in range(self.config.n_experiment_runs):
            run_seed = 42 + run_id
            
            if self.config.n_experiment_runs > 1:
                print(f"\n{'='*60}")
                print(f"运行实验 #{run_id + 1}/{self.config.n_experiment_runs} (seed={run_seed})")
                print(f"{'='*60}")
            
            # 运行单次实验
            run_result = self._run_single_experiment(run_seed)
            all_run_results.append({
                'run_id': run_id,
                'seed': run_seed,
                'results': run_result['results'],
                'summary_stats': run_result['summary_stats']
            })
            
            print(f"\n✓ Experiment #{run_id + 1} completed")
        
        # Aggregate results from multiple runs
        print(f"\n{'='*60}")
        print("Aggregate multiple experiment results...")
        print(f"{'='*60}")
        
        if self.config.n_experiment_runs == 1:
            # Single run: use results directly
            self.results = all_run_results[0]['results']
            self.summary_stats = all_run_results[0]['summary_stats']
        else:
            # Multiple runs: save all raw results while computing aggregate statistics
            self.all_run_results = all_run_results
            self.results, self.summary_stats = self._aggregate_multiple_runs(all_run_results)
        
        # Save raw experiment data (for subsequent independent plotting)
        print("\nSaving experiment data...")
        self._save_raw_experiment_data()
        
        # Generate report
        print("\nGenerating experiment report...")
        self._generate_experiment_report()
        
        # Final cleanup
        self._cleanup_caches()
        
        total_elapsed = time.time() - start_time
        print(f"\nExperiment total time: {total_elapsed:.1f}s")
        
        return {
            "config": asdict(self.config),
            "results": self.results,
            "summary": self.summary_stats,
            "all_runs": all_run_results if self.config.n_experiment_runs > 1 else None
        }
    
    def _run_single_experiment(self, random_seed: int) -> Dict[str, Any]:
        """Run single experiment (can be called multiple times)"""
        # Generate experimental datasets
        print("\n1. Generate experimental datasets...")
        datasets = self.generate_experimental_datasets()
        print(f"Generated {len(datasets)} datasets")
        
        # Estimate total workload
        total_metrics = len(self.metric_registry.get_metric_names())
        total_work = len(datasets) * total_metrics
        print(f"Total need to evaluate: {total_work} (dataset × metric) combinations")
        
        # 初始化缓存（每次运行都重置）
        self._random_predictions_cache = {}
        self._quality_gradient_cache = {}
        
        # Initialize result storage
        single_run_results = {}
        
        # Run experiment for each dataset - optimized version
        print("\n2. Running dataset experiments...")
        completed_work = 0
        
        for i, (dataset_id, dataset_dict) in enumerate(datasets):
            dataset_start = time.time()
            print(f"\nProcessing dataset {i+1}/{len(datasets)}: {dataset_id}")
            
            # Prepare data
            gt_labels = dataset_dict["labels"]
            time_series = dataset_dict["data"]
            dataset_for_constructor = self._create_dataset_for_constructor(dataset_dict)
            
            # Create prediction test suite (create once)
            test_suite = PredictionTestSuite(dataset_for_constructor, random_seed=42)
            
            # Pre-generate adversarial predictions (generate once, use multiple times)
            adversarial_results = test_suite.generate_adversarial_suite()
            oracle_scores_base = []
            for strategy_name, predictions in adversarial_results.items():
                if isinstance(predictions, list):
                    for pred in predictions:
                        if hasattr(pred, 'labels'):
                            oracle_scores_base.append((pred.labels, pred.scores))
                        else:
                            oracle_scores_base.append((pred, None))
                else:
                    if hasattr(predictions, 'labels'):
                        oracle_scores_base.append((predictions.labels, predictions.scores))
                    else:
                        oracle_scores_base.append((predictions, None))
            
            # Pre-generate real model predictions (generate once, use multiple times)
            genuine_predictions_base = []
            
            # 准备训练数据
            test_data = dataset_dict.get("data", [])
            if "baseline" in dataset_dict:
                train_data = dataset_dict["baseline"]
            else:
                normal_indices = np.where(gt_labels == 0)[0]
                if len(normal_indices) > 100:
                    train_data = [test_data[i] for i in normal_indices[:len(normal_indices)//2]]
                else:
                    train_data = test_data[:len(test_data)//2]
            
            contamination_rate = np.mean(gt_labels)
            
            # 根据配置选择串行或并行训练
            if self.config.parallel_model_training and len(self.config.gpu_devices) > 1 and REAL_MODELS_AVAILABLE:
                print(f"    Using parallel training ({len(self.config.gpu_devices)} GPUs)...")
                try:
                    from parallel_model_trainer import train_models_parallel
                    
                    model_names = list(self.real_models.keys())
                    results_dict = train_models_parallel(
                        model_names=model_names,
                        train_data=np.array(train_data),
                        test_data=np.array(test_data),
                        gt_labels=gt_labels,
                        contamination_rate=contamination_rate,
                        gpu_devices=self.config.gpu_devices,
                        random_seed=random_seed
                    )
                    
                    for model_name, (pred_labels, pred_scores) in results_dict.items():
                        genuine_predictions_base.append((model_name, pred_labels, pred_scores))
                    
                    print(f"    ✓ 并行训练完成，{len(genuine_predictions_base)}个模型")
                    
                except Exception as e:
                    print(f"    ✗ 并行训练失败: {e}")
                    print(f"    回退到串行训练...")
                    # 回退到串行训练
                    for model_name, model_detector in self.real_models.items():
                        try:
                            pred_labels, pred_scores = model_detector.train_and_predict(
                                train_data, test_data, contamination_rate
                            )
                            genuine_predictions_base.append((model_name, pred_labels, pred_scores))
                        except Exception as e2:
                            print(f"    ✗ {model_name}模型预测失败: {str(e2)}")
                            continue
            else:
                # 串行训练模型
                for model_name, model_detector in self.real_models.items():
                    try:
                        pred_labels, pred_scores = model_detector.train_and_predict(
                            train_data, test_data, contamination_rate
                        )
                        genuine_predictions_base.append((model_name, pred_labels, pred_scores))
                    except Exception as e:
                        print(f"    ✗ {model_name}模型预测失败: {str(e)}")
                        continue
            
            # Use optimized metric name retrieval
            performance_mode = self.config.get_performance_mode()
            metric_names = self.metric_registry.get_metric_names(performance_mode)
            
            # 根据优先级排序指标（优先处理快速指标）
            if self.config.enable_parallel:
                metric_names = sorted(metric_names, 
                                    key=lambda x: self.metric_registry.get_metric_priority(x))
            
            print(f"    Using {len(metric_names)} metrics (performance mode: {performance_mode})")
            
            # 启用内存优化
            self._optimize_for_memory()
            
            # Important: If model parallel training is used, disable metric parallel processing to avoid spawn process conflicts
            # Using multi-threading/multi-processing again in spawn-launched subprocesses will cause deadlock
            use_metric_parallel = (
                len(metric_names) > 4 and 
                self.config.enable_parallel and 
                not self.config.parallel_model_training  # 如果模型并行，则禁用指标并行
            )
            
            if use_metric_parallel:
                print(f"    Using parallel processing for {len(metric_names)} metrics...")
                try:
                    # Use joblib for parallel metric processing
                    from joblib import Parallel, delayed
                    
                    def process_single_metric(metric_name):
                        return self._evaluate_metric_robustness_fast(
                            metric_name, test_suite, gt_labels, oracle_scores_base, genuine_predictions_base
                        )
                    
                    # Parallel execution (use fewer processes to avoid memory pressure)
                    results_parallel = Parallel(n_jobs=min(4, len(metric_names)), backend='threading')(
                        delayed(process_single_metric)(metric_name) for metric_name in metric_names
                    )
                    
                    # 收集结果
                    dataset_results = {}
                    for j, (metric_name, result) in enumerate(zip(metric_names, results_parallel)):
                        dataset_results[metric_name] = result
                        completed_work += 1
                        progress = (completed_work / total_work) * 100
                        print(f"    ✓ {metric_name} - 总进度: {progress:.1f}%")
                        
                except Exception as e:
                    print(f"    并行处理失败，回退到串行处理: {str(e)}")
                    # 回退到串行处理
                    dataset_results = self._process_metrics_serial(
                        metric_names, test_suite, gt_labels, oracle_scores_base, 
                        genuine_predictions_base, completed_work, total_work
                    )
                    completed_work += len(metric_names)
            else:
                # 串行处理指标（模型并行时必须串行，避免死锁）
                if self.config.parallel_model_training:
                    print(f"    串行处理 {len(metric_names)} 个指标（模型已并行）...")
                dataset_results = self._process_metrics_serial(
                    metric_names, test_suite, gt_labels, oracle_scores_base, 
                    genuine_predictions_base, completed_work, total_work
                )
                completed_work += len(metric_names)
            
            dataset_elapsed = time.time() - dataset_start
            print(f"  Dataset completed ({dataset_elapsed:.1f}s)")
            
            # Save dataset results
            single_run_results[dataset_id] = DatasetResult(
                dataset_id=dataset_id,
                dataset_config={
                    "length": len(time_series),
                    "anomaly_ratio": np.mean(gt_labels) * 100,
                    "n_anomalies": len(dataset_dict.get("segments", []))
                },
                metric_results=dataset_results
            )
            
            # Regularly clean cache to control memory usage
            if i > 0 and i % 2 == 0:  # Clean cache every 2 datasets processed
                self._cleanup_caches()
        
        # Temporarily set self.results for summary statistics
        self.results = single_run_results
        
        # Summary analysis
        print("\n3. Summary statistical analysis...")
        summary_stats = self._compute_summary_statistics()
        
        # Clean cache
        self._cleanup_caches()
        
        return {
            "results": single_run_results,
            "summary_stats": summary_stats
        }
    
    def _process_metrics_serial(self, metric_names, test_suite, gt_labels, 
                              oracle_scores_base, genuine_predictions_base, 
                              completed_work, total_work):
        """串行处理指标"""
        dataset_results = {}
        for j, metric_name in enumerate(metric_names):
            print(f"  评估指标 ({j+1}/{len(metric_names)}): {metric_name}")
            try:
                start_time_metric = time.time()
                dataset_results[metric_name] = self._evaluate_metric_robustness_fast(
                    metric_name, test_suite, gt_labels, oracle_scores_base, genuine_predictions_base
                )
                elapsed = time.time() - start_time_metric
                completed_work += 1
                progress = (completed_work / total_work) * 100
                print(f"    ✓ Completed ({elapsed:.1f}s) - Total progress: {progress:.1f}%")
            except Exception as e:
                print(f"    ✗ Failed: {str(e)}")
                # Create default result
                dataset_results[metric_name] = MetricResult(
                    metric_name=metric_name,
                    genuine_scores=[0.0],
                    random_scores=[0.0],
                    oracle_scores=[0.0],
                    mannwhitney_statistic=0,
                    mannwhitney_pvalue=1.0,
                    effect_size=0.0,
                    auc_score=0.5,
                    spearman_rho=0.0,
                    spearman_pvalue=1.0,
                    random_mean=0.0,
                    random_std=0.0,
                    genuine_mean=0.0,
                    genuine_std=0.0,
                    pass_verdict="fail",
                    failure_reasons=[f"Computation error: {str(e)}"]
                )
        return dataset_results
    
    def _aggregate_multiple_runs(self, all_run_results: List[Dict]) -> Tuple[Dict, Dict]:
        """
        聚合多次实验运行的结果
        
        策略：
        - Plotting: Use all raw data points from runs (not averaged)
        - Tables: Use aggregate statistics (based on config.aggregate_runs)
        
        Returns:
            (aggregated_results, aggregated_summary_stats)
        """
        print(f"Using aggregation method: {self.config.aggregate_runs}")
        
        n_runs = len(all_run_results)
        if n_runs == 1:
            return all_run_results[0]['results'], all_run_results[0]['summary_stats']
        
        # 收集所有运行的数据
        all_results_by_dataset = {}
        all_summary_stats = []
        
        for run_data in all_run_results:
            all_summary_stats.append(run_data['summary_stats'])
            
            for dataset_id, dataset_result in run_data['results'].items():
                if dataset_id not in all_results_by_dataset:
                    all_results_by_dataset[dataset_id] = []
                all_results_by_dataset[dataset_id].append(dataset_result)
        
        # 聚合每个数据集的结果
        aggregated_results = {}
        
        for dataset_id, dataset_results_list in all_results_by_dataset.items():
            # 聚合指标结果
            aggregated_metric_results = {}
            
            # Get all metric names
            metric_names = set()
            for dr in dataset_results_list:
                metric_names.update(dr.metric_results.keys())
            
            for metric_name in metric_names:
                # 收集该指标在所有运行中的结果
                metric_results_across_runs = []
                for dr in dataset_results_list:
                    if metric_name in dr.metric_results:
                        metric_results_across_runs.append(dr.metric_results[metric_name])
                
                if not metric_results_across_runs:
                    continue
                
                # 合并所有运行的scores（用于绘图）
                all_genuine_scores = []
                all_random_scores = []
                all_oracle_scores = []
                
                for mr in metric_results_across_runs:
                    all_genuine_scores.extend(mr.genuine_scores)
                    all_random_scores.extend(mr.random_scores)
                    all_oracle_scores.extend(mr.oracle_scores)
                
                # Calculate aggregate statistics (for tables)
                if self.config.aggregate_runs == "mean":
                    agg_effect_size = np.mean([mr.effect_size for mr in metric_results_across_runs])
                    agg_auc_score = np.mean([mr.auc_score for mr in metric_results_across_runs])
                    agg_genuine_mean = np.mean([mr.genuine_mean for mr in metric_results_across_runs])
                    agg_random_mean = np.mean([mr.random_mean for mr in metric_results_across_runs])
                elif self.config.aggregate_runs == "median":
                    agg_effect_size = np.median([mr.effect_size for mr in metric_results_across_runs])
                    agg_auc_score = np.median([mr.auc_score for mr in metric_results_across_runs])
                    agg_genuine_mean = np.median([mr.genuine_mean for mr in metric_results_across_runs])
                    agg_random_mean = np.median([mr.random_mean for mr in metric_results_across_runs])
                else:  # "all" - use first run value as representative
                    agg_effect_size = metric_results_across_runs[0].effect_size
                    agg_auc_score = metric_results_across_runs[0].auc_score
                    agg_genuine_mean = metric_results_across_runs[0].genuine_mean
                    agg_random_mean = metric_results_across_runs[0].random_mean
                
                # Create aggregated MetricResult
                # Use first run as template, but use aggregated data
                template = metric_results_across_runs[0]
                aggregated_metric_results[metric_name] = MetricResult(
                    metric_name=metric_name,
                    genuine_scores=all_genuine_scores,  # 所有运行的数据点（用于绘图）
                    random_scores=all_random_scores,
                    oracle_scores=all_oracle_scores,
                    mannwhitney_statistic=template.mannwhitney_statistic,
                    mannwhitney_pvalue=template.mannwhitney_pvalue,
                    effect_size=agg_effect_size,  # 聚合后的值（用于表格）
                    auc_score=agg_auc_score,
                    spearman_rho=template.spearman_rho,
                    spearman_pvalue=template.spearman_pvalue,
                    random_mean=agg_random_mean,
                    random_std=np.std(all_random_scores),
                    genuine_mean=agg_genuine_mean,
                    genuine_std=np.std(all_genuine_scores),
                    pass_verdict=template.pass_verdict,
                    failure_reasons=template.failure_reasons,
                    per_trial_verdicts=template.per_trial_verdicts,
                    per_trial_effect_sizes=template.per_trial_effect_sizes,
                    per_trial_aucs=template.per_trial_aucs
                )
            
            # 使用第一次运行的dataset_config
            aggregated_results[dataset_id] = DatasetResult(
                dataset_id=dataset_id,
                dataset_config=dataset_results_list[0].dataset_config,
                metric_results=aggregated_metric_results
            )
        
        # 聚合summary_stats（对表格统计数据进行聚合）
        aggregated_summary = self._aggregate_summary_stats(all_summary_stats)
        
        print(f"✓ Completed aggregation of {n_runs} experiments")
        print(f"  - Plotting uses: {len(all_genuine_scores)} data points (all runs)")
        print(f"  - Table uses: {self.config.aggregate_runs} aggregated statistics")
        
        return aggregated_results, aggregated_summary
    
    def _aggregate_summary_stats(self, all_summary_stats: List[Dict]) -> Dict:
        """聚合多次运行的汇总统计"""
        if not all_summary_stats:
            return {}
        
        if len(all_summary_stats) == 1:
            return all_summary_stats[0]
        
        # 提取metric_performance数据
        all_metric_perfs = []
        for summary in all_summary_stats:
            if 'metric_performance' in summary:
                all_metric_perfs.append(summary['metric_performance'])
        
        if not all_metric_perfs:
            return all_summary_stats[0]
        
        # 聚合每个指标的性能统计
        aggregated_metric_perf = {}
        metric_names = set()
        for mp in all_metric_perfs:
            metric_names.update(mp.keys())
        
        for metric_name in metric_names:
            values_across_runs = []
            for mp in all_metric_perfs:
                if metric_name in mp:
                    values_across_runs.append(mp[metric_name])
            
            if not values_across_runs:
                continue
            
            # 提取数值字段进行聚合
            numeric_fields = ['effect_size', 'auc_score', 'mannwhitney_pvalue', 
                            'random_mean', 'genuine_mean', 'spearman_rho']
            
            aggregated_values = {}
            for field in numeric_fields:
                field_values = [v.get(field, 0) for v in values_across_runs if field in v]
                if field_values:
                    if self.config.aggregate_runs == "mean":
                        aggregated_values[field] = np.mean(field_values)
                    elif self.config.aggregate_runs == "median":
                        aggregated_values[field] = np.median(field_values)
                    else:  # "all"
                        aggregated_values[field] = field_values[0]
            
            # 使用第一次运行的其他字段
            aggregated_metric_perf[metric_name] = {**values_across_runs[0], **aggregated_values}
        
        # 构建聚合的summary_stats
        aggregated_summary = {**all_summary_stats[0]}
        aggregated_summary['metric_performance'] = aggregated_metric_perf
        aggregated_summary['n_experiment_runs'] = len(all_summary_stats)
        aggregated_summary['aggregate_method'] = self.config.aggregate_runs
        
        return aggregated_summary
    
    def _evaluate_metric_robustness_fast(self, metric_name: str, test_suite: PredictionTestSuite,
                                        gt_labels: np.ndarray, oracle_scores_base: List, 
                                        genuine_predictions_base: List) -> MetricResult:
        """快速评估单个指标的抗随机性 - 优化版"""
        
        # Use pre-generated random predictions
        random_scores = self._generate_controlled_random_predictions(metric_name, test_suite, gt_labels)
        
        # Use pre-calculated adversarial prediction scores
        oracle_scores = []
        for pred_labels, pred_scores in oracle_scores_base:
            try:
                score = self.metric_registry.compute_metric(metric_name, gt_labels, pred_labels, pred_scores)
                if not np.isnan(score):
                    oracle_scores.append(score)
            except:
                continue
        
        # Use pre-generated quality gradient
        quality_gradient_scores = self._generate_quality_gradient_predictions(metric_name, gt_labels)
        
        # Use pre-calculated real model scores
        genuine_scores = []
        for model_name, pred_labels, pred_scores in genuine_predictions_base:
            try:
                score = self.metric_registry.compute_metric(metric_name, gt_labels, pred_labels, pred_scores)
                if not np.isnan(score):
                    genuine_scores.append(score)
            except:
                continue
        
        # 如果没有真实分数，记录警告（不再使用完美预测作为fallback）
        if len(genuine_scores) == 0:
            print(f"    ⚠ Warning: Metric '{metric_name}' all real models failed, cannot calculate genuine_scores")
            # Use a medium score as placeholder to avoid calculation failure
            genuine_scores = [0.5]  # 使用中等分数而不是完美预测
        
        # 统计分析（优化版）
        return self._compute_metric_statistics_fast(
            metric_name, genuine_scores, random_scores, oracle_scores, quality_gradient_scores
        )
    
    def _compute_metric_statistics_fast(self, metric_name: str, genuine_scores: List[float], 
                                      random_scores: List[float], oracle_scores: List[float],
                                      quality_gradient_scores: List[float]) -> MetricResult:
        """Fast calculation of metric statistics - optimized version"""
        
        if len(random_scores) > 0 and len(genuine_scores) > 0:
            # 快速统计检验
            try:
                mannwhitney_stat, mannwhitney_p = mannwhitneyu(
                    genuine_scores, random_scores, alternative='two-sided'
                )
            except:
                mannwhitney_stat, mannwhitney_p = 0, 1.0
            
            # Check metric type
            metric_info = self.metric_registry.metrics.get(metric_name, {})
            lower_is_better = metric_info.get("lower_is_better", False)
            
            # Fast statistical calculation
            random_mean = np.mean(random_scores)
            random_std = np.std(random_scores)
            genuine_mean = np.mean(genuine_scores)
            genuine_std = np.std(genuine_scores)
            
            # 效应量
            pooled_std = np.sqrt(((len(random_scores)-1)*random_std**2 + 
                                (len(genuine_scores)-1)*genuine_std**2) / 
                               (len(random_scores) + len(genuine_scores) - 2))
            
            if lower_is_better:
                effect_size = (random_mean - genuine_mean) / pooled_std if pooled_std > 0 else 0
            else:
                effect_size = (genuine_mean - random_mean) / pooled_std if pooled_std > 0 else 0
            
            # AUC
            try:
                y_true = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(random_scores))])
                if lower_is_better:
                    y_scores = np.concatenate([-np.array(genuine_scores), -np.array(random_scores)])
                else:
                    y_scores = np.concatenate([genuine_scores, random_scores])
                auc_score = roc_auc_score(y_true, y_scores)
            except:
                auc_score = 0.5
            
            # 单调性
            try:
                if len(quality_gradient_scores) > 1:
                    quality_levels = np.linspace(0.9, 0.1, len(quality_gradient_scores))
                    # Check if all scores are the same (variance = 0)
                    if np.var(quality_gradient_scores) == 0:
                        # 所有分数相同，单调性无意义，设为0
                        spearman_rho, spearman_p = 0.0, 1.0
                    else:
                        if lower_is_better:
                            spearman_rho, spearman_p = spearmanr(quality_levels, quality_gradient_scores)
                            spearman_rho = -spearman_rho if not np.isnan(spearman_rho) else 0.0
                        else:
                            spearman_rho, spearman_p = spearmanr(quality_levels, quality_gradient_scores)
                            spearman_rho = spearman_rho if not np.isnan(spearman_rho) else 0.0
                        
                        # 确保p值不是NaN
                        if np.isnan(spearman_p):
                            spearman_p = 1.0
                else:
                    spearman_rho, spearman_p = 0.0, 1.0
            except:
                spearman_rho, spearman_p = 0.0, 1.0
        else:
            mannwhitney_stat = mannwhitney_p = 0
            effect_size = auc_score = 0
            spearman_rho = spearman_p = 0
            random_mean = random_std = genuine_mean = genuine_std = 0
            lower_is_better = False
        
        # 判定逻辑（基于汇总统计）- 改进Oracle评估
        pass_verdict, failure_reasons = self._make_pass_verdict(
            mannwhitney_p, effect_size, auc_score,
            oracle_scores,  # 传入完整的oracle_scores列表
            genuine_scores,  # 传入完整的genuine_scores列表
            random_scores,   # 传入完整的random_scores列表
            random_mean / (random_mean + genuine_mean) if (random_mean + genuine_mean) > 0 else 0,
            spearman_rho,
            lower_is_better  # 传入指标类型
        )
        
        # Calculate independent effect size, AUC and judgment for each random trial (new)
        per_trial_effect_sizes = []
        per_trial_aucs = []
        per_trial_verdicts = []
        
        if genuine_scores and random_scores:
            # 假设random_scores按顺序存储每次随机试验的分数
            for random_score in random_scores:
                # Calculate the effect size for this random trial (genuine vs single random)
                if lower_is_better:
                    trial_effect = (random_score - genuine_mean) / genuine_std if genuine_std > 0 else 0
                else:
                    trial_effect = (genuine_mean - random_score) / genuine_std if genuine_std > 0 else 0
                per_trial_effect_sizes.append(trial_effect)
                
                # 计算该随机试验的AUC
                try:
                    y_true = np.concatenate([np.ones(len(genuine_scores)), np.zeros(1)])
                    if lower_is_better:
                        y_scores = np.concatenate([-np.array(genuine_scores), -np.array([random_score])])
                    else:
                        y_scores = np.concatenate([genuine_scores, [random_score]])
                    trial_auc = roc_auc_score(y_true, y_scores)
                except:
                    trial_auc = 0.5
                per_trial_aucs.append(trial_auc)
                
                # 为该随机试验做判定
                # 注意：这里简化判定，主要基于效应量和AUC
                trial_verdict = "fail"
                if trial_effect >= self.config.effect_size_threshold and trial_auc >= self.config.auc_threshold:
                    trial_verdict = "pass"
                elif trial_effect >= self.config.effect_size_threshold * 0.5 or trial_auc >= self.config.auc_threshold * 0.9:
                    trial_verdict = "weak"
                per_trial_verdicts.append(trial_verdict)
        
        return MetricResult(
            metric_name=metric_name,
            genuine_scores=genuine_scores,
            random_scores=random_scores,
            oracle_scores=oracle_scores,
            mannwhitney_statistic=mannwhitney_stat,
            mannwhitney_pvalue=mannwhitney_p,
            effect_size=effect_size,
            auc_score=auc_score,
            spearman_rho=spearman_rho,
            spearman_pvalue=spearman_p,
            random_mean=random_mean,
            random_std=random_std,
            genuine_mean=genuine_mean,
            genuine_std=genuine_std,
            pass_verdict=pass_verdict,
            failure_reasons=failure_reasons,
            per_trial_verdicts=per_trial_verdicts,
            per_trial_effect_sizes=per_trial_effect_sizes,
            per_trial_aucs=per_trial_aucs
        )
    
    def _cleanup_caches(self):
        """Clean cache to control memory usage - optimized version"""
        if hasattr(self, '_random_predictions_cache'):
            # Clean cache items that exceed the limit
            max_cache_size = getattr(self, '_max_cache_size', 5)
            if len(self._random_predictions_cache) > max_cache_size:
                # 保留最新的缓存项
                items = list(self._random_predictions_cache.items())
                self._random_predictions_cache = dict(items[-max_cache_size:])
                print(f"    Cleaning random prediction cache, keeping {max_cache_size} items")
        
        if hasattr(self, '_quality_gradient_cache'):
            max_cache_size = getattr(self, '_max_cache_size', 5)
            if len(self._quality_gradient_cache) > max_cache_size:
                items = list(self._quality_gradient_cache.items())
                self._quality_gradient_cache = dict(items[-max_cache_size:])
                print(f"    Cleaning quality gradient cache, keeping {max_cache_size} items")
        
        # 强制垃圾回收
        gc.collect()
        
        # Check memory usage
        self._check_memory_usage()
    
    def _evaluate_metric_robustness(self, metric_name: str, test_suite: PredictionTestSuite,
                                  gt_labels: np.ndarray) -> MetricResult:
        """评估单个指标的抗随机性"""
        
        print(f"    Generating random predictions...")
        # Use improved random strategy generation (ensure anomaly rate matching)
        random_scores = self._generate_controlled_random_predictions(metric_name, test_suite, gt_labels)
        
        print(f"    Generating adversarial predictions...")
        adversarial_results = test_suite.generate_adversarial_suite()
        
        oracle_scores = []
        for strategy_name, predictions in adversarial_results.items():
            if isinstance(predictions, list):
                for pred in predictions:
                    if hasattr(pred, 'labels'):
                        score = self.metric_registry.compute_metric(metric_name, gt_labels, pred.labels, pred.scores)
                    else:
                        score = self.metric_registry.compute_metric(metric_name, gt_labels, pred)
                    if not np.isnan(score):
                        oracle_scores.append(score)
            else:
                if hasattr(predictions, 'labels'):
                    score = self.metric_registry.compute_metric(metric_name, gt_labels, predictions.labels, predictions.scores)
                else:
                    score = self.metric_registry.compute_metric(metric_name, gt_labels, predictions)
                if not np.isnan(score):
                    oracle_scores.append(score)
        
        print(f"    Generating controlled quality gradient detector...")
        # Generate controlled quality gradient family (controllable variants from perfect to very poor) 
        quality_gradient_scores = self._generate_quality_gradient_predictions(metric_name, gt_labels)
        
        # Generate benchmark scores for "real detectors" (using multiple real models)
        genuine_scores = []
        
        # 尝试使用多个真实模型
        for model_name, model_detector in self.real_models.items():
            try:
                # Get dataset configuration 
                dataset_dict = test_suite.dataset
                test_data = dataset_dict.get("data", [])
                
                # 分离正常数据和异常数据进行训练
                if "baseline" in dataset_dict:
                    train_data = dataset_dict["baseline"]  # 使用基线数据作为正常训练数据
                else:
                    # 如果没有基线数据，使用测试数据中的正常部分
                    normal_indices = np.where(gt_labels == 0)[0]
                    if len(normal_indices) > 100:  # 确保有足够的正常数据
                        train_data = [test_data[i] for i in normal_indices[:len(normal_indices)//2]]
                    else:
                        train_data = test_data[:len(test_data)//2]  # 使用前半部分作为训练
                
                contamination_rate = np.mean(gt_labels)
                
                # 使用真实模型进行训练和预测
                pred_labels, pred_scores = model_detector.train_and_predict(
                    train_data, test_data, contamination_rate
                )
                
                genuine_score = self.metric_registry.compute_metric(metric_name, gt_labels, pred_labels, pred_scores)
                
                if not np.isnan(genuine_score):
                    genuine_scores.append(genuine_score)
                    print(f"    ✓ {model_name}模型分数: {genuine_score:.4f}")
                else:
                    print(f"    ✗ {model_name}模型分数无效")
                    
            except Exception as e:
                print(f"    ✗ {model_name}模型预测失败: {str(e)}")
        
        # 如果没有任何真实模型可用，记录警告（不再使用完美预测）
        if len(genuine_scores) == 0:
            print(f"    ⚠ 警告: 所有真实模型都失败，使用中等分数作为占位符")
            genuine_scores = [0.5]  # 使用中等分数而不是完美预测
        
        print(f"    统计分析...")
        # 统计分析
        if len(random_scores) > 0 and len(genuine_scores) > 0:
            # Mann-Whitney U检验
            try:
                mannwhitney_stat, mannwhitney_p = mannwhitneyu(
                    genuine_scores, random_scores, alternative='two-sided'
                )
            except:
                mannwhitney_stat, mannwhitney_p = 0, 1.0
            
            # Check if metric is "lower is better" type
            metric_info = self.metric_registry.metrics.get(metric_name, {})
            lower_is_better = metric_info.get("lower_is_better", False)
            
            # 效应量计算 (Cohen's d) - 针对不同指标类型调整
            random_mean = np.mean(random_scores)
            random_std = np.std(random_scores)
            genuine_mean = np.mean(genuine_scores)
            genuine_std = np.std(genuine_scores)
            
            pooled_std = np.sqrt(((len(random_scores)-1)*random_std**2 + 
                                (len(genuine_scores)-1)*genuine_std**2) / 
                               (len(random_scores) + len(genuine_scores) - 2))
            
            if lower_is_better:
                # 对于"越小越好"的指标，效应量应该是 (random - genuine) / pooled_std
                effect_size = (random_mean - genuine_mean) / pooled_std if pooled_std > 0 else 0
                print(f"    ℹ️  {metric_name} 是越小越好的指标，调整效应量计算")
            else:
                # 对于"越大越好"的指标，效应量是 (genuine - random) / pooled_std
                effect_size = (genuine_mean - random_mean) / pooled_std if pooled_std > 0 else 0
            
            # AUC分析 - 针对不同指标类型调整
            try:
                y_true = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(random_scores))])
                if lower_is_better:
                    # 对于"越小越好"的指标，取负值使得小分数对应高预测概率
                    y_scores = np.concatenate([-np.array(genuine_scores), -np.array(random_scores)])
                else:
                    # 对于"越大越好"的指标，直接使用原分数
                    y_scores = np.concatenate([genuine_scores, random_scores])
                auc_score = roc_auc_score(y_true, y_scores)
            except:
                auc_score = 0.5
            
            # 单调性检验 (使用质量梯度分数) - 针对不同指标类型调整
            try:
                if len(quality_gradient_scores) > 1:
                    # 质量级别：从0.95到0.0递减（排除完美预测1.0）
                    quality_levels = np.linspace(0.95, 0.0, len(quality_gradient_scores))
                    if lower_is_better:
                        # 对于"越小越好"的指标，期望负相关（质量高→分数小）
                        spearman_rho, spearman_p = spearmanr(quality_levels, quality_gradient_scores)
                        spearman_rho = -spearman_rho  # 转换为正值表示良好的单调性
                    else:
                        # 对于"越大越好"的指标，期望正相关（质量高→分数大）
                        spearman_rho, spearman_p = spearmanr(quality_levels, quality_gradient_scores)
                else:
                    # 回退到简单检验
                    spearman_rho, spearman_p = spearmanr(gt_labels, [genuine_mean] * len(gt_labels))
            except:
                spearman_rho, spearman_p = 0, 1.0
        else:
            mannwhitney_stat = mannwhitney_p = 0
            effect_size = auc_score = 0
            spearman_rho = spearman_p = 0
            random_mean = random_std = genuine_mean = genuine_std = 0
            lower_is_better = False
        
        # 判定逻辑（基于汇总统计）- 改进Oracle评估
        pass_verdict, failure_reasons = self._make_pass_verdict(
            mannwhitney_p, effect_size, auc_score,
            oracle_scores,  # 传入完整的oracle_scores列表
            genuine_scores,  # 传入完整的genuine_scores列表
            random_scores,   # 传入完整的random_scores列表
            random_mean / (random_mean + genuine_mean) if (random_mean + genuine_mean) > 0 else 0,
            spearman_rho,
            lower_is_better  # 传入指标类型
        )
        
        # Calculate independent effect size, AUC and judgment for each random trial (new)
        per_trial_effect_sizes = []
        per_trial_aucs = []
        per_trial_verdicts = []
        
        if genuine_scores and random_scores:
            # Get metric type
            metric_info = self.metric_registry.metrics.get(metric_name, {})
            lower_is_better = metric_info.get("lower_is_better", False)
            
            # 假设random_scores按顺序存储每次随机试验的分数
            for random_score in random_scores:
                # Calculate the effect size for this random trial (genuine vs single random)
                if lower_is_better:
                    trial_effect = (random_score - genuine_mean) / genuine_std if genuine_std > 0 else 0
                else:
                    trial_effect = (genuine_mean - random_score) / genuine_std if genuine_std > 0 else 0
                per_trial_effect_sizes.append(trial_effect)
                
                # Calculate the AUC for this random trial
                try:
                    y_true = np.concatenate([np.ones(len(genuine_scores)), np.zeros(1)])
                    if lower_is_better:
                        y_scores = np.concatenate([-np.array(genuine_scores), -np.array([random_score])])
                    else:
                        y_scores = np.concatenate([genuine_scores, [random_score]])
                    trial_auc = roc_auc_score(y_true, y_scores)
                except:
                    trial_auc = 0.5
                per_trial_aucs.append(trial_auc)
                
                # 为该随机试验做判定
                trial_verdict = "fail"
                if trial_effect >= self.config.effect_size_threshold and trial_auc >= self.config.auc_threshold:
                    trial_verdict = "pass"
                elif trial_effect >= self.config.effect_size_threshold * 0.5 or trial_auc >= self.config.auc_threshold * 0.9:
                    trial_verdict = "weak"
                per_trial_verdicts.append(trial_verdict)
        
        return MetricResult(
            metric_name=metric_name,
            genuine_scores=genuine_scores,
            random_scores=random_scores,
            oracle_scores=oracle_scores,
            mannwhitney_statistic=mannwhitney_stat,
            mannwhitney_pvalue=mannwhitney_p,
            effect_size=effect_size,
            auc_score=auc_score,
            spearman_rho=spearman_rho,
            spearman_pvalue=spearman_p,
            random_mean=random_mean,
            random_std=random_std,
            genuine_mean=genuine_mean,
            genuine_std=genuine_std,
            pass_verdict=pass_verdict,
            failure_reasons=failure_reasons,
            per_trial_verdicts=per_trial_verdicts,
            per_trial_effect_sizes=per_trial_effect_sizes,
            per_trial_aucs=per_trial_aucs
        )
    
    def _make_pass_verdict(self, mannwhitney_pvalue: float, effect_size: float, 
                          auc_score: float, oracle_scores: List[float],
                          genuine_scores: List[float], random_scores: List[float],
                          false_positive_rate: float, spearman_rho: float,
                          lower_is_better: bool = False) -> Tuple[str, List[str]]:
        """判定指标是否通过
        
        【改进】Oracle攻击评估：
        - 旧方法：oracle_mean >= 0.5 （不科学，不同指标范围不同）
        - 新方法：Oracle vs Genuine相对位置（适用所有指标）
        """
        
        # Calculate Oracle's relative threat level (Oracle's position between Random and Genuine)
        oracle_vulnerability = 0.0  # 默认值：无威胁
        
        if oracle_scores and genuine_scores and random_scores:
            oracle_mean = np.mean(oracle_scores)
            genuine_mean = np.mean(genuine_scores)
            random_mean = np.mean(random_scores)
            
            # 计算Oracle在Random→Genuine光谱上的位置
            # 返回值范围 [0, 1+]：
            #   0: Oracle = Random（攻击无效）
            #   0.5: Oracle介于中间（部分有效）
            #   1.0: Oracle = Genuine（完全攻破）
            #   >1.0: Oracle > Genuine（超越真实模型！严重问题）
            
            if lower_is_better:
                # 越小越好：Random(大) → Oracle → Genuine(小)
                denominator = random_mean - genuine_mean
                if abs(denominator) > 1e-10:
                    oracle_vulnerability = (random_mean - oracle_mean) / denominator
                else:
                    oracle_vulnerability = 0.0
            else:
                # 越大越好：Random(小) → Oracle → Genuine(大)
                denominator = genuine_mean - random_mean
                if abs(denominator) > 1e-10:
                    oracle_vulnerability = (oracle_mean - random_mean) / denominator
                else:
                    oracle_vulnerability = 0.0
            
            # 限制在合理范围（防止异常值）
            oracle_vulnerability = np.clip(oracle_vulnerability, -0.5, 2.0)
        
        # 通过条件（改进后的Oracle判定）
        pass_conditions = [
            effect_size >= self.config.effect_size_threshold,
            auc_score >= self.config.auc_threshold,
            false_positive_rate < self.config.false_positive_threshold,
            spearman_rho >= self.config.spearman_threshold,
            mannwhitney_pvalue < self.config.alpha,
            oracle_vulnerability < 0.7  # Oracle应该明显弱于Genuine（不到70%）
        ]
        
        # 失败原因
        reasons = []
        
        if effect_size < self.config.effect_size_threshold:
            reasons.append(f"Low effect size: {effect_size:.3f}")
        
        if auc_score < self.config.auc_threshold:
            reasons.append(f"Low AUC: {auc_score:.3f}")
        
        if false_positive_rate >= self.config.false_positive_threshold:
            reasons.append(f"High false positive rate: {false_positive_rate:.3f}")
        
        if spearman_rho <= self.config.spearman_threshold:
            reasons.append(f"Low monotonicity: {spearman_rho:.3f}")
        
        if mannwhitney_pvalue >= self.config.alpha:
            reasons.append(f"Non-significant difference: p={mannwhitney_pvalue:.3f}")
        
        if oracle_vulnerability >= 0.7:
            reasons.append(f"Oracle attack too effective: {oracle_vulnerability:.1%} towards genuine")
        
        # 判定逻辑
        n_pass = sum(pass_conditions)
        
        if n_pass >= 4:
            return "pass", []
        elif n_pass >= 2:
            return "weak", reasons
        else:
            return "fail", reasons
    
    def _compute_summary_statistics(self) -> Dict[str, Any]:
        """计算汇总统计"""
        # Get actually used metric names (consistent with runtime)
        performance_mode = self.config.get_performance_mode()
        actual_metric_names = self.metric_registry.get_metric_names(performance_mode)
        
        summary = {
            "total_datasets": len(self.results),
            "total_metrics": len(actual_metric_names),
            "metric_performance": {},
            "dataset_performance": {}
        }
        
        # 收集所有p值进行多重比较校正
        all_pvalues = []
        metric_dataset_pairs = []
        
        for metric_name in actual_metric_names:
            for dataset_id, dataset_result in self.results.items():
                if metric_name in dataset_result.metric_results:
                    result = dataset_result.metric_results[metric_name]
                    all_pvalues.append(result.mannwhitney_pvalue)
                    metric_dataset_pairs.append((metric_name, dataset_id))
        
        # 应用Benjamini-Hochberg多重比较校正
        if all_pvalues:
            rejected, corrected_pvalues, _, _ = multipletests(
                all_pvalues, alpha=self.config.alpha, method='fdr_bh'
            )
            
            # 更新校正后的p值到结果中
            for i, (metric_name, dataset_id) in enumerate(metric_dataset_pairs):
                if metric_name in self.results[dataset_id].metric_results:
                    self.results[dataset_id].metric_results[metric_name].mannwhitney_pvalue = corrected_pvalues[i]
                    # 根据校正后的p值重新判定
                    old_verdict = self.results[dataset_id].metric_results[metric_name].pass_verdict
                    result = self.results[dataset_id].metric_results[metric_name]
                    
                    # Get metric type information
                    metric_info = self.metric_registry.metrics.get(metric_name, {})
                    lower_is_better = metric_info.get("lower_is_better", False)
                    
                    new_verdict, new_reasons = self._make_pass_verdict(
                        corrected_pvalues[i], result.effect_size, result.auc_score,
                        result.oracle_scores,  # 传入完整的oracle_scores列表
                        result.genuine_scores,  # 传入完整的genuine_scores列表
                        result.random_scores,   # 传入完整的random_scores列表
                        result.random_mean / (result.random_mean + result.genuine_mean) if (result.random_mean + result.genuine_mean) > 0 else 0,
                        result.spearman_rho,
                        lower_is_better  # 传入指标类型
                    )
                    self.results[dataset_id].metric_results[metric_name].pass_verdict = new_verdict
                    self.results[dataset_id].metric_results[metric_name].failure_reasons = new_reasons
            
            print(f"✓ Applied multiple comparison correction: {len(all_pvalues)} p-values, {np.sum(rejected)} significant")
        
        # Summarize by metric (using corrected results) - only process actually run metrics
        # [Fixed]: Changed to statistics per dataset, not per random trial
        for metric_name in actual_metric_names:
            metric_results = []
            
            for dataset_result in self.results.values():
                if metric_name in dataset_result.metric_results:
                    result = dataset_result.metric_results[metric_name]
                    metric_results.append(result)
            
            if metric_results:
                # 【关键修复】：基于数据集级别的判定计算pass/weak/fail rate
                # 每个数据集一个判定结果（pass_verdict）
                pass_count = sum(1 for r in metric_results if r.pass_verdict == "pass")
                weak_count = sum(1 for r in metric_results if r.pass_verdict == "weak")
                fail_count = sum(1 for r in metric_results if r.pass_verdict == "fail")
                total_datasets = len(metric_results)
                
                summary["metric_performance"][metric_name] = {
                    "pass_rate": pass_count / total_datasets,
                    "weak_rate": weak_count / total_datasets,
                    "fail_rate": fail_count / total_datasets,
                    "total_datasets": total_datasets,  # 改为：总数据集数
                    "avg_effect_size": np.mean([r.effect_size for r in metric_results]),
                    "avg_auc": np.mean([r.auc_score for r in metric_results]),
                    "avg_random_score": np.mean([r.random_mean for r in metric_results]),
                    "avg_genuine_score": np.mean([r.genuine_mean for r in metric_results]),
                    "avg_spearman_rho": np.mean([r.spearman_rho for r in metric_results])
                }
        
        # Debug output: print some metric dataset statistics
        if actual_metric_names and actual_metric_names[0] in summary["metric_performance"]:
            first_metric = actual_metric_names[0]
            perf = summary["metric_performance"][first_metric]
            print(f"    Debug info: Metric '{first_metric}' performance on {perf['total_datasets']} datasets")
            print(f"      Pass: {perf['pass_rate']:.1%}, Weak: {perf['weak_rate']:.1%}, Fail: {perf['fail_rate']:.1%}")
        
        return summary
    
    def _save_raw_experiment_data(self):
        """Save raw experimental data (for subsequent independent plotting)"""
        from data_utils import save_experiment_data
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        print(f"  Saving experimental data to: {self.output_dir}")
        
        # If there are multiple runs, save all raw data
        if hasattr(self, 'all_run_results') and self.all_run_results:
            # Save raw data from all runs
            saved_files = save_experiment_data(
                results=self.results,
                summary_stats=self.summary_stats,
                config=self.config,
                output_dir=self.output_dir,
                timestamp=timestamp,
                all_runs=self.all_run_results  # Additionally save data from all runs
            )
            print(f"  ✓ Experiment data saved ({len(saved_files)} files)")
            print(f"    - Contains raw data from {len(self.all_run_results)} independent runs")
        else:
            # Single run
            saved_files = save_experiment_data(
                results=self.results,
                summary_stats=self.summary_stats,
                config=self.config,
                output_dir=self.output_dir,
                timestamp=timestamp
            )
            print(f"  ✓ Experiment data saved ({len(saved_files)} files)")
        
        print(f"    - Can use plot_results.py to regenerate plots")
    
    def _generate_experiment_report(self):
        """Generate experiment report"""
        
        # Create report directory
        report_dir = self.output_dir / "reports"
        report_dir.mkdir(exist_ok=True)
        
        # Save raw results
        if self.config.save_data:
            results_file = self.output_dir / f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Serialize results to JSON
            serializable_results = {}
            for dataset_id, dataset_result in self.results.items():
                serializable_results[dataset_id] = {
                    "dataset_id": dataset_result.dataset_id,
                    "dataset_config": dataset_result.dataset_config,
                    "metric_results": {
                        metric_name: asdict(metric_result)
                        for metric_name, metric_result in dataset_result.metric_results.items()
                    }
                }
            
            with open(results_file, 'w') as f:
                json.dump({
                    "config": asdict(self.config),
                    "results": serializable_results,
                    "summary": self.summary_stats
                }, f, indent=2, default=str)
            
            print(f"✓ Results saved: {results_file}")
        
        # Generate summary table
        self._generate_summary_table(report_dir)
        
        # Generate visualization plots
        if self.config.save_plots:
            self._generate_visualization_plots(report_dir)
        
        print(f"✓ Report generated: {report_dir}")
    
    def _generate_summary_table(self, report_dir: Path):
        """Generate summary table"""
        # Create metric summary table - use actually saved metrics (no longer filtered by performance mode)
        metric_summary = []
        # 直接使用summary_stats中实际存在的指标，避免性能模式导致的过滤
        actual_metric_names = list(self.summary_stats["metric_performance"].keys())
        
        for metric_name in actual_metric_names:
            if metric_name in self.summary_stats["metric_performance"]:
                perf = self.summary_stats["metric_performance"][metric_name]
                metric_summary.append({
                    "Metric": metric_name,
                    "Pass Rate": f"{perf['pass_rate']:.1%}",
                    "Weak Rate": f"{perf['weak_rate']:.1%}",
                    "Fail Rate": f"{perf['fail_rate']:.1%}",
                    "Avg Effect Size": f"{perf['avg_effect_size']:.3f}",
                    "Avg AUC": f"{perf['avg_auc']:.3f}",
                    "Avg Genuine Score": f"{perf['avg_genuine_score']:.3f}",  # 真实模型平均分数
                    "Avg Random Score": f"{perf['avg_random_score']:.3f}",
                    "Monotonicity (ρ)": f"{perf['avg_spearman_rho']:.3f}"  # 单调性表征
                })
        
        df_summary = pd.DataFrame(metric_summary)
        
        # Save as CSV
        summary_file = report_dir / "metric_summary.csv"
        df_summary.to_csv(summary_file, index=False)
        print(f"✓ Summary table saved: {summary_file}")
        
        # Print to console
        print("\n=== Metric Performance Summary ===")
        print(df_summary.to_string(index=False))
    
    def _generate_visualization_plots(self, report_dir: Path):
        """Generate visualization charts (including both original and abbreviated versions)"""
        
        # 设置绘图风格和字体
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 设置全局字体为 Times New Roman（学术风格）
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
        plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式也使用Times风格
        
        # 增大全局字体大小
        plt.rcParams['font.size'] = 12          # 基础字体大小
        plt.rcParams['axes.titlesize'] = 16     # 标题字体大小
        plt.rcParams['axes.labelsize'] = 14     # 坐标轴标签字体大小
        plt.rcParams['xtick.labelsize'] = 11    # x轴刻度字体大小
        plt.rcParams['ytick.labelsize'] = 11    # y轴刻度字体大小
        plt.rcParams['legend.fontsize'] = 11    # 图例字体大小
        plt.rcParams['legend.title_fontsize'] = 12  # 图例标题字体大小
        
        print("\nGenerating visualization plots...")
        
        # Generate original version (without abbreviations)
        print("\n[Generating original version plots]")
        self._plot_pass_rates(report_dir, use_abbreviation=False)
        self._plot_score_distribution(report_dir, use_abbreviation=False)
        self._plot_violin_distribution(report_dir, use_abbreviation=False)
        self._plot_auc_vs_effect_size(report_dir, use_abbreviation=False)
        self._plot_metric_dataset_heatmap(report_dir, use_abbreviation=False)
        
        # Generate abbreviated version (using LaTeX abbreviations)
        print("\n[Generating abbreviated version plots]")
        self._plot_pass_rates(report_dir, use_abbreviation=True)
        self._plot_score_distribution(report_dir, use_abbreviation=True)
        self._plot_violin_distribution(report_dir, use_abbreviation=True)
        self._plot_auc_vs_effect_size(report_dir, use_abbreviation=True)
        self._plot_metric_dataset_heatmap(report_dir, use_abbreviation=True)
        
        print(f"\n✓ Visualization plots saved: {report_dir}")
        print("  - Original version uses full metric names")
        print("  - Abbreviated version uses LaTeX format academic abbreviations")
    
    def _compute_composite_ranking(self) -> List[str]:
        """计算指标的综合排名（效应量60% + AUC40%）
        
        Returns:
            List[str]: 按综合分数排序的指标名称列表（从高到低）
        """
        # Use actually saved metrics to avoid filtering caused by performance mode
        actual_metric_names = list(self.summary_stats["metric_performance"].keys())
        
        # 过滤掉简化版的pointwise_f1（保留complex版本）
        actual_metric_names = [m for m in actual_metric_names if m != 'pointwise_f1']
        
        # 收集每个指标的效应量和AUC
        metric_summary = {}
        for metric_name in actual_metric_names:
            effect_sizes = []
            auc_scores = []
            
            for dataset_id, dataset_result in self.results.items():
                if metric_name in dataset_result.metric_results:
                    result = dataset_result.metric_results[metric_name]
                    if not np.isnan(result.effect_size) and not np.isnan(result.auc_score):
                        effect_sizes.append(result.effect_size)
                        auc_scores.append(result.auc_score)
            
            if effect_sizes and auc_scores:
                metric_summary[metric_name] = {
                    'mean_effect_size': np.mean(effect_sizes),
                    'mean_auc': np.mean(auc_scores)
                }
        
        if not metric_summary:
            return actual_metric_names  # 回退到默认顺序
        
        # 计算综合排名
        metrics_with_scores = []
        for metric_name, values in metric_summary.items():
            metrics_with_scores.append({
                'metric': metric_name,
                'mean_effect_size': values['mean_effect_size'],
                'mean_auc': values['mean_auc']
            })
        
        # 按效应量排名
        sorted_by_effect = sorted(metrics_with_scores, key=lambda x: x['mean_effect_size'], reverse=True)
        for rank, item in enumerate(sorted_by_effect):
            item['effect_rank'] = rank
        
        # 按AUC排名
        sorted_by_auc = sorted(metrics_with_scores, key=lambda x: x['mean_auc'], reverse=True)
        for rank, item in enumerate(sorted_by_auc):
            item['auc_rank'] = rank
        
        # 计算综合分数（权重：效应量60%，AUC40%）
        n_metrics = len(metrics_with_scores)
        for item in metrics_with_scores:
            norm_effect = (n_metrics - item['effect_rank']) / n_metrics
            norm_auc = (n_metrics - item['auc_rank']) / n_metrics
            item['composite_score'] = 0.6 * norm_effect + 0.4 * norm_auc
        
        # 按综合分数排序（从高到低）
        metrics_with_scores.sort(key=lambda x: x['composite_score'], reverse=True)
        
        return [item['metric'] for item in metrics_with_scores]
    
    def _plot_pass_rates(self, report_dir: Path, use_abbreviation: bool = False):
        """绘制通过率条形图"""
        
        metrics = []
        pass_rates = []
        weak_rates = []
        fail_rates = []
        
        # Use actually saved metrics to avoid filtering caused by performance mode
        actual_metric_names = list(self.summary_stats["metric_performance"].keys())
        
        for metric_name in actual_metric_names:
            # Skip simplified version of pointwise_f1 (keep complex version)
            if metric_name == 'pointwise_f1':
                continue
                
            if metric_name in self.summary_stats["metric_performance"]:
                perf = self.summary_stats["metric_performance"][metric_name]
                # 使用显示名称（根据use_abbreviation参数）
                display_name = self.metric_registry.get_display_name(metric_name, use_abbreviation)
                metrics.append(display_name)
                pass_rates.append(perf["pass_rate"])
                weak_rates.append(perf["weak_rate"])
                fail_rates.append(perf["fail_rate"])
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(14, 7))
        
        x = np.arange(len(metrics))
        width = 0.8
        
        p1 = ax.bar(x, pass_rates, width, label='Pass', color='green', alpha=0.7)
        p2 = ax.bar(x, weak_rates, width, bottom=pass_rates, label='Weak', color='orange', alpha=0.7)
        p3 = ax.bar(x, fail_rates, width, bottom=np.array(pass_rates) + np.array(weak_rates), 
                   label='Fail', color='red', alpha=0.7)
        
        ax.set_xlabel('Metrics', fontsize=14, fontweight='bold')
        ax.set_ylabel('Rate', fontsize=14, fontweight='bold')
        ax.set_title('Metric Pass/Weak/Fail Rates', fontsize=16, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right', fontsize=11)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 根据版本选择文件名
        filename = "metric_pass_rates_abbr.png" if use_abbreviation else "metric_pass_rates.png"
        plt.savefig(report_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        version_str = "abbreviated version" if use_abbreviation else "original version"
        print(f"    ✓ Pass rate chart ({version_str}) saved: {filename}")
    
    def _plot_score_distribution(self, report_dir: Path, use_abbreviation: bool = False):
        """绘制分数分布箱线图（自适应图表类型，按综合排名排序）"""
        
        # 计算综合排名
        sorted_metric_names = self._compute_composite_ranking()
        print(f"    使用综合排名顺序（前3名: {', '.join(sorted_metric_names[:3])}）")
        
        data = []
        
        # 使用排序后的指标名称收集数据
        for metric_name in sorted_metric_names:
            # Skip simplified version of pointwise_f1 (keep complex version)
            if metric_name == 'pointwise_f1':
                continue
            # Get display names (data points use regular font to avoid visual conflict with rotated labels)
            display_name = self.metric_registry.get_display_name(metric_name, use_abbreviation, use_upright=True)
            
            for dataset_id, dataset_result in self.results.items():
                if metric_name in dataset_result.metric_results:
                    result = dataset_result.metric_results[metric_name]
                    
                    # 添加所有真实检测器的分数（绿色）
                    for score in result.genuine_scores:
                        if not np.isnan(score):
                            data.append({
                                'Metric': display_name,
                                'OriginalMetric': metric_name,  # 保留原始名称用于归一化
                                'Score': score,
                                'Type': 'Genuine (Real Model)',
                                'Category': 'pass'  # 真实模型预期应该通过
                            })
                    
                    # 添加所有随机预测的分数（红色）
                    for score in result.random_scores:
                        if not np.isnan(score):
                            data.append({
                                'Metric': display_name,
                                'OriginalMetric': metric_name,
                                'Score': score,
                                'Type': 'Random Prediction',
                                'Category': 'fail'  # 随机预测预期应该失败
                            })
                    
                    # 添加对抗预测的分数（橙色）
                    for score in result.oracle_scores:
                        if not np.isnan(score):
                            data.append({
                                'Metric': display_name,
                                'OriginalMetric': metric_name,
                                'Score': score,
                                'Type': 'Oracle Attack',
                                'Category': 'weak'  # 对抗攻击是中间状态
                            })
        
        if not data:
            print("    Warning: No score data, skipping plot")
            return
        
        df = pd.DataFrame(data)
        
        # 关键修复：归一化特殊指标的分数到[0,1]范围（使用原始名称）
        print(f"    Checking and normalizing special metrics...")
        
        for original_metric_name in df['OriginalMetric'].unique():
            metric_data = df[df['OriginalMetric'] == original_metric_name]
            
            # NAB分数：范围(-∞, 100]，归一化到[0,1]
            if original_metric_name == 'nab_score':
                scores = metric_data['Score'].values
                # NAB: 0分=基线，100分=完美，负分=有害
                # 映射策略：将[-100, 100]映射到[0, 1]，更负的分数映射到接近0
                normalized = []
                for s in scores:
                    if np.isnan(s) or np.isinf(s):
                        normalized.append(np.nan)
                    else:
                        # 将分数限制在合理范围，然后线性映射
                        s_clipped = np.clip(s, -100, 100)
                        normalized.append((s_clipped + 100) / 200)  # [-100,100] -> [0,1]
                
                df.loc[df['OriginalMetric'] == original_metric_name, 'Score'] = normalized
                print(f"      ✓ 归一化 {original_metric_name}: 原始范围[-100,100] -> [0,1]")
                print(f"        原始: min={np.nanmin(scores):.2f}, max={np.nanmax(scores):.2f}")
                print(f"        归一化后: min={np.nanmin(normalized):.3f}, max={np.nanmax(normalized):.3f}")
            
            # Temporal Distance：越小越好，需要反转并归一化
            elif original_metric_name == 'temporal_distance':
                scores = metric_data['Score'].values
                valid_scores = scores[~np.isnan(scores)]
                
                if len(valid_scores) > 0:
                    min_score = np.min(valid_scores)
                    max_score = np.max(valid_scores)
                    
                    if max_score > min_score:
                        # 反转：距离越小越好 -> 分数越高越好
                        # 然后归一化到[0,1]
                        normalized = 1 - (scores - min_score) / (max_score - min_score)
                        df.loc[df['OriginalMetric'] == original_metric_name, 'Score'] = normalized
                        print(f"      ✓ 归一化 {original_metric_name}: 反转并归一化到[0,1]")
                        print(f"        原始: min={min_score:.2f}, max={max_score:.2f}")
                        print(f"        归一化后: min={np.nanmin(normalized):.3f}, max={np.nanmax(normalized):.3f}")
        
        # 统计每个指标的数据点数量
        points_per_metric = df.groupby('Metric').size()
        avg_points = points_per_metric.mean()
        total_points = len(df)
        
        print(f"    分数分布数据: {total_points}个数据点, 平均每指标{avg_points:.1f}个")
        print(f"      - 真实模型分数: {len(df[df['Type']=='Genuine (Real Model)'])}个")
        print(f"      - 随机预测分数: {len(df[df['Type']=='Random Prediction'])}个")
        print(f"      - 对抗攻击分数: {len(df[df['Type']=='Oracle Attack'])}个")
        
        # 设置指标顺序为综合排名顺序（使用显示名称，横轴旋转使用正体）
        display_names_sorted = [self.metric_registry.get_display_name(m, use_abbreviation, use_upright=True) 
                               for m in sorted_metric_names]
        df['Metric'] = pd.Categorical(df['Metric'], categories=display_names_sorted, ordered=True)
        
        fig, ax = plt.subplots(figsize=(18, 9))
        
        # 自适应选择图表类型
        # 关键修复：即使数据点较少，也应该能看到分布
        if avg_points < 10:
            # 数据点很少，使用大散点
            print(f"    使用大散点图模式（数据点很少: {avg_points:.1f}/指标）")
            
            # 使用更大的散点，更强的对比
            sns.stripplot(data=df, x='Metric', y='Score', 
                         hue='Type', ax=ax, 
                         palette={
                             'Genuine (Real Model)': '#2ecc71',  # 亮绿色
                             'Random Prediction': '#e74c3c',      # 亮红色
                             'Oracle Attack': '#f39c12'           # 亮橙色
                         },
                         dodge=True, alpha=0.8, size=10, jitter=0.4,
                         edgecolor='black', linewidth=0.5)
            
            title = f'Score Distribution by Metric - Sorted by Composite Score'
            
        else:
            # 数据点适中/充足，使用散点+箱线图组合（最好看）
            print(f"    使用散点+箱线图模式（数据点: {avg_points:.1f}/指标）")

            # 第二步：再画箱线图（上层，半透明）
            # 注意：这里不传hue，而是手动为每组设置颜色
            sns.boxplot(data=df, x='Metric', y='Score',
                       hue='Type', ax=ax,
                       palette={
                           'Genuine (Real Model)': '#2ecc71',
                           'Random Prediction': '#e74c3c',
                           'Oracle Attack': '#f39c12'
                       },
                       showfliers=True,  # 显示离群点
                       width=0.5,         # 箱线图宽度
                       linewidth=1,     # 线条粗细
                       medianprops=dict(color='black', linewidth=1.4, alpha=0.8))  # 中位线加粗
            
            title = f'Score Distribution by Metric - Sorted by Composite Score'
        
        ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
        ax.set_xlabel('Metric', fontsize=13, fontweight='bold')
        ax.set_ylabel('Metric Score', fontsize=13, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        ax.set_ylim(-0.05, 1.05)  # 确保0-1范围可见
        
        # 优化图例
        handles, labels = ax.get_legend_handles_labels()
        # 去除重复的图例（stripplot会产生重复）
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), 
                 loc='upper right', framealpha=0.95, fontsize=10,
                 title='Prediction Type', title_fontsize=11)
        
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # 添加0.5参考线（很多指标的期望随机水平）
        ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Random Baseline (0.5)')
        
        plt.tight_layout()
        
        # Save high-resolution images (select filename based on version)
        filename = "score_distribution_abbr.png" if use_abbreviation else "score_distribution.png"
        output_file = report_dir / filename
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        version_str = "abbreviated version" if use_abbreviation else "original version"
        print(f"    ✓ Score distribution chart ({version_str}) saved: {output_file}")
        print(f"    ✓ Chart contains {total_points} data points, average {avg_points:.1f} per metric")
    
    def _plot_violin_distribution(self, report_dir: Path, use_abbreviation: bool = False):
        """绘制纯小提琴图版本（用于对比可视化效果，按综合排名排序）"""
        
        print("  Generating violin plot version...")
        
        # 计算综合排名（与分数分布图保持一致）
        sorted_metric_names = self._compute_composite_ranking()
        
        data = []
        
        # 使用排序后的指标名称收集数据
        for metric_name in sorted_metric_names:
            # Skip simplified version of pointwise_f1 (keep complex version)
            if metric_name == 'pointwise_f1':
                continue
            # Get display names (data points use regular font to avoid visual conflict with rotated labels)
            display_name = self.metric_registry.get_display_name(metric_name, use_abbreviation, use_upright=True)
            
            for dataset_id, dataset_result in self.results.items():
                if metric_name in dataset_result.metric_results:
                    result = dataset_result.metric_results[metric_name]
                    
                    # 添加所有真实检测器的分数
                    for score in result.genuine_scores:
                        if not np.isnan(score):
                            data.append({
                                'Metric': display_name,
                                'OriginalMetric': metric_name,  # 保留原始名称用于归一化
                                'Score': score,
                                'Type': 'Genuine (Real Model)',
                                'Category': 'pass'
                            })
                    
                    # 添加所有随机预测的分数
                    for score in result.random_scores:
                        if not np.isnan(score):
                            data.append({
                                'Metric': display_name,
                                'OriginalMetric': metric_name,
                                'Score': score,
                                'Type': 'Random Prediction',
                                'Category': 'fail'
                            })
                    
                    # 添加对抗预测的分数
                    for score in result.oracle_scores:
                        if not np.isnan(score):
                            data.append({
                                'Metric': display_name,
                                'OriginalMetric': metric_name,
                                'Score': score,
                                'Type': 'Oracle Attack',
                                'Category': 'weak'
                            })
        
        if not data:
            print("    Warning: No score data, skipping violin plot")
            return
        
        df = pd.DataFrame(data)
        
        # 应用相同的归一化逻辑（使用OriginalMetric）
        for original_metric_name in df['OriginalMetric'].unique():
            metric_data = df[df['OriginalMetric'] == original_metric_name]
            
            # NAB分数归一化
            if original_metric_name == 'nab_score':
                scores = metric_data['Score'].values
                normalized = []
                for s in scores:
                    if np.isnan(s) or np.isinf(s):
                        normalized.append(np.nan)
                    else:
                        s_clipped = np.clip(s, -100, 100)
                        normalized.append((s_clipped + 100) / 200)
                df.loc[df['OriginalMetric'] == original_metric_name, 'Score'] = normalized
            
            # Temporal Distance归一化
            elif original_metric_name == 'temporal_distance':
                scores = metric_data['Score'].values
                valid_scores = scores[~np.isnan(scores)]
                if len(valid_scores) > 0:
                    min_score = np.min(valid_scores)
                    max_score = np.max(valid_scores)
                    if max_score > min_score:
                        normalized = 1 - (scores - min_score) / (max_score - min_score)
                        df.loc[df['OriginalMetric'] == original_metric_name, 'Score'] = normalized
        
        total_points = len(df)
        avg_points = df.groupby('Metric').size().mean()
        
        # 设置指标顺序为综合排名顺序（使用显示名称，横轴旋转使用正体）
        display_names_sorted = [self.metric_registry.get_display_name(m, use_abbreviation, use_upright=True) 
                               for m in sorted_metric_names]
        df['Metric'] = pd.Categorical(df['Metric'], categories=display_names_sorted, ordered=True)
        
        # Create violin plot
        fig, ax = plt.subplots(figsize=(18, 9))
        
        print(f"    绘制小提琴图（{total_points}个数据点，平均{avg_points:.1f}个/指标）")
        
        # 纯小提琴图，内部不显示箱线图
        sns.violinplot(data=df, x='Metric', y='Score', 
                      hue='Type', ax=ax, 
                      palette={
                          'Genuine (Real Model)': '#2ecc71',
                          'Random Prediction': '#e74c3c',
                          'Oracle Attack': '#f39c12'
                      },
                      split=False, inner='quartile', cut=0, 
                      linewidth=1.5, alpha=0.7)
        
        # 标题和标签
        ax.set_title(f'Score Distribution by Metric - Violin Plot (Sorted by Composite Score)\n({total_points} data points, {avg_points:.1f} per metric)', 
                    fontsize=15, fontweight='bold', pad=20)
        ax.set_xlabel('Metric', fontsize=13, fontweight='bold')
        ax.set_ylabel('Metric Score', fontsize=13, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        
        # 图例
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), 
                 loc='upper right', framealpha=0.95, fontsize=10,
                 title='Prediction Type', title_fontsize=11)
        
        # 网格和参考线
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        
        plt.tight_layout()
        
        # Save (select filename based on version)
        filename = "score_distribution_violin_abbr.png" if use_abbreviation else "score_distribution_violin.png"
        output_file = report_dir / filename
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        version_str = "abbreviated version" if use_abbreviation else "original version"
        print(f"    ✓ Violin plot ({version_str}) saved: {output_file}")
    
    def _plot_auc_vs_effect_size(self, report_dir: Path, use_abbreviation: bool = False):
        """绘制效应量和AUC的综合对比图（合并版，按综合排名排序）"""
        
        print("  Generating effect size and AUC comprehensive comparison chart...")
        
        # 收集每个指标在每个数据集每次随机试验上的效应量和AUC
        metric_summary = {}
        # Use actually saved metrics to avoid filtering caused by performance mode
        actual_metric_names = list(self.summary_stats["metric_performance"].keys())
        
        # Get the number of random trials for each dataset (assuming all datasets have the same number of trials)
        n_random_trials = self.config.n_random_trials
        
        for metric_name in actual_metric_names:
            # Skip simplified version of pointwise_f1 (keep complex version)
            if metric_name == 'pointwise_f1':
                continue
                
            effect_sizes = []  # 收集每个数据集的总体效应量（与CSV一致）
            auc_scores = []    # 收集每个数据集的总体AUC（与CSV一致）
            
            for dataset_id, dataset_result in self.results.items():
                if metric_name in dataset_result.metric_results:
                    result = dataset_result.metric_results[metric_name]
                    
                    # 【关键修复】使用已经计算好的总体效应量和AUC（与CSV表格一致）
                    # result.effect_size 是所有genuine_scores vs 所有random_scores的效应量
                    # 使用pooled_std计算，是正确的统计方法
                    if not np.isnan(result.effect_size) and not np.isnan(result.auc_score):
                        effect_sizes.append(result.effect_size)
                        auc_scores.append(result.auc_score)
            
            if effect_sizes and auc_scores:
                metric_summary[metric_name] = {
                    'effect_sizes': effect_sizes,  # 每个数据集一个总体效应量
                    'auc_scores': auc_scores,      # 每个数据集一个总体AUC  
                    'n_datasets': len(effect_sizes)
                }
        
        if actual_metric_names and actual_metric_names[0] in metric_summary:
            n_datasets = metric_summary[actual_metric_names[0]]['n_datasets']
            print(f"    调试信息: 指标'{actual_metric_names[0]}'收集到 {n_datasets} 个数据集的总体效应量和AUC")
        
        if not metric_summary:
            print("    Warning: No AUC/effect size data, skipping plot")
            return
        
        # 计算综合排名（使用归一化加权平均，基于均值）
        metrics_with_scores = []
        for metric_name, values in metric_summary.items():
            metrics_with_scores.append({
                'metric': metric_name,
                'mean_effect_size': np.mean(values['effect_sizes']),
                'mean_auc': np.mean(values['auc_scores']),
                'n_datasets': values['n_datasets']
            })
        
        # 按效应量均值排名
        sorted_by_effect = sorted(metrics_with_scores, key=lambda x: x['mean_effect_size'], reverse=True)
        for rank, item in enumerate(sorted_by_effect):
            item['effect_rank'] = rank
        
        # 按AUC均值排名
        sorted_by_auc = sorted(metrics_with_scores, key=lambda x: x['mean_auc'], reverse=True)
        for rank, item in enumerate(sorted_by_auc):
            item['auc_rank'] = rank
        
        # 计算综合分数（归一化加权平均，权重：效应量60%，AUC40%）
        n_metrics = len(metrics_with_scores)
        for item in metrics_with_scores:
            norm_effect = (n_metrics - item['effect_rank']) / n_metrics
            norm_auc = (n_metrics - item['auc_rank']) / n_metrics
            item['composite_score'] = 0.6 * norm_effect + 0.4 * norm_auc
        
        # 按综合分数排序（从高到低）
        metrics_with_scores.sort(key=lambda x: x['composite_score'], reverse=True)
        sorted_metric_names = [item['metric'] for item in metrics_with_scores]
        
        # Create display name list (use regular font for horizontal axis rotation)
        display_names = [self.metric_registry.get_display_name(m, use_abbreviation, use_upright=True) 
                        for m in sorted_metric_names]
        
        # 统计数据集数量
        actual_n_datasets = len(self.results)
        print(f"    处理 {n_metrics} 个指标，{actual_n_datasets} 个数据集")
        print(f"    每个指标有 {actual_n_datasets} 个数据点（每个数据集一个总体效应量/AUC）")
        
        # Create chart - use dual Y-axis: effect size with box plot (left axis), AUC with scatter plot (right axis)
        fig, ax1 = plt.subplots(figsize=(max(16, n_metrics * 0.8), 8))
        
        x = np.arange(len(sorted_metric_names))
        
        # === 左轴：效应量箱线图 ===
        ax1.set_xlabel('Metric (sorted by composite score)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Effect Size', fontsize=13, fontweight='bold', color='#3498db')
        ax1.tick_params(axis='y', labelcolor='#3498db')
        
        # 准备箱线图数据
        effect_size_data = [metric_summary[m]['effect_sizes'] for m in sorted_metric_names]
        
        # 绘制箱线图
        bp1 = ax1.boxplot(effect_size_data, positions=x, widths=0.6,
                          patch_artist=True,
                          boxprops=dict(facecolor='#3498db', alpha=0.6),
                          medianprops=dict(color='#2c3e50', linewidth=2),
                          whiskerprops=dict(color='#3498db', linewidth=1.5),
                          capprops=dict(color='#3498db', linewidth=1.5),
                          flierprops=dict(marker='o', markerfacecolor='#3498db', 
                                        markersize=4, alpha=0.5))
        
        # 添加效应量阈值线
        ax1.axhline(y=self.config.effect_size_threshold, color='#3498db', 
                   linestyle='--', linewidth=2, alpha=0.7, 
                   label=f'Effect Size Threshold = {self.config.effect_size_threshold}')
        
        # 设置左轴范围
        max_effect = max([max(vals) if vals else 0 for vals in effect_size_data])
        ax1.set_ylim(-5, max(max_effect * 1.1, self.config.effect_size_threshold * 1.3))
        ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # === 右轴：AUC散点图 ===
        ax2 = ax1.twinx()
        ax2.set_ylabel('AUC', fontsize=13, fontweight='bold', color='#e74c3c')
        ax2.tick_params(axis='y', labelcolor='#e74c3c')
        
        # 绘制AUC散点图（添加水平抖动以避免重叠）
        np.random.seed(42)  # 固定随机种子以确保可复现
        for i, metric_name in enumerate(sorted_metric_names):
            auc_values = metric_summary[metric_name]['auc_scores']
            # 添加小的水平抖动（范围: ±0.15）
            jitter = np.random.uniform(-0.15, 0.15, len(auc_values))
            x_positions = [i + j for j in jitter]
            ax2.scatter(x_positions, auc_values, 
                       color='#e74c3c', s=80, alpha=0.6, zorder=3,
                       edgecolors='#c0392b', linewidths=1)
        
        # 添加AUC阈值线
        ax2.axhline(y=self.config.auc_threshold, color='#e74c3c', 
                   linestyle='--', linewidth=2, alpha=0.7,
                   label=f'AUC Threshold = {self.config.auc_threshold}')
        
        # 设置右轴范围（降低上限以避免挡住图例）
        ax2.set_ylim(-0.05, 1.02)
        
        # 图表装饰
        ax1.set_title(f'Effect Size & AUC Comparison', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.set_xticks(x)
        ax1.set_xticklabels(display_names, rotation=45, ha='right', fontsize=10)
        
        # 合并两个轴的图例
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Patch(facecolor='#3498db', alpha=0.6, label='Effect Size (boxplot)'),
            Line2D([0], [0], color='#3498db', linewidth=2, linestyle='--', 
                   label=f'Effect Size Threshold = {self.config.effect_size_threshold}'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', 
                   markersize=8, alpha=0.6, label='AUC (scatter)'),
            Line2D([0], [0], color='#e74c3c', linewidth=2, linestyle='--',
                   label=f'AUC Threshold = {self.config.auc_threshold}')
        ]
        ax1.legend(handles=legend_elements, loc='lower left', fontsize=10, framealpha=0.95)
        
        plt.tight_layout()
        
        # Save (select filename based on version)
        filename = "effect_size_and_auc_comparison_abbr.png" if use_abbreviation else "effect_size_and_auc_comparison.png"
        output_file = report_dir / filename
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        version_str = "abbreviated version" if use_abbreviation else "original version"
        print(f"    ✓ Comprehensive comparison of effect size and AUC ({version_str}) saved: {output_file}")
        
        # Print ranking information
        top_3_names = [self.metric_registry.get_display_name(m, use_abbreviation) for m in sorted_metric_names[:3]]
        print(f"    ✓ Metrics sorted by comprehensive ranking (top 3: {', '.join(top_3_names)})")
    
    def _plot_metric_dataset_heatmap(self, report_dir: Path, use_abbreviation: bool = False):
        """绘制指标×数据集热图（AUC + 效应量）"""
        
        # Create matrix: metrics × datasets - use actually saved metrics
        # Use actually saved metrics to avoid filtering caused by performance mode
        metrics = list(self.summary_stats["metric_performance"].keys())
        
        # 过滤掉简化版的pointwise_f1（保留complex版本）
        metrics = [m for m in metrics if m != 'pointwise_f1']
        
        dataset_ids = list(self.results.keys())
        
        # Create display name list
        display_names = [self.metric_registry.get_display_name(m, use_abbreviation) 
                        for m in metrics]
        
        # Create short display labels (take only length and anomaly ratio)
        dataset_labels = []
        for ds_id in dataset_ids:
            config = self.results[ds_id].dataset_config
            length = config["length"]
            anomaly_ratio = config["anomaly_ratio"]
            dataset_labels.append(f"L{length}_R{int(anomaly_ratio)+1}%")
        
        # AUC矩阵和效应量矩阵
        auc_matrix = np.full((len(metrics), len(dataset_ids)), np.nan)
        effect_size_matrix = np.full((len(metrics), len(dataset_ids)), np.nan)
        
        for i, metric_name in enumerate(metrics):
            for j, dataset_id in enumerate(dataset_ids):
                if metric_name in self.results[dataset_id].metric_results:
                    result = self.results[dataset_id].metric_results[metric_name]
                    auc_matrix[i, j] = result.auc_score
                    effect_size_matrix[i, j] = result.effect_size
        
        # 绘制双热图：AUC + 效应量
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 11))
        
        # 左侧：AUC热图
        sns.heatmap(auc_matrix, 
                    xticklabels=dataset_labels, 
                    yticklabels=display_names,
                    annot=True, 
                    fmt='.3f',
                    cmap='YlGnBu',  # 黄-绿-蓝渐变，更学术
                    vmin=0.0, 
                    vmax=1.0,
                    ax=ax1,
                    cbar_kws={'label': 'AUC Score'},
                    linewidths=0.5,
                    linecolor='white')
        ax1.set_title('Metric AUC Scores (Random vs Real Detection)', 
                     fontsize=16, fontweight='bold', pad=15)
        ax1.set_xlabel('Datasets', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Metrics', fontsize=14, fontweight='bold')
        
        # 右侧：效应量热图
        # 使用自适应的颜色范围（基于数据的实际范围）
        effect_vmin = max(0, np.nanmin(effect_size_matrix) - 1)
        effect_vmax = np.nanmax(effect_size_matrix) + 1
        
        sns.heatmap(effect_size_matrix,
                    xticklabels=dataset_labels,
                    yticklabels=display_names, 
                    annot=True,
                    fmt='.2f',
                    cmap='RdYlGn',  # 红-黄-绿渐变（红=低，绿=高）
                    vmin=effect_vmin,
                    vmax=effect_vmax,
                    ax=ax2,
                    cbar_kws={'label': 'Effect Size (Cohen\'s d)'},
                    linewidths=0.5,
                    linecolor='white')
        ax2.set_title('Metric Effect Sizes (Random vs Real Detection)', 
                     fontsize=16, fontweight='bold', pad=15)
        ax2.set_xlabel('Datasets', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Metrics', fontsize=14, fontweight='bold')
        
        # 添加效应量阈值线的说明（可选）
        '''        threshold = self.config.effect_size_threshold
        ax2.text(0.02, 0.98, f'Threshold = {threshold}', 
                transform=ax2.transAxes,
                fontsize=11, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))'''
        
        plt.tight_layout()
        
        # Save (select filename based on version)
        filename = "metric_dataset_heatmap_abbr.png" if use_abbreviation else "metric_dataset_heatmap.png"
        output_file = report_dir / filename
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        version_str = "abbreviated version" if use_abbreviation else "original version"
        print(f"    ✓ Metric × dataset heatmap ({version_str}) saved (AUC + effect size)")


def main():
    """Main program entry"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Time series anomaly detection metric robustness experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --test                    # Quick test mode
  %(prog)s                           # Formal experiment mode
  %(prog)s --no-plots                # Run experiment but do not generate plots
        """
    )
    
    parser.add_argument(
        '--test', '--debug',
        action='store_true',
        dest='test_mode',
        help='Quick test mode (small datasets, fast validation)'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Only run experiment and save data, do not generate plots (can generate later with plot_results.py)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory (default: ./test_results or ./experiment_results)'
    )
    
    args = parser.parse_args()
    
    print("=== Time Series Anomaly Detection Metric Robustness Experiment ===")
    print("This is an experimental framework for evaluating the reliability of anomaly detection metrics")
    print()
    
    if args.test_mode:
        print("Quick code check mode (test all metrics, verify functionality)")
        # Quick configuration - test all metrics but use small datasets and few trials
        config = ExperimentConfig(
            n_datasets=10,          # Maximum dataset count (actual determined by lengths×ratios, set high to avoid truncation)
            n_random_trials=5,     # Slightly increase random trial count to test metric stability
            dataset_lengths=[5000,8000], # 2 sequence lengths
            anomaly_ratios=[0.10], # 2 anomaly ratios -> total 2×2=4 datasets
            alpha=0.05,
            effect_size_threshold=0.3,
            auc_threshold=0.7,
            false_positive_threshold=0.1,
            spearman_threshold=0.5,
            output_dir=args.output or "./test_results",
            enable_parallel=False,  # Disable parallel in test mode for debugging
            skip_slow_metrics=False, # Do not skip slow metrics, test all metrics
            memory_limit_mb=3000,  # Moderate memory limit
            disable_metrics=[  # Disable particularly slow or heavy external dependency metrics for debugging
                "pate_auc_default", "pate_auc_small"
            ],
            save_plots=not args.no_plots  # Decide whether to generate charts based on parameters
        )
        print("Note: This is code test mode, use small datasets to test all metrics")
        print(f"Actual dataset count: {len(config.dataset_lengths)} × {len(config.anomaly_ratios)} = {len(config.dataset_lengths) * len(config.anomaly_ratios)}")
    else:
        print("Formal experiment mode")
        # Standard experiment configuration (according to recommended experiment scale in documentation)
        config = ExperimentConfig(
            n_datasets=30,        # Allow maximum standard dataset count
            n_random_trials=10,    # Standard random score count
            dataset_lengths=[5000, 10000, 50000], # Multiple sequence lengths
            anomaly_ratios=[0.05, 0.10, 0.15], # Multiple anomaly ratios
            alpha=0.05,
            effect_size_threshold=0.3,  # Cohen's d threshold
            auc_threshold=0.9,          # AUC threshold  
            false_positive_threshold=0.1, # False positive threshold
            spearman_threshold=0.5,     # Monotonicity threshold
            output_dir=args.output or "./experiment_results",
            save_plots=not args.no_plots  # Decide whether to generate charts based on parameters
        )
    
    if args.no_plots:
        print("Plotting disabled, can use 'python plot_results.py' to generate plots separately after experiment")
    
    print(f"Experiment configuration:")
    print(f"  - Dataset count: {config.n_datasets}")
    print(f"  - Random trial count: {config.n_random_trials}")
    print(f"  - Sequence lengths: {config.dataset_lengths}")
    print(f"  - Anomaly ratios: {config.anomaly_ratios}")
    print(f"  - Result output directory: {config.output_dir}")
    print(f"  - Generate plots: {'No (can generate later)' if args.no_plots else 'Yes'}")
    print()
    
    # Create and run experiment
    experiment = MetricRobustnessExperiment(config)
    
    try:
        results = experiment.run_experiment()
        
        print("\n=== Experiment Completed ===")
        print(f"✓ Processed {len(results['results'])} datasets")
        
        # 显示使用的指标
        performance_mode = config.get_performance_mode()
        used_metrics = experiment.metric_registry.get_metric_names(performance_mode)
        print(f"✓ Evaluated {len(used_metrics)} metrics")
        print(f"✓ Results saved to: {config.output_dir}")
        
        # Display metric list (grouped)
        print("\nEvaluated Metrics:")
        for i, metric_name in enumerate(used_metrics, 1):
            priority = experiment.metric_registry.get_metric_priority(metric_name)
            priority_label = ["", "快速", "中快", "标准", "慢", "很慢"][priority]
            print(f"  {i:2d}. {metric_name} {priority_label}")
        
        # 显示简要汇总
        summary =  results['summary']
        print("\n Summary:")
        print(f"  {summary['total_datasets']}")
        print(f"{summary['total_metrics']}")
        
        if 'metric_performance' in summary and summary['metric_performance']:
            print("\n Metric Performance Overview:")
            for metric_name, perf in summary['metric_performance'].items():
                pass_rate = perf['pass_rate']
                print(f"  {metric_name}: {pass_rate:.1%} Pass Rate")
        
        print(f"\nDetailed reports and charts saved to: {config.output_dir}/reports/")
        
    except Exception as e:
        print(f"\n✗ Experiment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Experiment completed successfully!")
    else:
        print("\n❌ Experiment failed")
        sys.exit(1)