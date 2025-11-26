"""
Industrial Anomaly Detection Evaluation Metrics - Fuzzy Boundary Weighted Scoring System
========================================================================================

Design Philosophy:
1. Considers annotation boundary ambiguity, supports three annotation styles: over-labeling, balanced, conservative
2. Weight-based TP/FP/FN calculation with gradual transition in boundary regions
3. Timeliness reward: earlier anomaly detection receives higher rewards
4. Sparse false positive penalty: dispersed false positives penalized more than clustered ones
"""

import numpy as np
from typing import Tuple, Literal, Optional
import warnings


class FB_TDF_Metric:
    """
    Fuzzy Boundary Anomaly Detection Evaluation Metric
    
    Parameters
    ----------
    tolerance : int
        Tolerance value (fuzzy interval length), defines the width of boundary fuzzy region
    mode : {'over_label', 'balanced', 'under_label'}
        Annotation mode:
        - 'over_label': Over-labeling mode (annotations are longer than actual)
        - 'balanced': Balanced mode (moderate annotations, boundary weights disabled)
        - 'under_label': Conservative mode (annotations are shorter, only core regions labeled)
    enable_timeliness : bool, default=True
        Whether to enable timeliness reward
    timeliness_decay : float, default=0.5
        Timeliness decay rate, controls weight decay speed from core start to end
        Larger values mean faster decay (heavier penalty for late detection)
    fp_cluster_gap : Optional[int], default=None
        Maximum gap allowed for FP clusters, None means auto-calculate (using median of anomaly segment lengths)
    enable_fp_cluster_penalty : bool, default=False
        Whether to enable FP cluster penalty (dispersed false positives penalized more than clustered ones)
    dispersed_decay : float, default=4.0
        Decay factor for FP cluster penalty (only effective when enable_fp_cluster_penalty=True)
    """
    
    def __init__(
        self,
        tolerance: int,
        mode: Literal['over_label', 'balanced', 'under_label'] = 'balanced',
        enable_timeliness: bool = True,
        timeliness_decay: float = 0.5,
        fp_cluster_gap: Optional[int] = None,
        enable_fp_cluster_penalty: bool = False,
        dispersed_decay: float = 4.0
    ):
        self.tolerance = tolerance
        self.mode = mode
        self.enable_timeliness = enable_timeliness
        self.timeliness_decay = timeliness_decay
        self.fp_cluster_gap = fp_cluster_gap
        self.enable_fp_cluster_penalty = enable_fp_cluster_penalty  # FP cluster penalty switch
        self.dispersed_decay = dispersed_decay  # Adjustable dispersion penalty decay factor
        
        # Validate parameters
        if tolerance < 0:
            raise ValueError("tolerance must be a non-negative integer")
        if not 0 <= timeliness_decay <= 1:
            raise ValueError("timeliness_decay must be in range [0, 1]")
    
    def _get_anomaly_segments(self, labels: np.ndarray) -> list:
        """
        Extract start and end positions of anomaly segments
        
        Returns
        -------
        list of tuple : [(start, end), ...]
        """
        segments = []
        in_anomaly = False
        start = 0
        
        for i in range(len(labels)):
            if labels[i] == 1 and not in_anomaly:
                start = i
                in_anomaly = True
            elif labels[i] == 0 and in_anomaly:
                segments.append((start, i - 1))
                in_anomaly = False
        
        if in_anomaly:
            segments.append((start, len(labels) - 1))
        
        return segments
    
    def _calculate_boundary_weights(self, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate boundary weights
        
        Returns
        -------
        tp_fn_weights : np.ndarray
            Weights for TP and FN (increases from boundary to core within anomaly segments)
        fp_weights : np.ndarray
            Weights for FP (opposite of TP/FN weights)
        """
        n = len(labels)
        tp_fn_weights = np.zeros(n)
        fp_weights = np.ones(n)  # Initialize FP weights to 1.0
        
        # Balanced mode: all weights are 1
        if self.mode == 'balanced' or self.tolerance == 0:
            tp_fn_weights[labels == 1] = 1.0
            fp_weights = 1.0 - tp_fn_weights
            return tp_fn_weights, fp_weights
        
        segments = self._get_anomaly_segments(labels)
        
        for start, end in segments:
            if self.mode == 'over_label':
                # Over-labeling mode: annotation is longer than actual anomaly
                # Increase from annotation start point through tolerance points to core start
                core_start = min(start + self.tolerance, end)
                core_end = max(end - self.tolerance, core_start)
                
                # Start to core start: weight increases linearly from 0 to 1
                transition_length = core_start - start
                if transition_length > 0:
                    tp_fn_weights[start:core_start] = np.linspace(0, 1, transition_length, endpoint=False)
                
                # Core region: weight is 1
                tp_fn_weights[core_start:core_end+1] = 1.0
                
                # Core end to end: weight decreases linearly from 1 to 0
                transition_length_end = end - core_end
                if transition_length_end > 0:
                    tp_fn_weights[core_end+1:end+1] = np.linspace(1, 0, transition_length_end, endpoint=False)
            
            elif self.mode == 'under_label':
                # Conservative mode: annotation is shorter than actual anomaly, only core region labeled
                # Annotation start is core start, weights need to extend tolerance points forward
                core_start = start  # Annotation start is core start
                core_end = end      # Annotation end is core end
                
                # TP/FN weights: from core_start-tolerance to core_end+tolerance
                weight_start = max(0, core_start - self.tolerance)
                weight_end = min(len(labels) - 1, core_end + self.tolerance)
                
                # Weight start to core start: weight increases linearly from 0 to 1
                transition_length = core_start - weight_start
                if transition_length > 0:
                    tp_fn_weights[weight_start:core_start] = np.linspace(0, 1, transition_length, endpoint=False)
                
                # Core region (labeled region): weight is 1
                tp_fn_weights[core_start:core_end+1] = 1.0
                
                # Core end to weight end: weight decreases linearly from 1 to 0
                transition_length_end = weight_end - core_end
                if transition_length_end > 0:
                    tp_fn_weights[core_end+1:weight_end+1] = np.linspace(1, 0, transition_length_end, endpoint=False)
                
                # FP weights: need transition around labeled region
                # Decrease from core_start-tolerance position, reach 0 at core_start
                # Increase from core_end, reach 1.0 at core_end+tolerance
                if transition_length > 0:
                    fp_weights[weight_start:core_start] = np.linspace(1, 0, transition_length, endpoint=False)
                # Core region FP weight is 0
                fp_weights[core_start:core_end+1] = 0.0
                # Increase from core end
                if transition_length_end > 0:
                    fp_weights[core_end+1:weight_end+1] = np.linspace(0, 1, transition_length_end, endpoint=False)
        
        # For over_label and balanced modes, FP weights are inverse of TP/FN weights
        if self.mode != 'under_label':
            fp_weights = 1.0 - tp_fn_weights
            # Ensure FP weights are maximum (1.0) in normal regions
            fp_weights[labels == 0] = 1.0
        
        return tp_fn_weights, fp_weights
    
    def _calculate_timeliness_weights(self, labels: np.ndarray) -> np.ndarray:
        """
        Calculate timeliness weights
        
        Timeliness weights are designed as a reward mechanism:
        - Early detection (near core start): maximum weight (1.0 + bonus)
        - Mid-period detection: weight=1.0 (baseline)
        - Late detection (near core end): weight<1.0 (penalty)
        
        Key design:
        1. Regardless of boundary fuzzy mode, timeliness weights apply within labeled region
        2. From core start to core end: weight decays linearly from (1+decay) to (1-decay)
        3. In fuzzy boundary mode, weights extend to boundary transition regions
        
        Returns
        -------
        timeliness_weights : np.ndarray
            Timeliness weights, same shape as labels
        """
        n = len(labels)
        timeliness_weights = np.ones(n)
        
        if not self.enable_timeliness:
            return timeliness_weights
        
        segments = self._get_anomaly_segments(labels)
        
        for start, end in segments:
            # Determine core region (core part of labeled region)
            if self.mode == 'under_label':
                # Conservative mode: labeled region is the core region
                core_start = start
                core_end = end
            elif self.mode == 'over_label':
                # Over-labeling mode: core start is after annotation start
                core_start = min(start + self.tolerance, end)
                core_end = max(end - self.tolerance, core_start)
            else:  # balanced
                # Balanced mode: labeled region is the core region
                core_start = start
                core_end = end
            
            # Timeliness weights in core region: decay linearly from (1+decay) to (1-decay)
            # Early detection gets reward (>1.0), late detection gets penalty (<1.0)
            core_length = core_end - core_start + 1
            max_weight = 1.0 + self.timeliness_decay
            min_weight = 1.0 - self.timeliness_decay
            
            if core_length > 1:
                timeliness_weights[core_start:core_end+1] = np.linspace(
                    max_weight, min_weight, core_length
                )
            else:
                timeliness_weights[core_start:core_end+1] = 1.0
            
            # In boundary transition region: weight remains at boundary value
            if self.tolerance > 0:
                if self.mode == 'under_label':
                    # under_label mode: extend tolerance points before and after, maintain max/min weights
                    # Before core start: extend to core_start - tolerance, maintain max weight (reward zone)
                    timeliness_start = max(0, core_start - self.tolerance)
                    if core_start > timeliness_start:
                        timeliness_weights[timeliness_start:core_start] = max_weight
                    # After core end: extend to core_end + tolerance, maintain min weight (penalty zone)
                    timeliness_end = min(core_end + self.tolerance, n - 1)
                    if timeliness_end > core_end:
                        timeliness_weights[core_end+1:timeliness_end+1] = min_weight
                elif self.mode == 'over_label':
                    # over_label mode: maintain max weight before core start
                    if core_start > start:
                        timeliness_weights[start:core_start] = max_weight
                    # Maintain min weight after core end
                    if end > core_end:
                        timeliness_weights[core_end+1:end+1] = min_weight
                else:  # balanced mode
                    # balanced mode: no extension, core region is labeled region
                    pass
        
        return timeliness_weights
    
    def _merge_weights(self, boundary_weights: np.ndarray, timeliness_weights: np.ndarray) -> np.ndarray:
        """
        Merge boundary weights and timeliness weights
        
        Recommended multiplication approach:
        - Multiplication preserves constraining properties of both weights
        - If either weight is 0, final weight is 0
        - Both weights play a role, avoiding excessive weights from addition
        
        Alternative approaches:
        1. Weighted average: alpha * boundary + (1-alpha) * timeliness
        2. Minimum: min(boundary, timeliness)
        3. Harmonic mean: 2 / (1/boundary + 1/timeliness)
        
        Returns
        -------
        merged_weights : np.ndarray
            Merged weights
        """
        # Use multiplication to merge
        merged = boundary_weights * timeliness_weights
        return merged

    def _count_fp_clusters(self, fp_mask: np.ndarray, labels: np.ndarray) -> int:
        """
        Count the number of FP clusters
        
        Parameters
        ----------
        fp_mask : np.ndarray
            Boolean mask of False Positives
        labels : np.ndarray
            True labels
            
        Returns
        -------
        cluster_count : int
            Number of FP clusters
        """
        if not np.any(fp_mask):
            return 0
        
        # Determine cluster gap allowance
        if self.fp_cluster_gap is None:
            # Auto-calculate: use half of median anomaly segment length
            segments = self._get_anomaly_segments(labels)
            if segments:
                lengths = [end - start + 1 for start, end in segments]
                gap = int(np.median(lengths) / 2)
            else:
                gap = 1  # Default value 1 if no anomaly segments
        else:
            gap = self.fp_cluster_gap
        
        # Expand FP regions (FPs within gap range considered as one cluster)
        fp_positions = np.nonzero(fp_mask)[0]
        
        if len(fp_positions) == 0:
            return 0
        
        cluster_count = 1
        for i in range(1, len(fp_positions)):
            if fp_positions[i] - fp_positions[i-1] > gap:
                cluster_count += 1
        
        return cluster_count
    
    def calculate_metrics(
        self, 
        y_true: np.ndarray, 
        anomaly_scores: np.ndarray,
        threshold: Optional[float] = None
    ) -> dict:
        """
        Calculate evaluation metrics
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels (0: normal, 1: anomaly)
        anomaly_scores : np.ndarray
            Anomaly scores
        threshold : float, optional
            Threshold, if None will auto-select best threshold
            
        Returns
        -------
        metrics : dict
            Dictionary containing various evaluation metrics
        """
        y_true = np.asarray(y_true)
        anomaly_scores = np.asarray(anomaly_scores)
        
        if len(y_true) != len(anomaly_scores):
            raise ValueError("y_true and anomaly_scores must have the same length")
        
        # If no threshold provided, use median as default threshold
        if threshold is None:
            threshold = np.median(anomaly_scores)
        
        y_pred = (anomaly_scores >= threshold).astype(int)
        
        # Calculate boundary weights
        tp_fn_boundary_weights, fp_boundary_weights = self._calculate_boundary_weights(y_true)
        
        # Calculate timeliness weights
        timeliness_weights = self._calculate_timeliness_weights(y_true)
        
        # Merge TP/FN weights
        tp_fn_weights = self._merge_weights(tp_fn_boundary_weights, timeliness_weights)
        
        # FP weights only use boundary weights (not affected by timeliness)
        fp_weights = fp_boundary_weights
        
        # Calculate confusion matrix elements
        tp_mask = (y_true == 1) & (y_pred == 1)
        fp_mask = (y_true == 0) & (y_pred == 1)
        fn_mask = (y_true == 1) & (y_pred == 0)
        tn_mask = (y_true == 0) & (y_pred == 0)
        
        # Weighted counting
        weighted_tp = np.sum(tp_fn_weights[tp_mask])
        weighted_fp = np.sum(fp_weights[fp_mask])
        weighted_fn = np.sum(tp_fn_weights[fn_mask])
        weighted_tn = np.sum((1.0 - fp_weights)[tn_mask])
        
        # FP cluster penalty (optional feature)
        fp_cluster_count = self._count_fp_clusters(fp_mask, y_true)
        fp_point_count = np.sum(fp_mask)
        
        if self.enable_fp_cluster_penalty:
            # Sparsity penalty factor: more clusters (more dispersed), heavier penalty
            if fp_point_count > 0:
                # Dispersion calculation: cluster count divided by point count
                # When all FPs are dispersed, dispersion=1; when all FPs clustered, dispersion≈0
                sparsity_factor = fp_cluster_count / fp_point_count
                
                # Fix: Linear interpolation penalty
                # Mathematical guarantee:
                # 1. Fully clustered (sparsity=0): penalty = 1.0 (no penalty)
                # 2. Fully dispersed (sparsity=1): penalty = dispersed_decay (maximum penalty)
                # 3. Slope = dispersed_decay - 1 (larger decay, steeper slope)
                sparse_penalty = 1.0 + sparsity_factor * (self.dispersed_decay - 1.0)
                weighted_fp_penalized = weighted_fp * sparse_penalty
            else:
                sparse_penalty = 1.0
                weighted_fp_penalized = 0.0
        else:
            # FP cluster penalty not enabled, use original weighted FP directly
            sparse_penalty = 1.0
            weighted_fp_penalized = weighted_fp
        
        # Calculate precision, recall, F1 score
        # Precision uses penalized FP (if FP cluster penalty enabled)
        precision = weighted_tp / (weighted_tp + weighted_fp_penalized) if (weighted_tp + weighted_fp_penalized) > 0 else 0.0
        recall = weighted_tp / (weighted_tp + weighted_fn) if (weighted_tp + weighted_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calculate accuracy
        # Accuracy uses original FP (not penalized FP, as accuracy reflects classification weights, should not be affected by sparsity penalty)
        accuracy = (weighted_tp + weighted_tn) / (weighted_tp + weighted_fp + weighted_fn + weighted_tn)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'weighted_tp': weighted_tp,
            'weighted_fp': weighted_fp,
            'weighted_fp_penalized': weighted_fp_penalized,
            'weighted_fn': weighted_fn,
            'weighted_tn': weighted_tn,
            'fp_cluster_count': fp_cluster_count,
            'fp_point_count': fp_point_count,
            'sparse_penalty_factor': sparse_penalty,
            'threshold': threshold
        }
    
    def find_best_threshold(
        self,
        y_true: np.ndarray,
        anomaly_scores: np.ndarray,
        n_thresholds: int = 100
    ) -> Tuple[float, dict]:
        """
        Find the best threshold (maximizing F1 score)
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels
        anomaly_scores : np.ndarray
            Anomaly scores
        n_thresholds : int
            Number of thresholds to test
            
        Returns
        -------
        best_threshold : float
            Best threshold
        best_metrics : dict
            Metrics corresponding to best threshold
        """
        min_score = np.min(anomaly_scores)
        max_score = np.max(anomaly_scores)
        thresholds = np.linspace(min_score, max_score, n_thresholds)
        
        best_f1 = -1
        best_threshold = None
        best_metrics = None
        
        for threshold in thresholds:
            metrics = self.calculate_metrics(y_true, anomaly_scores, threshold)
            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                best_threshold = threshold
                best_metrics = metrics
        
        return best_threshold, best_metrics


# Example usage
if __name__ == "__main__":
    # Generate sample data
    rng = np.random.default_rng(42)
    n = 1000
    
    # Create true labels: 3 anomaly segments
    y_true = np.zeros(n)
    y_true[100:150] = 1  # Anomaly segment 1: length 50
    y_true[300:320] = 1  # Anomaly segment 2: length 20
    y_true[600:680] = 1  # Anomaly segment 3: length 80
    
    # Generate anomaly scores
    anomaly_scores = rng.standard_normal(n) * 0.3
    anomaly_scores[y_true == 1] += 1.5  # Higher scores in anomaly regions
    
    # Add some boundary fuzziness and delayed detection
    anomaly_scores[90:110] += 0.8   # Early detection
    anomaly_scores[140:160] += 0.8  # Delayed detection
    anomaly_scores[250:270] += 1.0  # Dispersed false positives
    anomaly_scores[500:510] += 1.0  # Dispersed false positives
    
    # Test different modes
    modes = ['over_label', 'balanced', 'under_label']
    
    print("=" * 80)
    print("Fuzzy Boundary Anomaly Detection Evaluation Metric Test")
    print("=" * 80)
    
    for mode in modes:
        print(f"\nMode: {mode}")
        print("-" * 80)
        
        # Create evaluator
        metric = FB_TDF_Metric(
            tolerance=10,
            mode=mode,
            enable_timeliness=True,
            timeliness_decay=0.5
        )
        
        # Find best threshold
        best_threshold, best_metrics = metric.find_best_threshold(y_true, anomaly_scores)
        
        print(f"Best threshold: {best_threshold:.4f}")
        print(f"Precision: {best_metrics['precision']:.4f}")
        print(f"Recall: {best_metrics['recall']:.4f}")
        print(f"F1 Score: {best_metrics['f1_score']:.4f}")
        print(f"Accuracy: {best_metrics['accuracy']:.4f}")
        print(f"FP cluster count: {best_metrics['fp_cluster_count']}")
        print(f"FP point count: {best_metrics['fp_point_count']}")
        print(f"Sparse penalty factor: {best_metrics['sparse_penalty_factor']:.4f}")