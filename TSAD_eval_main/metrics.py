import numpy as np
import sys
import os
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score, average_precision_score

from nabscore import Sweeper
from affiliation.metrics import pr_from_events as affiliation_pr
from prts import ts_recall, ts_precision
import time_tolerant as ttol
import latency_sparsity_aware
from eTaPR_pkg import etapr, tapr
from eTaPR_pkg.DataManage import File_IO, Range
from vus.analysis.robustness_eval import generate_curve

# Import PATE metric (Precision and Recall with Time-series Evaluation)
# Try multiple import paths to support different installation methods
PATE_AVAILABLE = False
pate_score = None
convert_vector_to_events_PATE = None
ACF_find_buffer_size = None

try:
    # Try direct import (if pate is installed as a package)
    from pate.PATE_metric import PATE as pate_score
    from pate.PATE_utils import convert_vector_to_events_PATE, ACF_find_buffer_size
    PATE_AVAILABLE = True
except ImportError:
    try:
        # Try importing from PATE-main subdirectory
        _current_dir = os.path.dirname(os.path.abspath(__file__))
        _pate_path = os.path.join(_current_dir, 'PATE-main')
        if _pate_path not in sys.path:
            sys.path.insert(0, _pate_path)
        from pate.PATE_metric import PATE as pate_score
        from pate.PATE_utils import convert_vector_to_events_PATE, ACF_find_buffer_size
        PATE_AVAILABLE = True
    except ImportError:
        pass  # PATE not available

# Import FB_TDF_Metric (Fuzzy Boundary Time-aware Detection F1)
FB_TDF_AVAILABLE = False
FB_TDF_Metric = None

try:
    from new_metric import FB_TDF_Metric
    FB_TDF_AVAILABLE = True
except ImportError:
    pass  # FB_TDF not available


# NOTE:
# Binary anomaly time series (either labels or predictions) are represented in 3 different ways.
# This is done to suit the different metrics.
#
# Example:
# A time series of length 10 (t=0 to t=9)  with anomalies at times t=2, t=6 and t=7 is represented like this:
# Segmentwise: [[2,2], [6,7]]
# Pointwise: [2,6,7]
# Full_series: [0,0,1,0,0,0,1,1,0,0]
#
# The class Binary_anomalies is used to access these various formats.


def pointwise_to_segmentwise(pointwise):
    """Reformat anomaly time series from pointwise to segmentwise"""
    segmentwise = []

    prev = -10
    for point in pointwise:
        if point > prev + 1:
            segmentwise.append([point, point])
        else:
            segmentwise[-1][-1] += 1
        prev = point
    return np.array(segmentwise)


def segmentwise_to_pointwise(segmentwise):
    """Reformat anomaly time series from segmentwise to pointwise"""
    pointwise = []

    for start, end in segmentwise:
        for point in range(start, end + 1):
            pointwise.append(point)

    return np.array(pointwise)


def pointwise_to_full_series(pointwise, length):
    """Reformat anomaly time series from pointwise to full_series"""
    anomalies_full_series = np.zeros(length)
    if len(pointwise) > 0:
        assert pointwise[-1] < length
        anomalies_full_series[pointwise] = 1
    return anomalies_full_series


class Binary_anomalies:
    def __init__(self, length, anomalies):
        self._length = length
        self._set_anomalies(anomalies)

    def _set_anomalies(self, anomalies):
        anomalies = np.array(anomalies)
        if self._is_pointwise(anomalies):
            anomalies_ptwise = anomalies
            anomalies_segmentwise = pointwise_to_segmentwise(anomalies)
            anomalies_full_series = pointwise_to_full_series(anomalies_ptwise, self._length)
        elif self._is_full_series(anomalies):
            raise NotImplementedError
        elif self._is_segmentwise(anomalies):
            anomalies_segmentwise = anomalies
            anomalies_ptwise = segmentwise_to_pointwise(anomalies)
            anomalies_full_series = pointwise_to_full_series(anomalies_ptwise, self._length)
        else:
            raise ValueError(f"Illegal shape of anomalies:\n{anomalies}")

        if len(anomalies_ptwise) > 0:
            assert all(anomalies_ptwise == np.sort(anomalies_ptwise))
            assert anomalies_ptwise[0] >= 0
            assert len(anomalies_ptwise) == len(np.unique(anomalies_ptwise))
            assert len(anomalies_ptwise) == sum(anomalies_full_series)

            assert all(anomalies_segmentwise[:, 0] == np.sort(anomalies_segmentwise[:, 0]))
            assert all(anomalies_segmentwise[:, 1] >= anomalies_segmentwise[:, 0])

        self.anomalies_segmentwise = anomalies_segmentwise
        self.anomalies_ptwise = anomalies_ptwise
        self.anomalies_full_series = anomalies_full_series

    def _is_pointwise(self, anomalies):
        return len(anomalies.shape) == 1 and len(anomalies) < self._length

    def _is_full_series(self, anomalies):
        return len(anomalies.shape) == 1 and len(anomalies) == self._length

    def _is_segmentwise(self, anomalies):
        return len(anomalies.shape) == 2

    def get_length(self):
        return self._length


class Binary_detection:
    """This class represents a binary detection as a set of two time series:
    gt: the binary labels
    prediction: the binary predictions for corresponding to the labels"""

    def __init__(self, length, gt_anomalies, predicted_anomalies):
        self._length = length
        self._gt = Binary_anomalies(length, gt_anomalies)
        self._prediction = Binary_anomalies(length, predicted_anomalies)

    def get_length(self):
        return self._length

    def get_gt_anomalies_ptwise(self):
        return self._gt.anomalies_ptwise

    def get_gt_anomalies_segmentwise(self):
        return self._gt.anomalies_segmentwise

    def get_predicted_anomalies_ptwise(self):
        return self._prediction.anomalies_ptwise

    def get_predicted_anomalies_segmentwise(self):
        return self._prediction.anomalies_segmentwise

    def get_predicted_anomalies_full_series(self):
        return self._prediction.anomalies_full_series

    def get_gt_anomalies_full_series(self):
        return self._gt.anomalies_full_series


class Nonbinary_detection:
    """This class represents a nonbinary detection as a set of two time series:
    gt: the binary labels
    anomaly score: the time series defining the degree of anomaly at each time point"""

    def __init__(self, gt_anomalies, anomaly_score):
        self._length = len(anomaly_score)
        self._gt = Binary_anomalies(self._length, gt_anomalies)
        self._anomaly_score = anomaly_score

    def get_gt_anomalies_ptwise(self):
        return self._gt.anomalies_ptwise

    def get_gt_anomalies_segmentwise(self):
        return self._gt.anomalies_segmentwise

    def get_gt_anomalies_full_series(self):
        return self._gt.anomalies_full_series

    def get_anomaly_score(self):
        return self._anomaly_score


def f1_from_pr(p, r, beta=1):
    if r == 0 and p == 0:
        return 0
    return ((1 + beta**2) * r * p) / (beta**2 * p + r)


def f1_score(*args, tp, fp, fn, beta=1):
    r = recall(tp=tp, fn=fn)
    p = precision(tp=tp, fp=fp)
    return f1_from_pr(p, r, beta=beta)


def recall(*args, tp, fn):
    return 0 if tp + fn == 0 else tp / (tp + fn)


def precision(*args, tp, fp):
    return 0 if tp + fp == 0 else tp / (tp + fp)


class Pointwise_metrics(Binary_detection):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = "\\pwf[1]"
        self.set_confusion()

    def set_confusion(self):
        gt = self.get_gt_anomalies_full_series()
        pred = self.get_predicted_anomalies_full_series()

        self.tp = np.sum(pred * gt)
        self.fp = np.sum(pred * (1 - gt))
        self.fn = np.sum((1 - pred) * gt)

    def get_score(self):
        return f1_score(tp=self.tp, fn=self.fn, fp=self.fp)


class DelayThresholdedPointAdjust(Pointwise_metrics):
    def __init__(self, *args, k=2):
        super().__init__(*args)
        self.name = f"\\dtpaf[1]{{{k}}}"
        self.k = k
        self.adjust()
        self.set_confusion()

    def adjust(self):
        adjusted_prediction = self.get_predicted_anomalies_ptwise().tolist()
        for start, end in self.get_gt_anomalies_segmentwise():
            anomaly_adjusted = False
            for i in range(start, min(start + self.k + 1, end + 1)):
                if i in adjusted_prediction:
                    for j in range(start, end + 1):
                        adjusted_prediction.append(j)
                    anomaly_adjusted = True
                    break
            if anomaly_adjusted == False:
                for i in range(start, end + 1):
                    try:
                        adjusted_prediction.remove(i)
                    except ValueError:
                        pass

        self._prediction._set_anomalies(np.sort(np.unique(adjusted_prediction)))


class PointAdjust(DelayThresholdedPointAdjust):
    def __init__(self, *args):
        super().__init__(*args, k=args[0])  # set k to length of time series to avoid threshold making a difference
        self.name = "\\paf[1]"


class PointAdjustKPercent(Pointwise_metrics):
    def __init__(self, *args, k=0.2):
        super().__init__(*args)
        self.name = f"\\pakf[1]{{{int(k*100)}}}"
        self.k = k
        self.adjust()
        self.set_confusion()

    def adjust(self):
        adjusted_prediction = self.get_predicted_anomalies_ptwise().tolist()
        for start, end in self.get_gt_anomalies_segmentwise():
            correct_points = 0
            for i in range(start, end + 1):
                if i in adjusted_prediction:
                    correct_points += 1
                    if correct_points / (end + 1 - start) >= self.k:
                        for j in range(start, end + 1):
                            adjusted_prediction.append(j)
                        break

        self._prediction._set_anomalies(np.sort(np.unique(adjusted_prediction)))


class LatencySparsityAware(Binary_detection):
    def __init__(self, *args, tw=2):
        self.name = f"\\lsf[1]{{{tw}}}"
        super().__init__(*args)
        self.tw = tw

    def get_score(self):
        f1, p, r, FPR, self.tp, self.tn, self.fp, self.fn = latency_sparsity_aware.calc_twseq(
            self.get_predicted_anomalies_full_series(),
            self.get_gt_anomalies_full_series(),
            normal=0,
            threshold=0.5,
            tw=self.tw,
        )
        return f1


class Segmentwise_metrics(Pointwise_metrics):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = "\\segf[1]"
        self.set_confusion()

    def set_confusion(self):
        tp = 0
        fn = 0
        for gt_anomaly in self.get_gt_anomalies_segmentwise():
            found = False
            for predicted_anomaly in self.get_predicted_anomalies_segmentwise():
                if self._overlap(gt_anomaly, predicted_anomaly):
                    tp += 1
                    found = True
                    break
            if found == False:
                fn += 1
        fp = 0
        for predicted_anomaly in self.get_predicted_anomalies_segmentwise():
            found = False
            for gt_anomaly in self.get_gt_anomalies_segmentwise():
                if self._overlap(gt_anomaly, predicted_anomaly):
                    found = True
                    break
            if found == False:
                fp += 1
        self.fp = fp
        self.fn = fn
        self.tp = tp

    def _overlap(self, anomaly1, anomaly2):
        return not (anomaly1[1] < anomaly2[0] or anomaly2[1] < anomaly1[0])


class Redefined_PR_metric(Binary_detection):
    def __init__(self, *args):
        super().__init__(*args)

    def get_score(self):
        self.r = self.recall()
        self.p = self.precision()
        return f1_from_pr(self.p, self.r)

    def recall(self):
        raise NotImplementedError

    def precision(self):
        raise NotImplementedError


class Composite_f(Redefined_PR_metric):
    def __init__(self, *args):
        self.name = "\\cf[1]"
        super().__init__(*args)

        self.pointwise_metrics = Pointwise_metrics(*args)
        self.segmentwise_metrics = Segmentwise_metrics(*args)

    def recall(self):
        return recall(tp=self.segmentwise_metrics.tp, fn=self.segmentwise_metrics.fn)

    def precision(self):
        return precision(tp=self.pointwise_metrics.tp, fp=self.pointwise_metrics.fp)


class Affiliation(Redefined_PR_metric):
    def __init__(self, *args):
        self.name = "\\af[1]"
        super().__init__(*args)

    def get_score(self):
        pr_output = affiliation_pr(
            self._reformat_segments(self.get_predicted_anomalies_segmentwise()),
            self._reformat_segments(self.get_gt_anomalies_segmentwise()),
            (0, self.get_length()),
        )
        self.r = pr_output["recall"]
        self.p = pr_output["precision"]
        return f1_from_pr(self.p, self.r)

    def _reformat_segments(self, segments):
        segments = self._include_end_of_segments(segments)
        segments = self._tuplify_segments(segments)
        return segments

    def _include_end_of_segments(self, segments):
        return [[start, end + 1] for start, end in segments]

    def _tuplify_segments(self, segments):
        return [tuple(segment) for segment in segments]


class Range_PR(Redefined_PR_metric):
    def __init__(self, *args, alpha=0.2, bias="flat"):
        super().__init__(*args)
        self.alpha = alpha
        self.bias = bias
        self.set_name()

    def set_name(self):
        self.name = f"\\rf[1]{{{self.bias}}}{{{self.alpha}}}"

    def set_kwargs(self):
        real = np.zeros(self.get_length())
        real[self.get_gt_anomalies_ptwise()] = 1
        pred = np.zeros(self.get_length())
        pred[self.get_predicted_anomalies_ptwise()] = 1

        self.kwargs = {"real": real, "pred": pred, "alpha": self.alpha, "cardinality": "one", "bias": self.bias}

    def recall(self):
        self.set_kwargs()
        return ts_recall(**self.kwargs)

    def precision(self):
        self.set_kwargs()
        return ts_precision(**self.kwargs)


class TaF(Redefined_PR_metric):
    def __init__(self, *args, theta=0.5, alpha=0.5, delta=0):
        super().__init__(*args)
        self.alpha = alpha
        self.theta = theta
        self.delta = delta
        self.name = f"\\taf[1]{{{self.alpha}}}{{{self.delta}}}{{{self.theta}}}"

        self.prepare_scoring()

    def prepare_scoring(self):
        self.prepare_data()
        self.TaPR = tapr.TaPR(theta=self.theta, delta=self.delta)
        self.TaPR.set_anomalies(self.gt_anomalies)
        self.TaPR.set_predictions(self.predicted_anomalies)

    def prepare_data(self):
        self.write_data_files()
        self.read_data_files()

    def write_data_files(self):
        self.gt_filename = "temp_gt.txt"
        with open(self.gt_filename, "w") as f:
            for x in self.get_gt_anomalies_full_series():
                f.write(str(1 if x == 0 else -1))
                f.write("\n")
        self.pred_filename = "temp_pred.txt"
        with open(self.pred_filename, "w") as f:
            for x in self.get_predicted_anomalies_full_series():
                f.write(str(1 if x == 0 else -1))
                f.write("\n")

    def read_data_files(self):
        self.gt_anomalies = File_IO.load_file(self.gt_filename, "stream")
        self.predicted_anomalies = File_IO.load_file(self.pred_filename, "stream")

    def recall(self):
        tard_value, detected_list = self.TaPR.TaR_d()
        tarp_value = self.TaPR.TaR_p()
        return self.alpha * tard_value + (1 - self.alpha) * tarp_value

    def precision(self):
        tapd_value, correct_list = self.TaPR.TaP_d()
        tapp_value = self.TaPR.TaP_p()
        return self.alpha * tapd_value + (1 - self.alpha) * tapp_value


class eTaF(Redefined_PR_metric):
    def __init__(self, *args, theta_p=0.5, theta_r=0.1, delta=0.0):
        super().__init__(*args)
        self.theta_p = theta_p
        self.theta_r = theta_r
        self.delta = delta

        self.name = f"\\etaf[1]{{{self.theta_p}}}{{{self.theta_r}}}{{{self.delta}}}"

        self.make_scores()

    def make_scores(self):
        self.prepare_data()
        self.result = etapr.evaluate_w_ranges(
            self.gt_anomalies, self.predicted_anomalies, theta_p=self.theta_p, theta_r=self.theta_r, delta=self.delta
        )

    def prepare_data(self):
        self.write_data_files()
        self.read_data_files()

    def write_data_files(self):
        self.gt_filename = "temp_gt.txt"
        with open(self.gt_filename, "w") as f:
            for x in self.get_gt_anomalies_full_series():
                f.write(str(1 if x == 0 else -1))
                f.write("\n")
        self.pred_filename = "temp_pred.txt"
        with open(self.pred_filename, "w") as f:
            for x in self.get_predicted_anomalies_full_series():
                f.write(str(1 if x == 0 else -1))
                f.write("\n")

    def read_data_files(self):
        self.gt_anomalies = File_IO.load_file(self.gt_filename, "stream")
        self.predicted_anomalies = File_IO.load_file(self.pred_filename, "stream")

    def recall(self):
        return self.result["eTaR"]

    def precision(self):
        return self.result["eTaP"]


class Time_Tolerant(Redefined_PR_metric):
    def __init__(self, *args, d=2):
        super().__init__(*args)
        self.d = d
        self.name = f"\\ttolf[1]{{{d}}}"

    def recall(self):
        return ttol.recall(**self.get_kwargs())

    def precision(self):
        return ttol.precision(**self.get_kwargs())

    def get_kwargs(self):
        return {
            "A": np.pad(self.get_predicted_anomalies_full_series(), self.d),
            "E": np.pad(self.get_gt_anomalies_full_series(), self.d),
            "d": self.d,
        }


class Temporal_Distance(Binary_detection):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = f"\\tempdist"

    def get_score(self):
        a = np.array(self.get_gt_anomalies_ptwise())
        b = np.array(self.get_predicted_anomalies_ptwise())
        return self._dist(a, b) + self._dist(b, a)

    def _dist(self, a, b):
        dist = 0
        for pt in a:
            if len(b) > 0:
                dist += min(abs(b - pt))
            else:
                dist += self._length
        return dist


class NAB_score(Binary_detection):
    def __init__(self, *args):
        self.name = "\\nab"
        super().__init__(*args)

        self.sweeper = Sweeper(probationPercent=0, costMatrix={"tpWeight": 1, "fpWeight": 0.11, "fnWeight": 1})

    def get_score(self):
        if len(self.get_predicted_anomalies_ptwise()) == 0:
            return 0  # raw_score == null_score yeilds score = 0
        if len(self.get_gt_anomalies_ptwise()) == 0:
            return np.nan  # perfect_score == null_score yields /0
        try:
            null_score, raw_score = self.calculate_scores(self.get_predicted_anomalies_ptwise())
            null_score, perfect_score = self.calculate_scores(prediction=self.get_gt_anomalies_ptwise())
            return (raw_score - null_score) / (perfect_score - null_score) * 100
        except ZeroDivisionError:
            return np.nan

    def calculate_scores(self, prediction):
        anomaly_scores = pointwise_to_full_series(prediction, self.get_length())
        timestamps = np.arange(self.get_length())
        windowLimits = self.get_gt_anomalies_segmentwise()
        dataSetName = "dummyname"
        anomalyList = self.sweeper.calcSweepScore(timestamps, anomaly_scores, windowLimits, dataSetName)
        scoresByThreshold = self.sweeper.calcScoreByThreshold(anomalyList)

        assert scoresByThreshold[0].threshold == 1.1  # all points regarded normal
        assert scoresByThreshold[1].threshold == 1.0  # anomal points regarded anomal

        return scoresByThreshold[0].score, scoresByThreshold[1].score


class Best_threshold_pw(Nonbinary_detection):
    def __init__(self, *args):
        self.name = "\\bestpwf"
        super().__init__(*args)

    def get_score(self):
        scores = []
        for current_anomaly_score in self.get_anomaly_score():
            scores.append(self.get_score_given_anomaly_score_and_threshold(threshold=current_anomaly_score))
        return np.nanmax(scores)

    def get_score_given_anomaly_score_and_threshold(self, threshold):
        gt = self.get_gt_anomalies_full_series()
        pred = np.array(self.get_anomaly_score()) >= threshold
        return f1_score(tp=pred @ gt, fn=(1 - pred) @ gt, fp=(1 - gt) @ pred)


class AUC_ROC(Nonbinary_detection):
    def __init__(self, *args):
        self.name = "\\aucroc"
        super().__init__(*args)

    def get_score(self):
        gt = self.get_gt_anomalies_full_series()
        return roc_auc_score(gt, self.get_anomaly_score())


class AUC_PR_pw(Nonbinary_detection):
    def __init__(self, *args):
        self.name = "\\aucpr"
        super().__init__(*args)

    def get_score(self):
        gt = self.get_gt_anomalies_full_series()
        return average_precision_score(gt, self.get_anomaly_score())


class VUS_ROC(Nonbinary_detection):
    def __init__(self, *args, max_window=4):
        super().__init__(*args)
        self.name = f"\\vusroc{{{max_window}}}"
        self.max_window = max_window

    def get_score(self):
        gt = np.array(self.get_gt_anomalies_full_series())
        score = np.array(self.get_anomaly_score())
        _, _, _, _, _, _, roc, pr = generate_curve(gt, score, self.max_window)
        return roc


class VUS_PR(Nonbinary_detection):
    def __init__(self, *args, max_window=4):
        super().__init__(*args)
        self.name = f"\\vuspr{{{max_window}}}"
        self.max_window = max_window

    def get_score(self):
        gt = np.array(self.get_gt_anomalies_full_series())
        score = np.array(self.get_anomaly_score())
        _, _, _, _, _, _, roc, pr = generate_curve(gt, score, self.max_window)
        return pr


class PatK_pw(Nonbinary_detection):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = f"\\patk[{len(self.get_gt_anomalies_ptwise())}]"

    def get_score(self):
        gt = self.get_gt_anomalies_full_series()

        k = int(sum(gt))
        assert k > 0
        threshold = np.sort(self.get_anomaly_score())[-k]

        pred = self.get_anomaly_score() >= threshold
        assert sum(pred) >= k, (k, pred)

        return pred @ gt / sum(pred)


# =============================================================================
# PATE Metrics (Precision and Recall with Time-series Evaluation)
# =============================================================================

class PATE_F1(Binary_detection):
    """
    PATE F1 metric for binary predictions.
    
    This metric evaluates anomaly detection considering time-series characteristics
    with early and delayed buffer zones around anomalies.
    
    Reference:
    - PATE: Precision and Recall with Time-series Evaluation
    
    Args:
        e_buffer (int): Early detection buffer size (default: 100)
        d_buffer (int): Delayed detection buffer size (default: 100)
        num_splits_MaxBuffer (int): Number of buffer size splits to evaluate (default: 1)
        include_zero (bool): Whether to include buffer size of zero (default: True)
    """
    
    def __init__(self, *args, e_buffer=100, d_buffer=100, num_splits_MaxBuffer=1, include_zero=True):
        super().__init__(*args)
        self.e_buffer = e_buffer
        self.d_buffer = d_buffer
        self.num_splits_MaxBuffer = num_splits_MaxBuffer
        self.include_zero = include_zero
        self.name = f"\\patef1{{{e_buffer}}}{{{d_buffer}}}"
    
    def get_score(self):
        if not PATE_AVAILABLE:
            raise ImportError("PATE module is not available. Please install pate package.")
        
        gt = self.get_gt_anomalies_full_series()
        pred = self.get_predicted_anomalies_full_series()
        
        return pate_score(
            y_true=gt,
            y_score=pred,
            e_buffer=self.e_buffer,
            d_buffer=self.d_buffer,
            num_splits_MaxBuffer=self.num_splits_MaxBuffer,
            include_zero=self.include_zero,
            binary_scores=True
        )


class PATE_AUC_PR(Nonbinary_detection):
    """
    PATE AUC-PR metric for continuous anomaly scores.
    
    This metric evaluates anomaly detection by computing the Area Under the
    Precision-Recall curve with PATE's time-aware evaluation methodology.
    
    Reference:
    - PATE: Precision and Recall with Time-series Evaluation
    
    Args:
        e_buffer (int): Early detection buffer size (default: 100)
        d_buffer (int): Delayed detection buffer size (default: 100)
        n_jobs (int): Number of parallel jobs (default: 1)
        drop_intermediate (bool): Whether to drop intermediate thresholds (default: True)
        Big_Data (bool): Whether to reduce thresholds for large datasets (default: True)
        num_desired_thresholds (int): Number of thresholds to evaluate (default: 250)
        num_splits_MaxBuffer (int): Number of buffer size splits (default: 1)
        include_zero (bool): Whether to include buffer size of zero (default: True)
    """
    
    def __init__(self, *args, e_buffer=100, d_buffer=100, n_jobs=1, 
                 drop_intermediate=True, Big_Data=True, num_desired_thresholds=250,
                 num_splits_MaxBuffer=1, include_zero=True):
        super().__init__(*args)
        self.e_buffer = e_buffer
        self.d_buffer = d_buffer
        self.n_jobs = n_jobs
        self.drop_intermediate = drop_intermediate
        self.Big_Data = Big_Data
        self.num_desired_thresholds = num_desired_thresholds
        self.num_splits_MaxBuffer = num_splits_MaxBuffer
        self.include_zero = include_zero
        self.name = f"\\pateaucpr{{{e_buffer}}}{{{d_buffer}}}"
    
    def get_score(self):
        if not PATE_AVAILABLE:
            raise ImportError("PATE module is not available. Please install pate package.")
        
        gt = self.get_gt_anomalies_full_series()
        score = np.array(self.get_anomaly_score())
        
        return pate_score(
            y_true=gt,
            y_score=score,
            e_buffer=self.e_buffer,
            d_buffer=self.d_buffer,
            n_jobs=self.n_jobs,
            drop_intermediate=self.drop_intermediate,
            Big_Data=self.Big_Data,
            num_desired_thresholds=self.num_desired_thresholds,
            num_splits_MaxBuffer=self.num_splits_MaxBuffer,
            include_zero=self.include_zero,
            binary_scores=False
        )


class PATE_Auto_Buffer(Nonbinary_detection):
    """
    PATE AUC-PR metric with automatic buffer size determination using ACF.
    
    This variant automatically determines the buffer size based on 
    autocorrelation analysis of the time series data.
    
    Reference:
    - PATE: Precision and Recall with Time-series Evaluation
    - VUS paper for ACF-based buffer size selection
    
    Args:
        data (np.ndarray, optional): Time series data for buffer size estimation
        n_jobs (int): Number of parallel jobs (default: 1)
        drop_intermediate (bool): Whether to drop intermediate thresholds (default: True)
        Big_Data (bool): Whether to reduce thresholds for large datasets (default: True)
        num_desired_thresholds (int): Number of thresholds to evaluate (default: 250)
        num_splits_MaxBuffer (int): Number of buffer size splits (default: 1)
        include_zero (bool): Whether to include buffer size of zero (default: True)
    """
    
    def __init__(self, *args, data=None, n_jobs=1, drop_intermediate=True, 
                 Big_Data=True, num_desired_thresholds=250,
                 num_splits_MaxBuffer=1, include_zero=True):
        super().__init__(*args)
        self.data = data
        self.n_jobs = n_jobs
        self.drop_intermediate = drop_intermediate
        self.Big_Data = Big_Data
        self.num_desired_thresholds = num_desired_thresholds
        self.num_splits_MaxBuffer = num_splits_MaxBuffer
        self.include_zero = include_zero
        self.name = "\\pateauto"
    
    def _determine_buffer_size(self):
        if not PATE_AVAILABLE:
            raise ImportError("PATE module is not available. Please install pate package.")
        
        if self.data is not None:
            return ACF_find_buffer_size(self.data)
        else:
            # Use anomaly score as data if no data provided
            return ACF_find_buffer_size(np.array(self.get_anomaly_score()))
    
    def get_score(self):
        if not PATE_AVAILABLE:
            raise ImportError("PATE module is not available. Please install pate package.")
        
        buffer_size = self._determine_buffer_size()
        
        gt = self.get_gt_anomalies_full_series()
        score = np.array(self.get_anomaly_score())
        
        return pate_score(
            y_true=gt,
            y_score=score,
            e_buffer=buffer_size,
            d_buffer=buffer_size,
            n_jobs=self.n_jobs,
            drop_intermediate=self.drop_intermediate,
            Big_Data=self.Big_Data,
            num_desired_thresholds=self.num_desired_thresholds,
            num_splits_MaxBuffer=self.num_splits_MaxBuffer,
            include_zero=self.include_zero,
            binary_scores=False
        )


# =============================================================================
# FB_TDF Metrics (Fuzzy Boundary Time-aware Detection F1)
# =============================================================================

class FB_TDF_F1(Nonbinary_detection):
    """
    Fuzzy Boundary Time-aware Detection F1 metric for continuous anomaly scores.
    
    This metric evaluates anomaly detection considering:
    1. Boundary ambiguity with gradual weight transitions
    2. Timeliness rewards for early detection
    3. Optional FP cluster penalty for dispersed false positives
    
    Args:
        tolerance (int): Tolerance value defining the fuzzy boundary width (default: 10)
        mode (str): Labeling mode - 'over_label', 'balanced', or 'under_label' (default: 'balanced')
        enable_timeliness (bool): Whether to enable timeliness rewards (default: True)
        timeliness_decay (float): Timeliness decay rate in [0, 1] (default: 0.5)
        enable_fp_cluster_penalty (bool): Whether to enable FP cluster penalty (default: False)
        fp_cluster_gap (int, optional): Maximum gap for FP clustering (default: None, auto)
        dispersed_decay (float): FP cluster penalty decay factor (default: 4.0)
        threshold (float, optional): Classification threshold (default: None, uses best)
        n_thresholds (int): Number of thresholds to search when threshold=None (default: 100)
    """
    
    def __init__(self, *args, tolerance=10, mode='balanced', enable_timeliness=True,
                 timeliness_decay=0.5, enable_fp_cluster_penalty=False,
                 fp_cluster_gap=None, dispersed_decay=4.0, 
                 threshold=None, n_thresholds=100):
        super().__init__(*args)
        self.tolerance = tolerance
        self.mode = mode
        self.enable_timeliness = enable_timeliness
        self.timeliness_decay = timeliness_decay
        self.enable_fp_cluster_penalty = enable_fp_cluster_penalty
        self.fp_cluster_gap = fp_cluster_gap
        self.dispersed_decay = dispersed_decay
        self.threshold = threshold
        self.n_thresholds = n_thresholds
        self.name = f"\\fbtdf{{{tolerance}}}{{{mode}}}"
    
    def get_score(self):
        if not FB_TDF_AVAILABLE:
            raise ImportError("FB_TDF_Metric is not available. Please check new_metric.py.")
        
        gt = self.get_gt_anomalies_full_series()
        score = np.array(self.get_anomaly_score())
        
        metric = FB_TDF_Metric(
            tolerance=self.tolerance,
            mode=self.mode,
            enable_timeliness=self.enable_timeliness,
            timeliness_decay=self.timeliness_decay,
            fp_cluster_gap=self.fp_cluster_gap,
            enable_fp_cluster_penalty=self.enable_fp_cluster_penalty,
            dispersed_decay=self.dispersed_decay
        )
        
        if self.threshold is not None:
            result = metric.calculate_metrics(gt, score, threshold=self.threshold)
        else:
            _, result = metric.find_best_threshold(gt, score, n_thresholds=self.n_thresholds)
        
        # Store additional metrics for inspection
        self.precision = result['precision']
        self.recall = result['recall']
        self.best_threshold = result['threshold']
        self.weighted_tp = result['weighted_tp']
        self.weighted_fp = result['weighted_fp']
        self.weighted_fn = result['weighted_fn']
        
        return result['f1_score']
    
    def get_detailed_metrics(self):
        """Return detailed metrics after get_score() is called."""
        return {
            'precision': getattr(self, 'precision', None),
            'recall': getattr(self, 'recall', None),
            'threshold': getattr(self, 'best_threshold', None),
            'weighted_tp': getattr(self, 'weighted_tp', None),
            'weighted_fp': getattr(self, 'weighted_fp', None),
            'weighted_fn': getattr(self, 'weighted_fn', None)
        }


class FB_TDF_Binary(Binary_detection):
    """
    Fuzzy Boundary Time-aware Detection F1 metric for binary predictions.
    
    This variant uses binary predictions directly instead of continuous scores.
    
    Args:
        tolerance (int): Tolerance value defining the fuzzy boundary width (default: 10)
        mode (str): Labeling mode - 'over_label', 'balanced', or 'under_label' (default: 'balanced')
        enable_timeliness (bool): Whether to enable timeliness rewards (default: True)
        timeliness_decay (float): Timeliness decay rate in [0, 1] (default: 0.5)
        enable_fp_cluster_penalty (bool): Whether to enable FP cluster penalty (default: False)
        fp_cluster_gap (int, optional): Maximum gap for FP clustering (default: None, auto)
        dispersed_decay (float): FP cluster penalty decay factor (default: 4.0)
    """
    
    def __init__(self, *args, tolerance=10, mode='balanced', enable_timeliness=True,
                 timeliness_decay=0.5, enable_fp_cluster_penalty=False,
                 fp_cluster_gap=None, dispersed_decay=4.0):
        super().__init__(*args)
        self.tolerance = tolerance
        self.mode = mode
        self.enable_timeliness = enable_timeliness
        self.timeliness_decay = timeliness_decay
        self.enable_fp_cluster_penalty = enable_fp_cluster_penalty
        self.fp_cluster_gap = fp_cluster_gap
        self.dispersed_decay = dispersed_decay
        self.name = f"\\fbtdfbin{{{tolerance}}}{{{mode}}}"
    
    def get_score(self):
        if not FB_TDF_AVAILABLE:
            raise ImportError("FB_TDF_Metric is not available. Please check new_metric.py.")
        
        gt = self.get_gt_anomalies_full_series()
        pred = self.get_predicted_anomalies_full_series()
        
        metric = FB_TDF_Metric(
            tolerance=self.tolerance,
            mode=self.mode,
            enable_timeliness=self.enable_timeliness,
            timeliness_decay=self.timeliness_decay,
            fp_cluster_gap=self.fp_cluster_gap,
            enable_fp_cluster_penalty=self.enable_fp_cluster_penalty,
            dispersed_decay=self.dispersed_decay
        )
        
        # Use threshold=0.5 for binary predictions (0 or 1)
        result = metric.calculate_metrics(gt, pred, threshold=0.5)
        
        # Store additional metrics for inspection
        self.precision = result['precision']
        self.recall = result['recall']
        self.weighted_tp = result['weighted_tp']
        self.weighted_fp = result['weighted_fp']
        self.weighted_fn = result['weighted_fn']
        
        return result['f1_score']
    
    def get_detailed_metrics(self):
        """Return detailed metrics after get_score() is called."""
        return {
            'precision': getattr(self, 'precision', None),
            'recall': getattr(self, 'recall', None),
            'weighted_tp': getattr(self, 'weighted_tp', None),
            'weighted_fp': getattr(self, 'weighted_fp', None),
            'weighted_fn': getattr(self, 'weighted_fn', None)
        }