import numpy as np
import torch
import torch.nn.functional as F



def detection_adjustment(pred, gt):
    pred_copy = pred.copy()
    gt_copy = gt.copy()
    
    anomaly_state = False
    for i in range(len(gt_copy)):
        if gt_copy[i] == 1 and pred_copy[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt_copy[j] == 0:
                    break
                else:
                    if pred_copy[j] == 0:
                        pred_copy[j] = 1
            for j in range(i, len(gt_copy)):
                if gt_copy[j] == 0:
                    break
                else:
                    if pred_copy[j] == 0:
                        pred_copy[j] = 1
        elif gt_copy[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred_copy[i] = 1

    return pred_copy, gt_copy


def pak(scores, targets, k=20):
    scores_copy = scores.copy()
    targets_copy = targets.copy()

    print('scores_copy.shape: ', scores_copy.shape)
    print('targets_copy.shape: ', targets_copy.shape)


    print(f"Unique values in targets_copy: {np.unique(targets_copy)}")
    one_start_idx = np.where(np.diff(targets_copy, prepend=0) == 1)[0]
    zero_start_idx = np.where(np.diff(targets_copy, prepend=0) == -1)[0]

    if not (len(one_start_idx) == len(zero_start_idx) + 1 or len(one_start_idx) == len(zero_start_idx)):
        print(f'one_start_idx: {one_start_idx}')
        print(f'zero_start_idx: {zero_start_idx}')
        raise AssertionError("Lengths of one_start_idx and zero_start_idx do not match the expected relationship.")

    if len(one_start_idx) == len(zero_start_idx) + 1:
        zero_start_idx = np.append(zero_start_idx, len(scores_copy))

    for i in range(len(one_start_idx)):
        if scores_copy[one_start_idx[i]:zero_start_idx[i]].sum() > k / 100 * (zero_start_idx[i] - one_start_idx[i]):
            scores_copy[one_start_idx[i]:zero_start_idx[i]] = 1

    return scores_copy, targets_copy

def adjust_predictions_k(predicted_labels, true_labels, k):
    # Ensure inputs are NumPy arrays
    predicted_labels = np.array(predicted_labels)
    true_labels = np.array(true_labels)
    
    # Create a copy of predicted labels
    adjusted_predictions = predicted_labels.copy()
    
    # Find all true anomaly segments
    anomaly_segments = np.where(true_labels == 1)[0]
    
    if len(anomaly_segments) == 0:
        return adjusted_predictions
    
    # Find continuous anomaly segments
    segments = np.split(anomaly_segments, np.where(np.diff(anomaly_segments) != 1)[0] + 1)
    
    for segment in segments:
        # Calculate the proportion of correct predictions in this segment
        correct_predictions = np.sum(predicted_labels[segment] == 1)
        correct_ratio = correct_predictions / len(segment)
        
        # If the proportion of correct predictions is greater than or equal to k%, set all prediction labels in this segment to 1
        if correct_ratio >= k / 100:
            adjusted_predictions[segment] = 1
    
    return adjusted_predictions


def anomaly_adjustment(anomaly, mse_drop_all, mse, variate, thresh=3, drop=1):
    mse_decrease = dict.fromkeys(range(drop), [])
    for key in mse_drop_all:
        mse_drop_all[key] = np.array(mse_drop_all[key])
        mse_drop_all[key] = mse_drop_all[key].reshape(-1, variate)
        mse_decrease[key] = mse / mse_drop_all[key]
    k_value = []
    for ind in range(variate):
        for i in range(len(anomaly)):
            if anomaly[i] == 1:
                min_d = -1
                max_decrease = thresh
                for d in range(drop):
                    if mse_decrease[d][i, ind] > max_decrease:
                        mse[i, ind] = mse_drop_all[d][i, ind]
                        max_decrease = mse_decrease[d][i, ind]
                        min_d = d
                    else:
                        continue
                k_value.append(min_d)
    return mse, np.array(k_value)

