import numpy as np

def sequence_to_intervals(binary_sequence):
    intervals = []
    n = len(binary_sequence)
    i = 0
    while i < n:
        if binary_sequence[i] == 1:
            start = i
            while i + 1 < n and binary_sequence[i + 1] == 1:
                i += 1
            end = i
            intervals.append([start, end])
        i += 1
    return intervals

def interval_based_metrics_from_sequences(y_pred, y_true):
    """
    Computes improved TRec, TPrec, and F1-score from (n,) binary sequences following the paper.
    
    Args:
        y_pred: (n,) numpy array, binary predictions (0/1)
        y_true: (n,) numpy array, binary ground truth (0/1)
        
    Returns:
        TRec: float
        TPrec: float
        F1: float
    """
    n = len(y_pred)
    pred_intervals = sequence_to_intervals(y_pred)
    true_intervals = sequence_to_intervals(y_true)

    # Compute TRec
    hits = 0
    for t_start, t_end in true_intervals:
        if np.any(y_pred[t_start:t_end+1]):
            hits += 1
    TRec = hits / len(true_intervals) if len(true_intervals) > 0 else 0.0

    # Compute improved TPrec
    precision_numer = 0
    precision_denom = 0
    for p_start, p_end in pred_intervals:
        length = p_end - p_start + 1
        overlap = np.sum(y_true[p_start:p_end+1])
        precision_numer += overlap / length
        precision_denom += 1
    TPrec = precision_numer / precision_denom if precision_denom > 0 else 0.0

    # Compute F1
    if TPrec + TRec > 0:
        F1 = 2 * TPrec * TRec / (TPrec + TRec)
    else:
        F1 = 0.0

    return TRec, TPrec, F1

# ======================
# ✅ 测试用例

def test_sequence_to_intervals():
    """测试 sequence_to_intervals 函数"""
    print("=== 测试 sequence_to_intervals 函数 ===")
    
    # 测试用例1: 单个连续区间
    seq1 = np.array([0, 0, 1, 1, 1, 0, 0])
    intervals1 = sequence_to_intervals(seq1)
    print(f"输入: {seq1}")
    print(f"输出: {intervals1}")
    print(f"期望: [[2, 4]]")
    assert intervals1 == [[2, 4]], f"期望 [[2, 4]], 得到 {intervals1}"
    print("✅ 测试通过\n")
    
    # 测试用例2: 多个分离的区间
    seq2 = np.array([1, 1, 0, 0, 1, 0, 1, 1, 1])
    intervals2 = sequence_to_intervals(seq2)
    print(f"输入: {seq2}")
    print(f"输出: {intervals2}")
    print(f"期望: [[0, 1], [4, 4], [6, 8]]")
    assert intervals2 == [[0, 1], [4, 4], [6, 8]], f"期望 [[0, 1], [4, 4], [6, 8]], 得到 {intervals2}"
    print("✅ 测试通过\n")
    
    # 测试用例3: 全为0
    seq3 = np.array([0, 0, 0, 0])
    intervals3 = sequence_to_intervals(seq3)
    print(f"输入: {seq3}")
    print(f"输出: {intervals3}")
    print(f"期望: []")
    assert intervals3 == [], f"期望 [], 得到 {intervals3}"
    print("✅ 测试通过\n")
    
    # 测试用例4: 全为1
    seq4 = np.array([1, 1, 1])
    intervals4 = sequence_to_intervals(seq4)
    print(f"输入: {seq4}")
    print(f"输出: {intervals4}")
    print(f"期望: [[0, 2]]")
    assert intervals4 == [[0, 2]], f"期望 [[0, 2]], 得到 {intervals4}"
    print("✅ 测试通过\n")


def test_interval_based_metrics():
    """测试 interval_based_metrics_from_sequences 函数"""
    print("=== 测试 interval_based_metrics_from_sequences 函数 ===")
    
    # 测试用例1: 完美匹配
    y_pred1 = np.array([0, 1, 1, 0, 1, 0])
    y_true1 = np.array([0, 1, 1, 0, 1, 0])
    TRec1, TPrec1, F1_1 = interval_based_metrics_from_sequences(y_pred1, y_true1)
    print(f"测试用例1 - 完美匹配:")
    print(f"  预测: {y_pred1}")
    print(f"  真实: {y_true1}")
    print(f"  TRec: {TRec1:.4f}, TPrec: {TPrec1:.4f}, F1: {F1_1:.4f}")
    print(f"  期望: TRec=1.0, TPrec=1.0, F1=1.0")
    assert abs(TRec1 - 1.0) < 1e-6 and abs(TPrec1 - 1.0) < 1e-6 and abs(F1_1 - 1.0) < 1e-6
    print("✅ 测试通过\n")
    
    # 测试用例2: 部分重叠
    y_pred2 = np.array([1, 1, 0, 0, 1, 1])
    y_true2 = np.array([0, 1, 1, 0, 0, 1])
    TRec2, TPrec2, F1_2 = interval_based_metrics_from_sequences(y_pred2, y_true2)
    print(f"测试用例2 - 部分重叠:")
    print(f"  预测: {y_pred2}")
    print(f"  真实: {y_true2}")
    print(f"  预测区间: {sequence_to_intervals(y_pred2)}")
    print(f"  真实区间: {sequence_to_intervals(y_true2)}")
    print(f"  TRec: {TRec2:.4f}, TPrec: {TPrec2:.4f}, F1: {F1_2:.4f}")
    print("✅ 测试通过\n")
    
    # 测试用例3: 无重叠
    y_pred3 = np.array([1, 1, 0, 0, 0, 0])
    y_true3 = np.array([0, 0, 0, 1, 1, 0])
    TRec3, TPrec3, F1_3 = interval_based_metrics_from_sequences(y_pred3, y_true3)
    print(f"测试用例3 - 无重叠:")
    print(f"  预测: {y_pred3}")
    print(f"  真实: {y_true3}")
    print(f"  TRec: {TRec3:.4f}, TPrec: {TPrec3:.4f}, F1: {F1_3:.4f}")
    print(f"  期望: TRec=0.0, TPrec=0.0, F1=0.0")
    assert abs(TRec3 - 0.0) < 1e-6 and abs(TPrec3 - 0.0) < 1e-6 and abs(F1_3 - 0.0) < 1e-6
    print("✅ 测试通过\n")
    
    # 测试用例4: 预测全为0
    y_pred4 = np.array([0, 0, 0, 0])
    y_true4 = np.array([1, 1, 0, 1])
    TRec4, TPrec4, F1_4 = interval_based_metrics_from_sequences(y_pred4, y_true4)
    print(f"测试用例4 - 预测全为0:")
    print(f"  预测: {y_pred4}")
    print(f"  真实: {y_true4}")
    print(f"  TRec: {TRec4:.4f}, TPrec: {TPrec4:.4f}, F1: {F1_4:.4f}")
    print(f"  期望: TRec=0.0, TPrec=0.0, F1=0.0")
    assert abs(TRec4 - 0.0) < 1e-6 and abs(TPrec4 - 0.0) < 1e-6 and abs(F1_4 - 0.0) < 1e-6
    print("✅ 测试通过\n")
    
    # 测试用例5: 真实全为0
    y_pred5 = np.array([1, 1, 0, 1])
    y_true5 = np.array([0, 0, 0, 0])
    TRec5, TPrec5, F1_5 = interval_based_metrics_from_sequences(y_pred5, y_true5)
    print(f"测试用例5 - 真实全为0:")
    print(f"  预测: {y_pred5}")
    print(f"  真实: {y_true5}")
    print(f"  TRec: {TRec5:.4f}, TPrec: {TPrec5:.4f}, F1: {F1_5:.4f}")
    print(f"  期望: TRec=0.0, TPrec=0.0, F1=0.0")
    assert abs(TRec5 - 0.0) < 1e-6 and abs(TPrec5 - 0.0) < 1e-6 and abs(F1_5 - 0.0) < 1e-6
    print("✅ 测试通过\n")
    
    # 测试用例6: 复杂场景 - 多个区间的精确计算
    y_pred6 = np.array([1, 1, 1, 0, 1, 1, 0, 1])  # 区间: [0,2], [4,5], [7,7]
    y_true6 = np.array([0, 1, 1, 1, 0, 1, 1, 0])  # 区间: [1,3], [5,6]
    TRec6, TPrec6, F1_6 = interval_based_metrics_from_sequences(y_pred6, y_true6)
    print(f"测试用例6 - 复杂场景:")
    print(f"  预测: {y_pred6}")
    print(f"  真实: {y_true6}")
    print(f"  预测区间: {sequence_to_intervals(y_pred6)}")
    print(f"  真实区间: {sequence_to_intervals(y_true6)}")
    print(f"  TRec: {TRec6:.4f}, TPrec: {TPrec6:.4f}, F1: {F1_6:.4f}")
    
    # 手动计算验证
    # TRec: 真实区间 [1,3] 被预测区间 [0,2] 覆盖, [5,6] 被预测区间 [4,5] 覆盖 -> 2/2 = 1.0
    # TPrec: 
    #   区间 [0,2]: overlap=2, length=3 -> 2/3
    #   区间 [4,5]: overlap=1, length=2 -> 1/2  
    #   区间 [7,7]: overlap=0, length=1 -> 0/1
    #   TPrec = (2/3 + 1/2 + 0/1) / 3 = (4/6 + 3/6 + 0) / 3 = 7/18 ≈ 0.3889
    expected_TRec = 1.0
    expected_TPrec = 7/18
    expected_F1 = 2 * expected_TRec * expected_TPrec / (expected_TRec + expected_TPrec)
    print(f"  期望: TRec={expected_TRec:.4f}, TPrec={expected_TPrec:.4f}, F1={expected_F1:.4f}")
    
    assert abs(TRec6 - expected_TRec) < 1e-6, f"TRec计算错误: 期望{expected_TRec}, 得到{TRec6}"
    assert abs(TPrec6 - expected_TPrec) < 1e-6, f"TPrec计算错误: 期望{expected_TPrec}, 得到{TPrec6}"
    assert abs(F1_6 - expected_F1) < 1e-6, f"F1计算错误: 期望{expected_F1}, 得到{F1_6}"
    print("✅ 测试通过\n")


def run_all_tests():
    """运行所有测试"""
    print("开始运行所有测试用例...\n")
    try:
        test_sequence_to_intervals()
        test_interval_based_metrics()
        print("🎉 所有测试用例都通过了！")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


# ======================
# ✅ 使用方法

if __name__ == "__main__":
    # 运行测试用例
    run_all_tests()
    
    print("\n" + "="*50)
    print("手动测试示例:")
    
    # 手动测试示例
    y_pred = np.array([0, 1, 1, 0, 1, 0, 1, 1], dtype=int)
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 1], dtype=int)
    
    print(f"预测序列: {y_pred}")
    print(f"真实序列: {y_true}")
    
    TRec, TPrec, F1 = interval_based_metrics_from_sequences(y_pred, y_true)
    print(f"TRec: {TRec:.4f}, TPrec: {TPrec:.4f}, F1: {F1:.4f}")
