import numpy as np
import matplotlib.pyplot as plt

def compute_RMSE(kpts_pred, kpts_t):
    # 求解每个关键点的坐标差异的平方和
    squared_errors = np.sum(np.square(np.array(kpts_t) - np.array(kpts_pred)), axis=1)

    # 计算均方根误差
    RMSE = np.sqrt(np.mean(squared_errors))

    return RMSE

# 要求输入为size * n * 2的kpts
def calculate_oks_pt2(kpts_pred, kpts_true):

    assert kpts_pred.shape == kpts_true.shape, "Keypoints shape mismatch"
    
    oks = []
    N = kpts_pred.shape[0]   # batch_size
    K = kpts_pred.shape[1]   # 关键点数量

    for i in range(N):
        # 根据kpt max/min 计算尺度因子s
        max_x = np.max(kpts_pred[i, :, 0])
        min_x = np.min(kpts_pred[i, :, 0])
        max_y = np.max(kpts_pred[i, :, 1])
        min_y = np.min(kpts_pred[i, :, 1])
        s2 = (max_x - min_x)**2 + (max_y - min_y)**2
        k2 = 1
        exp_term_i = np.exp(
            -np.sum((kpts_pred[i] - kpts_true[i])**2, axis=1) / (2 * s2 * k2)
            )
        oks_i = np.sum(exp_term_i) / K
        oks.append(oks_i)
    
    return oks


def calculate_mAP(oks):
    thresholds = np.arange(0.5, 1.0, 0.05)
    num_thresholds = len(thresholds)

    average_precision = np.zeros(num_thresholds)

    stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)',
                'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']
    info_str = {}

    for ind, name in enumerate(stats_names):
        threshold = thresholds[ind]
        true_positives = np.sum(oks >= threshold)
        false_positives = np.sum(oks < threshold)
        false_negatives = np.sum(oks >= threshold)

        precision = true_positives / (true_positives + false_positives)
        info_str[name] = precision

    return info_str

def cal_mAP_RMSE(kpts_pred, kpts_t):
    kpts_pred = kpts_pred.cpu().detach().reshape(kpts_pred.shape[0], 21, 2)
    kpts_t = kpts_t.cpu().detach().reshape(kpts_pred.shape[0], 21, 2)
    
    RMSE = compute_RMSE(kpts_pred, kpts_t)
    oks = calculate_oks_pt2(np.array(kpts_pred), np.array(kpts_t))
    mAP_info_str = calculate_mAP(oks)

    return mAP_info_str, RMSE