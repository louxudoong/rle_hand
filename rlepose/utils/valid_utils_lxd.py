import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

def compute_RMSE(kpts_pred, kpts_t):
    # 求解每个关键点的坐标差异的平方和
    squared_errors = np.sum(np.square(np.array(kpts_t) - np.array(kpts_pred)), axis=0)

    # 计算均方根误差
    RMSE = np.sqrt(np.mean(squared_errors))

    return RMSE

# 要求输入为n * 2的kpts
def calculate_oks_pt2(kpts_pred, kpts_true):

    assert kpts_pred.shape == kpts_true.shape, "Keypoints shape mismatch"
    oks = []
    K = kpts_pred.shape[0]   # joints_num
    max_x = np.max(kpts_pred[:, 0])
    min_x = np.min(kpts_pred[:, 0])
    max_y = np.max(kpts_pred[:, 1])
    min_y = np.min(kpts_pred[:, 1])
    
    s2 = (max_x - min_x)**2 + (max_y - min_y)**2
    k2 = 1
    exp_term = np.exp(-np.sum((kpts_pred - kpts_true)**2, axis=1) / (2 * s2 * k2))
    oks = np.sum(exp_term) / K

    # 以下为适应batch_size * joints_num * 2的输入的内容
    # oks = []
    # N = kpts_pred.shape[0]   # batch_size
    # K = kpts_pred.shape[1]   # 关键点数量

    # for i in range(N):
    #     # 根据kpt max/min 计算尺度因子s
    #     max_x = np.max(kpts_pred[i, :, 0])
    #     min_x = np.min(kpts_pred[i, :, 0])
    #     max_y = np.max(kpts_pred[i, :, 1])
    #     min_y = np.min(kpts_pred[i, :, 1])
    #     s2 = (max_x - min_x)**2 + (max_y - min_y)**2
    #     k2 = 1
    #     exp_term_i = np.exp(
    #         -np.sum((kpts_pred[i] - kpts_true[i])**2, axis=1) / (2 * s2 * k2)
    #         )
    #     oks_i = np.sum(exp_term_i) / K
    #     oks.append(oks_i)
    
    return oks


def calculate_mAP(oks_list):
    '''
    输入要求为index * 1 的oks list
    输出单帧图像对应的不同阈值下的mAP info_str
    '''
    thresholds = np.arange(0.5, 1.0, 0.05)
    num_thresholds = len(thresholds)

    average_precision = np.zeros(num_thresholds)

    # stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)',
    #             'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']
    stats_names = ['Ap .5', 'AP .55', 'AP .60', 'AP .65',
            'Ap .70', 'AP .75', 'AP .80', 'AP .85', 'Ap .90', 'AP .95']
    info_str = {}

    for ind, name in enumerate(stats_names):
        threshold = thresholds[ind]
        true_positives = np.sum(oks_list >= threshold)
        false_positives = np.sum(oks_list < threshold)
        false_negatives = np.sum(oks_list >= threshold)

        precision = true_positives / (true_positives + false_positives)
        info_str[name] = precision
        average_precision[ind] = precision
    
    info_str['mAP'] = np.mean(average_precision)

    return info_str

def cal_mAP_RMSE(kpts_pred, kpts_t):
    '''
    要求输入的kpts_pred与kpts_t均为joint_num * 2的np array
    输出为单帧图像的mAP与RMSE
    mAP_info_str
    '''
    # kpts_pred = kpts_pred.cpu().detach().reshape(kpts_pred.shape[0], 21, 2)
    # kpts_t = kpts_t.cpu().detach().reshape(kpts_pred.shape[0], 21, 2)
    
    RMSE = compute_RMSE(kpts_pred, kpts_t)
    oks = calculate_oks_pt2(np.array(kpts_pred), np.array(kpts_t))
    mAP_info_str = calculate_mAP(oks)

    return mAP_info_str, RMSE

def paint(im, kpts):
    '''
    输入为单帧的im与其对应的一组kpts
    '''
    
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], \
          [128, 0, 128], [255, 192, 203], [0, 128, 128], [128, 128, 0], [128, 0, 0], [0, 128, 0], [0, 0, 128]]
    limbSeq = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]

    # im = cv2.imread(img_path)
    # draw points
    for k in kpts:
        x = int(k[0])
        y = int(k[1])
        # print(f'im .shape = {im.shape}')
        # print(f'x={x}, y={y}')
        cv2.circle(im, (x, y), 2, (0, 0, 255), -1)

    # draw lines
    # print("kpts shape: ", kpts)
    for i in range(len(limbSeq)):
        cur_im = im.copy()
        limb = limbSeq[i]
        # print(kpts[limb[0]])
        [Y0, X0] = kpts[limb[0]]
        [Y1, X1] = kpts[limb[1]]
        mX = np.mean([X0, X1])
        mY = np.mean([Y0, Y1])
        length = ((X0 - X1) ** 2 + (Y0 - Y1) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X0 - X1, Y0 - Y1))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), 4), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_im, polygon, colors[i])
        im = cv2.addWeighted(im, 0.4, cur_im, 0.6, 0)

    return im