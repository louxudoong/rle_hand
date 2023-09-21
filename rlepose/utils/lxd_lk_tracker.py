from __future__ import print_function

import imutils
import numpy as np
import cv2
import os
from pathlib import Path
import time


lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=500,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)


def draw_str(dst, target, s):
    """
    put text on the destination image.
    :param dst: image
    :param target: coordinates
    :param s:  strings
    :return:
    """
    x, y = target
    cv2.putText(dst, s, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)


def LK_Tracker(frame, count, last_tracks, prev_gray = None, track_len = 10, detect_interval = 5):

    motion = None
    p1 = None
    p0 = None

    frame_width = int(frame.shape[1])
    frame_height = int(frame.shape[0])
    size = (frame_width, frame_height)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vis = frame.copy()

    # skip first frame
    if len(last_tracks) > 0:
            img0, img1 = prev_gray, frame_gray
            p0 = np.float32([tr[-1] for tr in last_tracks]).reshape(-1, 1, 2)

            # calculate nextPts
            p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)

            motion = p1 - p0

            # re-calculate prevPts
            p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)

            # select good tracker for the nextPos
            d = abs(p0 - p0r).reshape(-1, 2).max(-1) # 逆向光流，剔除错误点
            good = d < 1
            new_tracks = []
            for tr, (x, y), good_flag in zip(last_tracks, p1.reshape(-1, 2), good):
                if not good_flag: # 跳过标签不为good的点
                    continue
                tr.append((x, y)) # 在轨迹中加入当前帧预测的关键点
                if len(tr) > track_len: # 超出队列长度，就踢出最早的点
                    del tr[0]
                new_tracks.append(tr)
                # draw filled red circle in origin image with radius of 5 px
                cv2.circle(vis, (int(x), int(y)), 5, (0, 0, 255), -1)
            last_tracks = new_tracks

            cv2.polylines(vis, [np.int32(tr) for tr in last_tracks], False, (0, 255, 0), 2)
            draw_str(vis, (20, 20), 'track count: %d' % len(last_tracks))

        # re-detect keypoints in every detect_interval frame
    if count % detect_interval == 0:
        mask = np.zeros_like(frame_gray)
        mask[:] = 255
        for x, y in [np.int32(tr[-1]) for tr in last_tracks]:
            # draw a filled circle on the mask with a radius of 5 px
            cv2.circle(mask, (x, y), 5, 0, -1)

        p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                last_tracks.append([(x, y)])

    # update frame
    prev_gray = frame_gray

    return vis, prev_gray, last_tracks, motion, p1, p0


def improved_LK_Tracker(frame, count, last_tracks, prev_gray=None, track_len=30, max_joints_num = 30, least_joints_num = 15, detect_interval=30, dist_thresh = 200):
    '''
    加入了追踪点队列维持策略，并根据到bbox的中心的距离筛选shi-tomas角点；
    根据distance剔除点可能是多余的，在角点检测时进行；
    只维护joints_num个点
    '''
    '''
    track_len: 某一关键点最长的追踪帧数
    joints_num: 维持的关键点数量
    '''
    frame_width = int(frame.shape[1])
    frame_height = int(frame.shape[0])
    size = (frame_width, frame_height)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vis = frame.copy()

    # Shi-Tomasi角点检测参数
    feature_params = dict(maxCorners=25, qualityLevel=0.3, minDistance=7, blockSize=7)
    
    # skip first frame
    if len(last_tracks) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([tr[-1] for tr in last_tracks]).reshape(-1, 1, 2)

        # 计算下一帧的角点位置
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, winSize=(15, 15), maxLevel=2,
                                                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # 重新计算前一帧的角点位置
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, winSize=(15, 15), maxLevel=2,
                                                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        # d = np.linalg.norm(p0 - p0r, axis=-1)
        good = d < 1
        new_tracks = []

        # tr是track_len * 2的list, tracks则是joints_num * track_len * 2的list
        for tr, (x, y), good_flag in zip(last_tracks, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            tr.append((x, y))
            if len(tr) > track_len: # 保持最长追踪帧数
                del tr[0]
            new_tracks.append(tr)
            # 在原始图像上绘制填充的红色圆形，半径为5像素
            cv2.circle(vis, (int(x), int(y)), 5, (0, 0, 255), -1)
        last_tracks = new_tracks

        # print("last_tracks shape = ", np.array(last_tracks).shape)
        cv2.polylines(vis, [np.int32(tr) for tr in last_tracks], False, (0, 255, 0), 2)
        draw_str(vis, (20, 20), 'track count: %d' % len(last_tracks))

    # re-detect keypoints in every detect_interval frame
    if (count % detect_interval == 0) | (len(last_tracks) < least_joints_num): # 在指定间隔帧或队列内点过少，重新进行角点检测
        mask = np.zeros_like(frame_gray)
        mask[:] = 255
        for x, y in [np.int32(tr[-1]) for tr in last_tracks]:
            # draw a filled circle on the mask with a radius of 5 px
            cv2.circle(mask, (x, y), 5, 0, -1)

        bbox_center = (frame_width // 2, frame_height // 2)

        p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
        print(count)
        print("*****************************")
        print("init p shape:", np.array(p).shape)
        if p is not None:
            dist_list = []
            for x, y in np.float32(p).reshape(-1, 2):
                dist = np.sqrt((x - bbox_center[0]) ** 2 + (y - bbox_center[1]) ** 2)
                dist_list.append(dist)

        for i, (x, y) in enumerate(np.float32(p).reshape(-1, 2)):
            dist = dist_list[i]
            if dist <= dist_thresh:
                if len(last_tracks) < max_joints_num:
                    last_tracks.append([(x, y)])
                else: # 候选列表中点多于joints_num，则只保留最近的
                    max_dist_index = max(range(len(last_tracks)), key=lambda i: np.sqrt((last_tracks[i][-1][0] - bbox_center[0]) ** 2 + (last_tracks[i][-1][1] - bbox_center[1]) ** 2))
                    max_dist = np.sqrt((last_tracks[max_dist_index][-1][0] - bbox_center[0]) ** 2 + (last_tracks[max_dist_index][-1][1] - bbox_center[1]) ** 2)
                
                    if dist < max_dist:
                        last_tracks[max_dist_index].append((x, y))
        
        print("*************")
        print("init last_tracks shape:", np.array(last_tracks).shape)

    # 更新帧
    prev_gray = frame_gray

    return vis, prev_gray, last_tracks


# def improved_LK_Tracker(frame, count, last_tracks, prev_gray=None, track_len=10, detect_interval=30):
#     # 计算图像灰度
#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     vis = frame.copy()

#     # 追踪点队列维持策略
#     if len(last_tracks) > 0:
#         # 光流跟踪
#         p0, p1, good = optical_flow_tracking(prev_gray, frame_gray, last_tracks)

#         # 更新追踪点队列
#         last_tracks = update_tracks(last_tracks, p1, good)
        
#         # 可视化追踪结果
#         draw_tracks(vis, last_tracks)

#     # 角点检测与追踪点队列更新
#     if (count % detect_interval == 0) or (len(last_tracks) < 20):
#         # 进行角点检测
#         p = detect_keypoints(frame_gray, last_tracks, feature_params)

#         # 更新追踪点队列
#         last_tracks = update_tracks(last_tracks, p)

#     # 更新帧
#     prev_gray = frame_gray

#     return vis, prev_gray, last_tracks


# def optical_flow_tracking(prev_gray, frame_gray, last_tracks):
#     p0 = np.float32([tr[-1] for tr in last_tracks]).reshape(-1, 1, 2)
#     p1, _st, _err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, winSize=(15, 15), maxLevel=2,
#                                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#     p0r, _st, _err = cv2.calcOpticalFlowPyrLK(frame_gray, prev_gray, p1, None, winSize=(15, 15), maxLevel=2,
#                                               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#     d = np.linalg.norm(p0 - p0r, axis=-1)
#     good = d < 1
#     return p0, p1, good


# def update_tracks(last_tracks, new_points, good=None, track_len=30):
#     new_tracks = []
#     for i, (tr, (x, y)) in enumerate(zip(last_tracks, new_points.reshape(-1, 2))):
#         if good is not None and not good[i]:
#             continue
#         tr.append((x, y))
#         if len(tr) > track_len:
#             del tr[0]
#         new_tracks.append(tr)
#     return new_tracks


# def detect_keypoints(frame_gray, last_tracks, feature_params):
#     mask = np.zeros_like(frame_gray)
#     mask[:] = 255
#     p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
#     if p is not None:
#         for x, y in np.float32(p).reshape(-1, 2):
#             last_tracks.append([(x, y)])
#     return p


# def draw_tracks(vis, tracks):
#     cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0), 2)
#     draw_str(vis, (20, 20), 'track count: %d' % len(tracks))