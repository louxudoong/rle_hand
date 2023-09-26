'''
使用yolov3作为hand detector, 并使用rle实现keypoints detection
加入了LK光流进行加速
'''
from rlepose.utils.valid_utils_lxd import paint
from rlepose.utils.bbox import _box_to_center_scale, _center_scale_to_box
import torch
import cv2
import numpy as np
import torch
from rlepose.models import builder
from rlepose.utils.config import update_config
from rlepose.utils.transforms import im_to_torch
import time

import os
from yolo_v3.utils.datasets import *
from yolo_v3.utils.utils import *
from yolo_v3.utils.parse_config import parse_data_cfg
from yolo_v3.yolov3 import Yolov3, Yolov3Tiny
from yolo_v3.utils.torch_utils import select_device

from rlepose.utils.lxd_lk_tracker import LK_Tracker, filter_background_motion

def process_data(img, img_size=416):# 图像预处理
    img, _, _, _ = letterbox(img, height=img_size)
    # Normalize RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    return img

def show_model_param(model):
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        print("该层的结构: {}, 参数和: {}".format(str(list(i.size())), str(l)))
        k = k + l
    print("----------------------")
    print("总参数数量和: " + str(k))

def refine_hand_bbox(bbox,img_shape):
    height,width,_ = img_shape

    x1,y1,x2,y2 = bbox

    expand_w = (x2-x1)
    expand_h = (y2-y1)

    x1 -= expand_w*0.06
    y1 -= expand_h*0.1
    x2 += expand_w*0.06
    y2 += expand_h*0.1

    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

    x1 = int(max(0,x1))
    y1 = int(max(0,y1))
    x2 = int(min(x2,width-1))
    y2 = int(min(y2,height-1))

    return (x1,y1,x2,y2)


def get_3rd_point(a, b):
    """Return vector c that perpendicular to (a - b)."""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_dir(src_point, rot_rad):
    """Rotate the point by `rot_rad` degree."""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0,
                         align=False):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def yolo_detect(
        rle_cfg,
        rle_weights,
        yolo_model_path,
        yolo_cfg,
        yolo_data_cfg,
        yolo_img_size=416,
        yolo_conf_thres=0.5,
        yolo_nms_thres=0.5,
        yolo_video_path = 0
):
    device = select_device()
    use_cuda = torch.cuda.is_available() 
    rle_cfg = update_config(rle_cfg)
    rle_input_size = rle_cfg.DATA_PRESET['IMAGE_SIZE'] # 256 * 192, 要输入到rle model中的图像的size

    yolo_pre_time = []
    rle_pre_time = []
    
    # 1. build rle model and yolo model
    time_0 = time.time()
    # rle
    rle_module = builder.build_sppe(rle_cfg.MODEL, preset_cfg=rle_cfg.DATA_PRESET)  # 根据cfg的配置信息构建模型

    rle_module.load_state_dict(torch.load(rle_weights, map_location='cpu'), 
                    strict=True)  # 加载权重
    
    rle_module.to(device).eval()#模型模式设置为 eval

    # yolo
    # print("****************************************************************")
    # print(os.path.splitext(yolo_data_cfg)[0] + ".names")
    # # print(parse_data_cfg(yolo_data_cfg)['names'])

    classes = load_classes(os.path.splitext(yolo_data_cfg)[0] + ".names")
    # classes = load_classes(parse_data_cfg(yolo_data_cfg)['names'])
    #print("****************************************************************")
    num_classes = len(classes)

    # Initialize model
    yolo_weights = yolo_model_path
    if "-tiny" in yolo_cfg:
        a_scalse = 416./yolo_img_size
        anchors=[(10, 14), (23, 27), (37, 58), (81, 82), (135, 169), (344, 319)]
        anchors_new = [ (int(anchors[j][0]/a_scalse),int(anchors[j][1]/a_scalse)) for j in range(len(anchors)) ]

        yolo_model = Yolov3Tiny(num_classes,anchors = anchors_new)

    else:
        a_scalse = 416./yolo_img_size
        anchors=[(10,13), (16,30), (33,23), (30,61), (62,45), (59,119), (116,90), (156,198), (373,326)]
        anchors_new = [ (int(anchors[j][0]/a_scalse),int(anchors[j][1]/a_scalse)) for j in range(len(anchors)) ]
        yolo_model = Yolov3(num_classes,anchors = anchors_new)

    # show_model_param(model)# 显示模型参数

    # Load weights
    if os.access(yolo_weights, os.F_OK):# 判断模型文件是否存在
        yolo_model.load_state_dict(torch.load(yolo_weights, map_location=device)['model'])
    else:
        print('error model not exists')
        return False
    yolo_model.to(device).eval()#模型模式设置为 eval

    time_1 = time.time()  # 记录开始时间
    time_loadweights = time_1 - time_0  # 计算耗时，单位为秒
    print("build model and load weights cost time: ", time_loadweights)


    # 创建摄像头对象
    cap = cv2.VideoCapture(0)

    max_LK_count = 1000 # LK连续预测的最大帧数
    least_track_num = 5 # 保持追踪的最小角点数
    motion_threshold = 2
    avg_motion = np.zeros(2)  # 平均光流向量


    frame_count = 0 # 摄像头输入图像的帧数
    LK_count = 0 # LK连续预测的帧数
    prev_track_num = 0 # 上一帧追踪的角点数
    last_tracks = []
    track_prev_gray = None
    lk_pre_bbox = [] # 包含上一帧的bbox
    cur_bbox = None # 包含当前帧的bbox，用于计算rle和LK
    track_vis = None
    initial_bbox = None
    last_motion = None

    while True:
        # 读取摄像头捕获的一帧图像
        ret, frame = cap.read()
        # 如果成功读取图像，则进行处理
        if ret:
            display_img = frame.copy()
            # 启动yolo的3个条件：是初始帧；或追踪的角点数少于阈值；或LK连续运行帧数超出阈值
            if (frame_count == 0) | (prev_track_num < least_track_num) | (LK_count > max_LK_count):
                print(f"yolo on, for prev_track_num = {prev_track_num}, LK_count = {LK_count}")

                # 2. 使用yolo读取图像，并输出hand bbox
                img = process_data(display_img, yolo_img_size) # 把图像调整为3 * 416 * 416
                if use_cuda:
                    torch.cuda.synchronize()
            
                img = torch.from_numpy(img).unsqueeze(0).to(device)
                time1 = time.time()
                pred, _ = yolo_model(img) #yolo prediction, 注意，这里输入模型的为
                time2 = time.time()
                yolo_cost = (time2 - time1) * 1000
                # print("yolo pre cost: ", yolo_cost)
                if (yolo_cost < 20):
                    yolo_pre_time.append(yolo_cost)

                if use_cuda:
                    torch.cuda.synchronize()

                detections = non_max_suppression(pred, yolo_conf_thres, yolo_nms_thres)[0] # nms

                if use_cuda:
                    torch.cuda.synchronize()

                if detections is None or len(detections) == 0:
                    cv2.namedWindow('hand detect',0)
                    cv2.imshow("hand detect", display_img)
                    key = cv2.waitKey(1)
                    
                    if key == 27:
                        break
                    continue

                detections[:, :4] = scale_coords(yolo_img_size, detections[:, :4], frame.shape).round() # 将yolo的bbox转换到rle_input的size

                result = []
                for res in detections:
                    result.append((classes[int(res[-1])], float(res[4]), [int(res[0]), int(res[1]), int(res[2]), int(res[3])]))

                if use_cuda:
                    torch.cuda.synchronize()

            
                lxd_output = [] # hand_nums * 5

                # Draw bounding boxes and labels of detections
                for *xyxy, conf, cls_conf, cls in detections:
                    label = '%s %.2f' % (classes[int(cls)], conf)
                    # label = '%s' % (classes[int(cls)])

                    # print(conf, cls_conf)
                    # xyxy = refine_hand_bbox(xyxy,im0.shape)
                    xyxy = int(xyxy[0]),int(xyxy[1])+6,int(xyxy[2]),int(xyxy[3])
                    # outputi = [int(xyxy[0]), int(xyxy[1])+6, int(xyxy[2]), int(xyxy[3]), conf] 不想要置信度了
                    lxd_output.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])

                    if int(cls) == 0:
                        plot_one_box(xyxy, display_img, label=label, color=(15,255,95),line_thickness = 3)
                    else:
                        plot_one_box(xyxy, display_img, label=label, color=(15,155,255),line_thickness = 3)

                # print(lxd_output)

                cv2.namedWindow('hand detect')
                cv2.imshow("hand detect", display_img)

                # str_fps = ("{:.2f} Fps".format(1./(s2 - t+0.00001)))
                # cv2.putText(init_img, str_fps, (5,init_img.shape[0]-3),cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 0, 255),4)
                # cv2.putText(init_img, str_fps, (5,init_img.shape[0]-3),cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 0),1)

                # cv2.namedWindow('yolo',0)
                # cv2.imshow("yolo", display_img)

                # 3. predict hand keypoints from rle
                # if lxd_output[4] 

                # 每一次调用yolo，都需要进行initial_bbox、avg_motion、LK_count的初始化
                cur_bbox = lxd_output
                initial_bbox = cur_bbox
                LK_count = 0
                avg_motion = np.zeros(2)

            else: # 不是第一帧的话，那我们就拥有历史信息：
                # 由于这次没有yolo给bbox了，如何给cur_bbox赋值呢？
                # 答案是根据last motion和last bbox来计算cur_bbox
                print("LK on")
                if use_cuda:
                    torch.cuda.synchronize()
                cur_bbox = lk_pre_bbox


                for lk_box in lk_pre_bbox:

                    plot_one_box(lk_box, display_img, color=(0,0,255),line_thickness = 3)

                cv2.namedWindow('hand detect')
                cv2.imshow("hand detect", display_img)
                LK_count += 1


            if 1 :
                # 注意，rle的所有输出统一到rle_input_size下
                lk_bbox = []
                for hand_num in range(len(cur_bbox)):
                    inps = frame.copy()
                    # 根据bbox计算inps
                    bbox = cur_bbox[hand_num][:4]
                    xmin, ymin, xmax, ymax = bbox
                    aspect_ratio = float(rle_input_size[1]) / rle_input_size[0]  # w / h
                    center, scale = _box_to_center_scale(xmin, ymin, xmax - xmin, ymax - ymin, aspect_ratio, scale_mult=1.25)
                    inp_h, inp_w = rle_input_size # camera input size
                    trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
                    # inps = inps.detach().numpy()
                    inps = cv2.warpAffine(inps, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
                    cv2.namedWindow(f"trans_img_{hand_num}")
                    cv2.imshow(f"trans_img_{hand_num}", inps)

                    # add LK-Tracker
                    roi_w = xmax - xmin
                    roi_h = ymax - ymin
                    track_vis, track_prev_gray, last_tracks, motion, p1, p0 = LK_Tracker(inps, frame_count, last_tracks, track_prev_gray)
                    prev_track_num = len(last_tracks)
                    
                    cv2.namedWindow(f"LK_img_{hand_num}")
                    cv2.imshow(f"LK_img_{hand_num}", track_vis)
                    if motion is not None:
                        # ("motion.shape = ", motion.shape) joints_num * 1 * 2
                        # motionxy = [- int(motion[:, :, 0].mean()), - int(motion[:, :, 1].mean())]

                        frame_width = frame.shape[1]
                        frame_height = frame.shape[0]
                        # next_bbox = (int(max(x_topleft, 0)), 
                        #              int(max(y_topleft, 0)), 
                        #              int(min(x_topleft + roi_w, frame_width)), 
                        #              int(min(y_topleft + roi_h, frame_height)))
                        inv_trans = cv2.invertAffineTransform(trans)

        
                        p1 =  np.array([p1], dtype=np.float32)
                        p0 =  np.array([p0], dtype=np.float32)
                        p1_inv = cv2.transform(p1.reshape(-1, 1, 2), inv_trans).reshape(-1, 2)
                        p0_inv = cv2.transform(p0.reshape(-1, 1, 2), inv_trans).reshape(-1, 2)

                        # motionxy = -p1_inv + p0_inv # 注意，在每一帧都随着yolo走时，由于inp是跟着hand走的，光流计算的是背景像素的运动，跟hand运动是相反的；
                        motionxy = p1_inv - p0_inv # 而在检测框被锁定在yolo的初始帧后，背景不再运动，动的是手，此时不用取逆。

                        # 为减少背景角点的干扰，根据到center_distance筛选最近的5个点
                        center_x = center[0]
                        center_y = center[1]
                        distances = [np.linalg.norm(p - [center_x, center_y]) for p in p1_inv]

                        if len(distances) >= 5:
                            sorted_indices = np.argsort(distances)
                            closest_points = sorted_indices[:5]
                            motionxy = np.array(motionxy)[closest_points, :]
                        
                        motionxy = [m for m in motionxy if abs(m[0]) > motion_threshold or abs(m[1]) > motion_threshold] # 只筛选具有位移的点

                        # 剔除与last_motion反方向的motion
                        if last_motion is not None and len(motionxy) > 0:
                            motionxy = filter_background_motion(np.array(last_motion), np.array(motionxy), 120)

                        if len(motionxy) > 0:
                            motionxy = np.array(motionxy, dtype=np.float32)
                            
                            motionxy = [np.mean(motionxy[:, 0]), np.mean(motionxy[:, 1])]
                            motionxy = np.array(motionxy, dtype=np.float32)
                            
                        else:
                            motionxy = np.zeros(2)

                        initial_bbox_n = initial_bbox[hand_num]

                        if np.linalg.norm(motionxy) > motion_threshold:  # 判断手部整体移动是否超过阈值
                            # x_left = xmin + avg_motion[0]
                            # y_top = ymin + avg_motion[1]
                            # avg_motion += motionxy # 无平滑
                            avg_motion += (motionxy / 2.) # 减少位移的累积
                            # avg_motion = (avg_motion + motionxy) / 2. # 平滑方法2
                            x_left = initial_bbox_n[0] + avg_motion[0]
                            y_top = initial_bbox_n[1] + avg_motion[1] # 将一次光流累计的motion作用在yolo给出的initial_bbox上
                        
                            next_bbox = (int(max(x_left, 0)), 
                                        int(max(y_top, 0)), 
                                        int(min(x_left + roi_w, frame_width)), 
                                        int(min(y_top + roi_h, frame_height)))
                            # 0922:
                            # 下礼拜1来了，尝试把光流估计的offset加到yolo给出的init bbox中，避免光流自己吃自己的屎把框搞飞了
                            # delta_vector = [next_bbox[i] - initial_bbox_n[i] for i in range(len(next_bbox))]

                            # cur_win_pos = (next_bbox[0], next_bbox[1])

                            # # 计算相对于初始bbox的偏移量
                            # win_pos_delta = (cur_win_pos[0] - prev_win_pos_n[0], cur_win_pos[1] - prev_win_pos_n[1])
                            # delta_vector = [delta_vector[i] - win_pos_delta[i] for i in range(len(delta_vector))]

                            # prev_win_pos = cur_win_pos
                            last_motion = avg_motion
                            print("motionxy: ", last_motion)
                            lk_bbox.append(next_bbox)
                            
                        else:
                            last_motion = np.zeros(2)
                            print("motionxy: ", last_motion)
                            lk_bbox.append(cur_bbox[hand_num]) # 不然不动，防止手静止不动光流窗口也会很快飞走的bug
                            

                        # plot_one_box(next_bbox, display_img, color=(0,0,255),line_thickness = 3)
                        # cv2.namedWindow(f"LK_pre_{hand_num}")
                        # cv2.imshow(f"LK_pre_{hand_num}", display_img)

                    else:
                        lk_bbox.append(cur_bbox[hand_num]) # motion = None的情况下
                    # **************************img transform done.****************************



                    # draw rle imgs
                    inps = cv2.cvtColor(inps, cv2.COLOR_BGR2RGB)  #在rle中，dataset的getitem中确实有这一条
                    imgwidth, imght = inps.shape[1], inps.shape[0]
                    assert imgwidth == inps.shape[1] and imght == inps.shape[0]

                    # inps = cv2.resize(inps, (int(inp_w), int(inp_h)))
                    rle_display_img = inps.copy()
                    inps = im_to_torch(inps)
                
                    inps[0].add_(-0.406)
                    inps[1].add_(-0.457)
                    inps[2].add_(-0.480)

                    inps = torch.from_numpy(np.expand_dims(inps, axis=0))

                    if use_cuda:
                        torch.cuda.synchronize()

                    inps = inps.cuda(device)
                    time1 = time.time()
                    # print("3********************************", inps.shape)
                    output = rle_module(inps)
                    if use_cuda:
                        torch.cuda.synchronize()
                    time2 = time.time()
                    rle_cost = (time2 - time1) * 1000
                    # print("rle pre cost: ", rle_cost)
                    if (rle_cost < 20):
                        rle_pre_time.append(rle_cost)

                    kpts_pre = output.pred_jts.cpu().numpy().reshape(-1, 21, 2)

                    for i in range(len(inps)):
                        imgi = inps[i].cpu().numpy()
                        imgi = np.transpose(imgi, (1, 2, 0))
                        imgi = (imgi + np.array([0.480, 0.457, 0.406], dtype=np.float32))  * np.array([255., 255., 255.], dtype=np.float32)
                        imgi = cv2.cvtColor(imgi, cv2.COLOR_BGR2RGB) 
                        img_h, img_w, _ = imgi.shape
                        imagei_pre = imgi.copy()
                        kpts_pre_i = np.array([(kpt + [0.5, 0.5]) * [img_w, img_h] for kpt in kpts_pre[i]])
                        rle_display_img = paint(rle_display_img, kpts_pre_i)
                        rle_display_img = rle_display_img[:, :, ::-1]
                        rle_display_img = rle_display_img.astype(np.uint8)

                        cv2.namedWindow(f"rle_display_{hand_num}")
                        cv2.imshow(f"rle_display_{hand_num}", rle_display_img)

                lk_pre_bbox = lk_bbox

        frame_count += 1     
        
        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"avg cost time: \n", "YOLO: ", {sum(yolo_pre_time) / len(yolo_pre_time)}, "\n", "RLE: ", {sum(rle_pre_time) / len(rle_pre_time)}, "\n")
            # import neptune
            
            # run = neptune.init_run(
            #     project="louxudong1125/abc",
            #     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhMmI2YjAwMS0zMzhmLTQzZGMtYTI1OS0wYmYxZTdhOTU3NDUifQ==",)
            
            # with open("./timecost.yaml", 'w') as file:
            #     file.write("YOLO: " + "\n")
            #     for item in yolo_pre_time:
            #         file.write(str(item) + '\n')     
            #         run["yolo_cost"].append(item)

            #     file.write("RLE: " + "\n")
            #     for item in rle_pre_time:
            #         file.write(str(item) + '\n')
            #         run["rle_cost"].append(item)
            break


if __name__ == "__main__":
    # 配置yolo
    import os
    current_dir = os.getcwd()
    yolo_voc_config = os.path.join(current_dir, 'yolo_v3', 'cfg', 'hand.data')
    yolo_model_path = os.path.join(current_dir, 'yolo_v3', 'weights', 'hand_416.pt')
    yolo_video_path = os.path.join(current_dir, 'yolo_v3', 'video', 'output.mp4')
    yolo_model_cfg = 'yolo' # yolo / yolo-tiny 模型结构
    rle_cfg_path = "./configs/256x192_res50_regress-flow_freihand.yaml"
    rle_weights_path = ".//weights//model_0919_355.pth"

    yolo_img_size = 416 # 图像尺寸
    yolo_conf_thres = 0.5# 检测置信度
    yolo_nms_thres = 0.6 # nms 阈值



    with torch.no_grad():#设置无梯度运行模型推理
        yolo_detect(rle_cfg = rle_cfg_path, 
                    rle_weights = rle_weights_path,
            yolo_model_path = yolo_model_path,
            yolo_cfg = yolo_model_cfg,
            yolo_data_cfg = yolo_voc_config,
            yolo_img_size = yolo_img_size,
            yolo_conf_thres = yolo_conf_thres,
            yolo_nms_thres = yolo_nms_thres,
            yolo_video_path = yolo_video_path)