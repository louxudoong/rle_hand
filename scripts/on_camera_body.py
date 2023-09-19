'''
本脚本需要置于原版的rle代码的scripts目录下
'''
from rlepose.utils.valid_utils_lxd import paint_body17
import torch
import cv2
import numpy as np
import torch
from rlepose.models import builder
from rlepose.utils.config import update_config
from rlepose.utils.transforms import im_to_torch
import time


def main():
    device = torch.device('cuda')
    cfg = update_config("./configs/256x192_res50_regress-flow.yaml")
    input_size = cfg.DATA_PRESET['IMAGE_SIZE']
    output_size = cfg.DATA_PRESET['HEATMAP_SIZE']

    time_0 = time.time()
    m = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)  # 根据cfg的配置信息构建模型
    m.load_state_dict(torch.load(".//weights//coco-laplace-rle.pth", map_location='cpu'), 
                    strict=True)  # 加载权重
    time_1 = time.time()  # 记录开始时间
    time_loadweights = time_1 - time_0  # 计算耗时，单位为秒
    print("build model and load weights cost time: ", time_loadweights)

    m.cuda(device)  # 把模型放到gpu中

    with torch.no_grad():
        # 创建摄像头对象
        cap = cv2.VideoCapture(0)
        count = 0

        while True:
            # 读取摄像头捕获的一帧图像
            ret, frame = cap.read()
            # 如果成功读取图像，则进行处理
            
            if ret:
                init_img = frame

                src = cv2.cvtColor(init_img, cv2.COLOR_BGR2RGB)
                imgwidth, imght = src.shape[1], src.shape[0]
                assert imgwidth == src.shape[1] and imght == src.shape[0]

                inp_h, inp_w = input_size
                img = cv2.resize(src, (int(inp_w), int(inp_h)))
                img = im_to_torch(img)
                img[0].add_(-0.406)
                img[1].add_(-0.457)
                img[2].add_(-0.480)

                inps = torch.from_numpy(np.expand_dims(img, axis=0))
                inps = inps.cuda(device)
                time_0 = time.time()
                output = m(inps)
                time_1 = time.time()
                time_pre = time_1 - time_0
                print("model pre cost time: ", time_pre)
    
                kpts_pre = output.pred_jts.cpu().numpy().reshape(-1, 17, 2)

                for i in range(len(inps)):
                    imgi = inps[i].cpu().numpy()
                    imgi = np.transpose(imgi, (1, 2, 0))
                    imgi = (imgi + np.array([0.480, 0.457, 0.406], dtype=np.float32))  * np.array([255., 255., 255.], dtype=np.float32)
                    imgi = cv2.cvtColor(imgi, cv2.COLOR_BGR2RGB) 
                    img_h, img_w, _ = imgi.shape
                    imagei_pre = imgi.copy()
                    kpts_pre_i = np.array([(kpt + [0.5, 0.5]) * [img_w, img_h] for kpt in kpts_pre[i]])
                    imagei_pre = paint_body17(imagei_pre, kpts_pre_i)
                    imagei_pre = imagei_pre.astype(np.uint8)
                    cv2.imshow('RLE', imagei_pre)
                    # cv2.imshow("0", init_img)
                    # cv2.waitKey(0)
                    # cv2.imwrite(f'./exp/output/{time_0}.jpg', imagei_pre)
                    # cv2.imwrite(f'./exp/out/init.jpg', init_img)

                    # 保存检测效果为gif
                    # img_list = []
                    # fps = 10
                    # output_path = f"./exp/output/{time_0}.gif"
                    # if True:
                    #     img_list.append(imagei_pre)
                    #     count += 1
                    #     if count % 30 == 0:
                    #         imageio.mimsave(output_path, img_list, duration=(1000 / fps))
                    #         raise


            # 按下 'q' 键退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    main()