import cv2
import numpy as np
import math
import torch
from rlepose.opt import cfg, logger, opt
import torch
from rlepose.models import builder
from rlepose.opt import cfg, opt

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
        x = k[0]
        y = k[1]
        print(f'im .shape = {im.shape}')
        print(f'x={x}, y={y}')
        cv2.circle(im, (x, y), 2, (0, 0, 255), -1)

    # draw lines
    for i in range(len(limbSeq)):
        cur_im = im.copy()
        limb = limbSeq[i]
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


def draw_output(m, device):
    from rlepose.datasets import Freihand_CustomDataset
    root_dir = "/home/louxd/dataset/FreiHand"
    split0_train = "FreiHAND_pub_v2/training"
    split1_train = "FreiHAND_pub_v2"
    split2_train = "training"
    split0_eval = "FreiHAND_pub_v2_eval/evaluation"
    split1_eval = "FreiHAND_pub_v2_eval"
    split2_eval = "evaluation"
    mode_train = "train"
    mode_eval = "eval"
    batch_size = cfg.TRAIN.BATCH_SIZE
    
    train_dataset = Freihand_CustomDataset(root_dir, split0_train, split1_train, split2_train,
                                                cfg, mode=mode_train)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=opt.world_size, rank=opt.rank)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=(train_sampler is None), num_workers=opt.nThreads, 
        sampler=train_sampler)

    for index, (inps, labels) in enumerate(train_loader):
        inps = inps.cuda(device)

        for k, _ in labels.items():
            if k == 'type':
                continue
            
            labels[k] = labels[k].cuda(device)

        output = m(inps, labels)

        if isinstance(inps, list):
            batch_size = inps[0].size(0)
        else:
            batch_size = inps.size(0)

        kpts = labels['target_uv'].view(32, 21, 2) # size * 42
        kpts_t = output.pred_jts

        # print & draw
        for i in range(len(inps)):
            if i == 0:
                #print(f'batch:{batch_idx}_{i}')
                imgi = inps[i].cpu().numpy()
                kptsi = kpts[i]
                kptsi_t = kpts_t[i]
                imagei = np.transpose(imgi, (1, 2, 0))
                imagei = imagei * np.array([0.229, 0.224, 0.225], dtype=np.float32)*255 + np.array([0.485, 0.456, 0.406], dtype=np.float32)*255
                imagei = imagei.copy()
                imagei_t = imagei.copy()
                imagei = paint(imagei, kptsi)
                imagei_t = paint(imagei_t, kptsi_t)
                cv2.imwrite(f'./output3/{index}_{i}.jpg', imagei)
                cv2.imwrite(f'./output3/{index}_{i}_t.jpg', imagei_t)


def main():
    device = torch.device('cuda')
    m = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)  # 根据cfg的配置信息构建模型

    print(f'Loading model from {opt.checkpoint}...')
    # print('Loading model from {}...'.format(opt.checkpoint))
    m.load_state_dict(torch.load(opt.checkpoint, map_location='cpu'), strict=True)  # 加载权重

    m.cuda(device)  # 把模型放到gpu中

    draw_output(m, device)


if __name__ == "__main__":

    main()