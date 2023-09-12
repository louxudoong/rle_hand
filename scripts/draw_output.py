import cv2
import numpy as np
import math
import torch
from rlepose.opt import cfg, logger, opt
import torch
from rlepose.models import builder

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


def draw_output(m, device):
    from rlepose.datasets import Freihand_CustomDataset
    gt_val_dataset = builder.build_dataset(cfg.DATASET.VAL, preset_cfg=cfg.DATA_PRESET, train=False, heatmap2coord=cfg.TEST.HEATMAP2COORD)
    gt_val_sampler = torch.utils.data.distributed.DistributedSampler(
        gt_val_dataset, num_replicas=opt.world_size, rank=opt.rank)

    gt_val_loader = torch.utils.data.DataLoader(
        gt_val_dataset, batch_size=16, shuffle=False, num_workers=20, drop_last=False, sampler=gt_val_sampler)
    
    

    for index, (inps, labels) in enumerate(gt_val_loader):
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

        kpts_pre = labels['target_uv'].cpu().numpy().reshape(-1, 21, 2) # size * 42
        print(kpts_pre.shape())
        kpts_gt = output.pred_jts.cpu().numpy().reshape(-1, 21, 2)

        # print & draw
        for i in range(len(inps)):
            if i == 0:
                #print(f'batch:{batch_idx}_{i}')
                imgi = inps[i].cpu().numpy()
                imgi = np.transpose(imgi, (1, 2, 0))
                imgi = (imgi + np.array([0.480, 0.457, 0.406], dtype=np.float32))  * np.array([255., 255., 255.], dtype=np.float32)
                imgi = cv2.cvtColor(imgi, cv2.COLOR_BGR2RGB) 
                img_h, img_w, _ = imgi.shape
                kpts_pre_i = np.array([(kpt + [0.5, 0.5]) * [img_w, img_h] for kpt in kpts_pre[i]])
                kpts_gt_i = np.array([(kpt + [0.5, 0.5]) * [img_w, img_h] for kpt in kpts_gt[i]])

                imagei_pre = imgi.copy()
                imagei_gt = imgi.copy()
                imagei_pre = paint(imagei_pre, kpts_pre_i)
                imagei_gt = paint(imagei_gt, kpts_gt_i)
                cv2.imwrite(f'./exp/output_114/{index}_{i}_pre.jpg', imagei_pre)
                cv2.imwrite(f'./exp/output_114/{index}_{i}_gt.jpg', imagei_gt)


def main():
    device = torch.device('cuda')
    m = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)  # 根据cfg的配置信息构建模型

    print(f'Loading model from {opt.checkpoint}...')
    # print('Loading model from {}...'.format(opt.checkpoint))
    m.load_state_dict(torch.load(opt.checkpoint, map_location='cpu'), strict=True)  # 加载权重

    m.cuda(device)  # 把模型放到gpu中

    with torch.no_grad():
        draw_output(m, device)


if __name__ == "__main__":

    main()