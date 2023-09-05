import cv2
import numpy as np
import math
import torch
from rlepose.opt import cfg, logger, opt

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

    for i, (inps, labels) in enumerate(train_loader):
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

        kpts = labels['target_uv'] # size * 42


        optimizer.clear_grad()  # 梯度清零
        heat1, heat2, heat3, heat4, heat5, heat6 = model(img, centermap)
        loss1 = criterion(heat1, heatmap) * heat_weight
        loss2 = criterion(heat2, heatmap) * heat_weight
        loss3 = criterion(heat3, heatmap) * heat_weight
        loss4 = criterion(heat4, heatmap) * heat_weight
        loss5 = criterion(heat5, heatmap) * heat_weight
        loss6 = criterion(heat6, heatmap) * heat_weight
        loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        loss.backward()  # 反向传播  
        optimizer.step() # 更新模型参数，根据反向传播backward对模型进行更新
        running_loss += loss.item()
        run["train/loss"].append(loss.item())
        if (batch_idx + 1) % 10 == 0:  # 每迭代10个批次打印一次损失
            # if(running_loss <= cur_loss):
            #     paddle.save(model.state_dict(), "cpm_init_params.pdparams")
            print(f"Batch [{batch_idx+1}/{len(train_dataloader)}], Loss: {running_loss/10:.4f}")
            running_loss = 0.0

        # print & draw
        # if loss.item() < 1:
        #     for i in range(len(img)):
        #         if (i + 1) % 5 == 0:
        #             #print(f'batch:{batch_idx}_{i}')
        #             imgi = img[i].cpu().numpy()
        #             heatmapi = heat6[i].cpu().numpy()
        #             heatmapi_t = heatmap[i].cpu().numpy()
        #             kptsi = get_kpts_from_heatmap(heatmapi, 368., 368.)
        #             kptsi_t = get_kpts_from_heatmap(heatmapi_t, 368., 368.)
        #             imagei = np.transpose(imgi, (1, 2, 0))
        #             imagei = imagei * np.array([0.229, 0.224, 0.225], dtype=np.float32)*255 + np.array([0.485, 0.456, 0.406], dtype=np.float32)*255
        #             imagei = imagei.copy()
        #             imagei_t = imagei.copy()
        #             draw_paint(imagei, kptsi)
        #             draw_paint(imagei_t, kptsi_t)
        #             cv2.imwrite(f'./output3/{batch_idx}_{i}.jpg', imagei)
        #             cv2.imwrite(f'./output3/{batch_idx}_{i}_t.jpg', imagei_t)