import math
import cv2
import numpy as np

def draw_output(imgs, heatmaps, centermaps, idx=0):
    for i in range(len(imgs)):
        img_i = imgs[i].cpu().numpy()
        heatmap_i = heatmaps[i].cpu().numpy()
        centermap_i = centermaps[i].cpu().numpy()
        kpts_i = get_kpts_from_heatmap(heatmap_i, 368., 368.)

        image = np.transpose(img_i, (1, 2, 0))
        image = image * np.array([0.229, 0.224, 0.225], dtype=np.float32)*255 + np.array([0.485, 0.456, 0.406], dtype=np.float32)*255
        image = image.copy()
        image = draw_paint(image, kpts_i)
        if (i + 1) % 10 == 0:
            cv2.imwrite(f"./cpm_output/{idx}_{i}.jpg", draw_paint(image, kpts_i))  
         
def get_kpts_from_heatmap(heatmap, img_h, img_w):
    heatmap = heatmap.copy()
    kpts = []
    for m in heatmap[1:]:
        coords = np.unravel_index(m.argmax(), m.shape)
        h, w = coords[0], coords[1]
        x = int(w * img_w / m.shape[1])
        y = int(h * img_h / m.shape[0])
        kpts.append([x,y])
    return kpts


def draw_paint(im, kpts):

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

def guassian_kernel(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)

def center_map_default(size_w, size_h, sigma):
    """
    生成均值位于图像中点的centermap。
    """
    center_x = (size_w + 1) // 2
    center_y = (size_h + 1) // 2
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    center_map = np.exp(-D2 / 2.0 / sigma / sigma)

    centermap = np.zeros((1, size_h, size_w), dtype=np.float32)

    center_map[center_map > 1] = 1
    center_map[center_map < 0.0099] = 0
    centermap[0, :, :] = center_map

    return centermap