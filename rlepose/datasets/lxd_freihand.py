# -*-coding:UTF-8-*-
import os
import numpy as np
import glob
# import torch.utils.data as data
from PIL import Image
import cv2
import sys
sys.path.append("../thirdparty")
import rlepose.datasets.mytransform as mytransform
import json

def read_data_file(root_dir, split="FreiHAND_pub_v2_eval/evaluation"):
    '''
    Args:
    split ="FreiHAND_pub_v2/training" or "FreiHAND_pub_v2_eval/evaluation"
    '''
    image_arr = np.array(glob.glob(os.path.join(root_dir, split, 'rgb/*.jpg'))) # 读取所有img，并存入image_arr
    image_nums_arr = np.array([float(s.rsplit('/')[-1][0:-4]) for s in image_arr]) # 读取image_arr中所有图像文件名的编号，并存入image_nums_arr
    sorted_image_arr = image_arr[np.argsort(image_nums_arr)] # 对image的文件路径按编号进行排序
    return sorted_image_arr.tolist() # 输出有序的所有img的路径列表


def read_mat_file(root_dir, split1, split2, img_list):
    '''
    Args:
    split1 ="FreiHAND_pub_v2" or "FreiHAND_pub_v2_eval"
    split2 = "training" or "evaluation"
    '''
    anno_xyz = []
    anno_K = []
    
    with open(os.path.join(root_dir, split1, f"{split2}_xyz.json")) as f:
        anno_xyz = json.load(f)
    with open(os.path.join(root_dir, split1, f"{split2}_K.json")) as f:
        anno_K = json.load(f)

    kpts = []
    centers = []
    scales = []

    '''
    Notice:
    在train数据集中，label的样本数为img的1/4，for循环以label的length为准
    '''
    for idx in range(len(anno_xyz)):
        xyz = np.array(anno_xyz[idx])
        K = np.array(anno_K[idx])
        uv_z = np.matmul(K, xyz.T).T
        uv = uv_z[:, :2] / uv_z[:, -1:]

        kpts.append(uv)

        im = Image.open(img_list[idx])
        w = im.size[0]
        h = im.size[1]

        max_x = np.max(np.clip(uv[:, 0], 0, w))
        min_x = np.min(np.clip(uv[:, 0], 0, w))
        max_y = np.max(np.clip(uv[:, 1], 0, h))
        min_y = np.min(np.clip(uv[:, 1], 0, h))

        center_x = (max_x + min_x) / 2
        center_y = (max_y + min_y) / 2
        
        centers.append([center_x, center_y])

        scale_x = (max_x - min_x + 4) / w
        scale_y = (max_y - min_y + 4) / h
        scale = max(scale_x, scale_y)

        scales.append(scale)
    
    return kpts, centers, scales


def guassian_kernel(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)


import re
def re_encode(params):
    # old_dic = params.
    keys = params.keys()
    counter = {}
    new_keys = []
    for item in keys:
        # 使用正则表达式匹配conv2d_xx的数字部分
        match = re.search(r'conv2d_(\d+)', item)
        if match:
            number = int(match.group(1))  # 提取匹配到的数字并转为整数
            suffix = re.search(r'(\w+)$', item).group()  # 提取后缀名部分

            # 检查字典中是否有后缀名对应的计数器
            if suffix not in counter:
                counter[suffix] = 0

            # 对应后缀名的计数器加1，并将新编号替换原字符串中的数字部分
            counter[suffix] += 1
            new_number = counter[suffix] - 1
            new_item = re.sub(r'conv2d_\d+', f'conv2d_{new_number}', item)
            # print(new_item)
            new_keys.append(new_item)
    
    new_params = {}
    for old_key, new_key in zip(keys, new_keys):
        old_value = params.get(old_key)
        if old_value is not None:
            new_params[new_key] = old_value
    
    return new_params


class FreiHand_RLE:
    """
        Args:
            root_dir (str): the path of dateset.
            stride (float): default = 8
            transformer (Mytransforms): expand dataset.
            mode: "eval" or "train"
        Notice:
            you have to change code to fit your own dataset except LSP

    """

    def __init__(self, root_dir, split0, split1, split2, mode="eval", stride=8, transformer=None):

        self.img_list = read_data_file(root_dir, split=split0)
        self.mode = mode
        kpt_list, center_list, scale_list= read_mat_file(root_dir, split1, split2, self.img_list)
        if self.mode == "train":
            self.kpt_list = kpt_list * 4
            self.center_list = center_list * 4
            self.scale_list =scale_list * 4
        if self.mode == "eval":
            self.kpt_list = kpt_list
            self.center_list = center_list
            self.scale_list =scale_list
        
        self.stride = stride
        self.transformer = transformer
        self.sigma = 3.0
        


    def __getitem__(self, index):

        img_path = self.img_list[index]
        img = np.array(cv2.imread(img_path), dtype=np.float32)

        import copy
        kpt = copy.deepcopy(self.kpt_list[index])
        center = copy.deepcopy(self.center_list[index])
        scale = copy.deepcopy(self.scale_list[index])

        # expand dataset
        #print(img.shape)
        img, kpt, center = self.transformer(img, kpt, center, scale)
        # height, width, _ = img.shape

        # #attention: / 改为 //
        # heatmap = np.zeros((height // self.stride, width // self.stride, len(kpt) + 1), dtype=np.float32)
        # for i in range(len(kpt)):
        #     x = int(kpt[i][0]) * 1.0 / self.stride
        #     y = int(kpt[i][1]) * 1.0 / self.stride
        #     heat_map = guassian_kernel(size_h=height / self.stride, size_w=width / self.stride, center_x=x, center_y=y, sigma=self.sigma)
        #     heat_map[heat_map > 1] = 1
        #     heat_map[heat_map < 0.0099] = 0
        #     heatmap[:, :, i + 1] = heat_map

        # heatmap[:, :, 0] = 1.0 - np.max(heatmap[:, :, 1:], axis=2)  # for background

        # centermap = np.zeros((height, width, 1), dtype=np.float32)
        # center_map = guassian_kernel(size_h=height, size_w=width, center_x=center[0], center_y=center[1], sigma=3)
        # center_map[center_map > 1] = 1
        # center_map[center_map < 0.0099] = 0
        # centermap[:, :, 0] = center_map
        # # img = copy.deepcopy(Mytransforms.normalize(Mytransforms.to_tensor(img), np.array([0.485, 0.456, 0.406])*255,
        # #                              np.array([0.229, 0.224, 0.225])*255))
        # img = copy.deepcopy(mytransform.normalize(mytransform.to_tensor(img), np.array([128., 128., 128.]),
        #                         np.array([256., 256., 256.])))
        # # img = Mytransforms.to_tensor(img)  在normalize已经to_tensor过了！！！！
        # heatmap = copy.deepcopy(mytransform.to_tensor(heatmap))
        # centermap = copy.deepcopy(mytransform.to_tensor(centermap))
        return img, kpt

    def __len__(self):
        return len(self.img_list)


