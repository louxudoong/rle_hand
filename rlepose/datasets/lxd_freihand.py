"""Custum training dataset."""
import copy
import os
import pickle as pk
from abc import abstractmethod, abstractproperty

import cv2
import torch.utils.data as data
from rlepose.utils.presets import SimpleTransform
from pycocotools.coco import COCO

import numpy as np
import glob
from PIL import Image
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
    bboxs = []

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

        bboxs.append([min_x, min_y, max_x, max_y])

        center_x = (max_x + min_x) / 2
        center_y = (max_y + min_y) / 2
        
        centers.append([center_x, center_y])

        scale_x = (max_x - min_x + 4) / w
        scale_y = (max_y - min_y + 4) / h
        scale = max(scale_x, scale_y)

        scales.append(scale)
    
    return kpts, centers, scales, bboxs

from rlepose.models.builder import DATASET
@DATASET.register_module
class Freihand_CustomDataset(data.Dataset):
    """Custom dataset.
    Modified from the coco dataset to fit Freihand

    Parameters
    ----------
    train: bool, default is True
        If true, will set as training mode.
    skip_empty: bool, default is False
        Whether skip entire image if no valid label is found.
    cfg: dict, dataset configuration.
    """

    CLASSES = None

    def __init__(self,
                 train=True,
                 skip_empty=True,
                 lazy_import=False,
                 **cfg
                 ):
        self._root_dir = cfg['root_dir']
        # self._root_dir = cfg['root_dir_autodl']
        self._split0 = cfg['split0']
        self._split1 = cfg['split1']
        self._split2 = cfg['split2']
        self._mode = cfg['mode']

        self._items = read_data_file(self._root_dir, split=self._split0)

        # 其实这里centerlist和scalelist都没用
        kpt_list, center_list, scale_list, bboxs= read_mat_file(self._root_dir, self._split1, self._split2, self._items)

        if self._mode == "train":
            self._kpt_list = kpt_list * 4
            self._center_list = center_list * 4
            self._scale_list =scale_list * 4
            self._bboxs = bboxs * 4
        if self._mode == "eval":
            self._kpt_list = kpt_list
            self._center_list = center_list
            self._scale_list =scale_list
            self._bboxs = bboxs

        self._cfg = cfg
        self._preset_cfg = cfg['PRESET']
        # self._root = cfg['ROOT']
        # self._img_prefix = cfg['IMG_PREFIX']
        # self._ann_file = os.path.join(self._root, cfg['ANN'])

        self._lazy_import = lazy_import
        self._skip_empty = skip_empty
        self._train = train

        if 'AUG' in cfg.keys():
            print("AUG ON.")
            self._scale_factor =  cfg['AUG']['SCALE_FACTOR']
            self._rot =  cfg['AUG']['ROT_FACTOR']
            self.num_joints_half_body =  cfg['AUG']['NUM_JOINTS_HALF_BODY']
            self.prob_half_body =  cfg['AUG']['PROB_HALF_BODY']
        else:
            print("AUG OFF.")
            self._scale_factor = 0
            self._rot = 0
            self.num_joints_half_body = -1
            self.prob_half_body = -1

        self._input_size = self._preset_cfg['IMAGE_SIZE']
        self._output_size = self._preset_cfg['HEATMAP_SIZE']

        self._sigma = self._preset_cfg['SIGMA']

        self._check_centers = False

        # self.num_class = len(self.CLASSES)

        self._loss_type = cfg['heatmap2coord']

        self._num_joints = 21

        # self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        # self.lower_body_ids = (11, 12, 13, 14, 15, 16)

        if self._preset_cfg['TYPE'] == 'simple':
            self.transformation = SimpleTransform(
                self, scale_factor=self._scale_factor,
                input_size=self._input_size,
                output_size=self._output_size,
                rot=self._rot, sigma=self._sigma,
                train=self._train, loss_type=self._loss_type)
        else:
            raise NotImplementedError

        # self._items, self._labels = self._lazy_load_json()

        # modi: 这里已经读取了图像路径列表为_items, labels由于只有坐标点信息，已经存为了_kpt_list
        # init中不再计算labels，因为他要进items的循环，放到getitem中来节省资源

    def __getitem__(self, idx):
        # get freihand imgs and gt_uv
        img_path = self._items[idx]
        kpt = copy.deepcopy(self._kpt_list[idx])
        # img = np.array(cv2.imread(img_path), dtype=np.float32)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        
        height, width = img.shape[0], img.shape[1]


        # modi: 由于transform里要用到centers， 直接塞到labels里去？还是
        bbox = self._bboxs[idx]
        label = self.check_load_keypoints(kpt, height, width, bbox) # 这个函数跟check没有关系了，只是把数据转成要求的格式，输出符合要求的label

        # import copy
        
        # center = copy.deepcopy(self.center_list[idx])
        # scale = copy.deepcopy(self.scale_list[idx])

        # get image id
        # img_path = self._items[idx]
        # img_id = int(os.path.splitext(os.path.basename(img_path))[0])

        # load ground truth, including bbox, keypoints, image size

        

        # transform ground truth into training label and apply data augmentation
        target = self.transformation(img, label)  
        img = target.pop('image')


        return img, target
    
    def __len__(self):
        return len(self._items)

    # def _lazy_load_ann_file(self):
    #     if os.path.exists(self._ann_file + '.pkl') and self._lazy_import:
    #         print('Lazy load json...')
    #         with open(self._ann_file + '.pkl', 'rb') as fid:
    #             return pk.load(fid)
    #     else:
    #         _database = COCO(self._ann_file)
    #         if os.access(self._ann_file + '.pkl', os.W_OK):
    #             with open(self._ann_file + '.pkl', 'wb') as fid:
    #                 pk.dump(_database, fid, pk.HIGHEST_PROTOCOL)
    #         return _database

    # def _lazy_load_json(self):
    #     if os.path.exists(self._ann_file + '_annot_keypoint.pkl') and self._lazy_import:
    #         print('Lazy load annot...')
    #         with open(self._ann_file + '_annot_keypoint.pkl', 'rb') as fid:
    #             items, labels = pk.load(fid)
    #     else:
    #         items, labels = self._load_jsons()
    #         if os.access(self._ann_file + '_annot_keypoint.pkl', os.W_OK):
    #             with open(self._ann_file + '_annot_keypoint.pkl', 'wb') as fid:
    #                 pk.dump((items, labels), fid, pk.HIGHEST_PROTOCOL)

    #     return items, labels

    # @abstractmethod
    # def _load_jsons(self):
    #     pass
    
    def check_load_keypoints(self, kpt, height, width, bbox): # to check img and labels, and convert kpts to 3D
        # frei中似乎没有置信度信息，取消一切关于检查的功能

        # ann_ids = coco.getAnnIds(imgIds=entry['id'], iscrowd=False)
        # objs = coco.loadAnns(ann_ids)
        # check valid bboxes
        valid_objs = []
        # width = entry['width']
        # height = entry['height']

        # for obj in objs:
        #     contiguous_cid = self.json_id_to_contiguous[obj['category_id']]
        #     if contiguous_cid >= self.num_class:
        #         # not class of interest
        #         continue
        #     if max(obj['keypoints']) == 0:
        #         continue
        # # convert from (x, y, w, h) to (xmin, ymin, xmax, ymax) and clip bound
        # xmin, ymin, xmax, ymax = bbox_clip_xyxy(bbox_xywh_to_xyxy(obj['bbox']), width, height)
        # require non-zero box area
        # if obj['area'] <= 0 or xmax <= xmin or ymax <= ymin:
        #     continue
        # if obj['num_keypoints'] == 0:
        #     continue
        # joints 3d: (num_joints, 3, 2); 3 is for x, y, z; 2 is for position, visibility
        joints_3d = np.zeros((self._num_joints, 3, 2), dtype=np.float32)
        for i in range(self._num_joints):
            joints_3d[i, 0, 0] = kpt[i, 0] # 按照设计，kpts为joints_num * 2的list
            joints_3d[i, 1, 0] = kpt[i, 1]
            # 避免出问题，把置信度全部改成1
            joints_3d[i, 0, 1] = 1.
            joints_3d[i, 1, 1] = 1.

            # joints_3d[i, 2, 0] = 0
            # visible = min(1, obj['keypoints'][i * 3 + 2])
            # joints_3d[i, :2, 1] = visible
            # joints_3d[i, 2, 1] = 0

            # if np.sum(joints_3d[:, 0, 1]) < 1:
            #     # no visible keypoint
            #     continue

            # if self._check_centers and self._train:
            #     bbox_center, bbox_area = self._get_box_center_area((xmin, ymin, xmax, ymax))
            #     kp_center, num_vis = self._get_keypoints_center_count(joints_3d)
            #     ks = np.exp(-2 * np.sum(np.square(bbox_center - kp_center)) / bbox_area)
            #     if (num_vis / 80.0 + 47 / 80.0) > ks:
            #         continue

            valid_objs = {
                # 'bbox': (xmin, ymin, xmax, ymax),
                'bbox': (bbox[0], bbox[1], bbox[2], bbox[3]), 
                'width': width,
                'height': height,
                'joints_3d': joints_3d
            }

        return valid_objs


    @abstractproperty
    def CLASSES(self):
        return None

    # @abstractproperty
    # def num_joints(self):
    #     return None

    @abstractproperty
    def joint_pairs(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return None
