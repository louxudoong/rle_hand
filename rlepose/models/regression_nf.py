import numpy as np
import torch
import torch.distributions as distributions
import torch.nn as nn
from easydict import EasyDict

from .builder import SPPE
from .layers.real_nvp import RealNVP
from .layers.Resnet import ResNet


# 两个子网络，其组成为fc+relu+fc+relu+fc， nets还添加了tanh来将输出限制在-1,1
def nets():
    return nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2), nn.Tanh())


def nett():
    return nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2))


# 一个线性层，其功能是生成可以包含归一化、bias功能的线性变换层
class Linear(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, norm=True):
        super(Linear, self).__init__()
        self.bias = bias
        self.norm = norm
        self.linear = nn.Linear(in_channel, out_channel, bias)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)

    def forward(self, x):
        y = x.matmul(self.linear.weight.t()) # 等价于y = self.linear(x)? (i think)

        # 如果设置了归一化，把batch_size * channels * height * weight的图像归一化为batch_size * 1 * height * weight
        # 计算的是各个通道的二范数
        # 若keepdim为false吗，输出的就是batch_size * height * weight
        if self.norm:
            x_norm = torch.norm(x, dim=1, keepdim=True)
            y = y / x_norm

        # 如果开了bias，就把bias加上
        if self.bias:
            y = y + self.linear.bias
        return y


@SPPE.register_module
class RegressFlow(nn.Module):
    '''
    MODEL:
    TYPE: 'RegressFlow'
    PRETRAINED: ''
    TRY_LOAD: ''
    NUM_FC_FILTERS:
    - -1
    HIDDEN_LIST: -1
    NUM_LAYERS: 50
    '''
    def __init__(self, norm_layer=nn.BatchNorm2d, **cfg):
        super(RegressFlow, self).__init__()
        self._preset_cfg = cfg['PRESET']
        self.fc_dim = cfg['NUM_FC_FILTERS']
        self._norm_layer = norm_layer
        self.num_joints = self._preset_cfg['NUM_JOINTS']
        self.height_dim = self._preset_cfg['IMAGE_SIZE'][0]
        self.width_dim = self._preset_cfg['IMAGE_SIZE'][1]

        self.preact = ResNet(f"resnet{cfg['NUM_LAYERS']}")  # 根据cfg的参数，这是resnet50

        # Imagenet pretrain model
        import torchvision.models as tm  # noqa: F401,F403
        assert cfg['NUM_LAYERS'] in [18, 34, 50, 101, 152]
        x = eval(f"tm.resnet{cfg['NUM_LAYERS']}(pretrained=True)")

        # 根据cfg中NUM_LAYERS的参数确定特征通道的数量
        self.feature_channel = {
            18: 512,
            34: 512,
            50: 2048,
            101: 2048,
            152: 2048
        }[cfg['NUM_LAYERS']]
        self.hidden_list = cfg['HIDDEN_LIST']

        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items()
                 if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)   # 搞成channels * 1 * 1的向量

        self.fcs, out_channel = self._make_fc_layer()

        self.fc_coord = Linear(out_channel, self.num_joints * 2)  # 用来预测关键点坐标的坐标
        self.fc_sigma = Linear(out_channel, self.num_joints * 2, norm=False)  # 用来预测关节点标准差

        self.fc_layers = [self.fc_coord, self.fc_sigma]

        prior = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))  # 定义一个二维01正态分布的先验分布
        masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))  # 用于确定RealNVP模型中每个coupling层的输入和输出变量的对应关系，其形状为6*2*2

        self.flow = RealNVP(nets, nett, masks, prior) # flow model，根据论文，用于实现简单分布-复杂分布的映射。这里的输入分别为nets,nett（这是两个由fc层、relu组成的网络块）、张量mask与先验分布prior

    # 根据fc_dim创建num_deconv个fc层，但这里参数为-1，不创建任何fc层
    # 如果fc_dim列表不为-1，那么就按照fc+bn+relu的顺序创建fc块
    def _make_fc_layer(self):
        fc_layers = []
        num_deconv = len(self.fc_dim)
        input_channel = self.feature_channel
        for i in range(num_deconv):
            if self.fc_dim[i] > 0:
                fc = nn.Linear(input_channel, self.fc_dim[i])
                bn = nn.BatchNorm1d(self.fc_dim[i])
                fc_layers.append(fc)
                fc_layers.append(bn)
                fc_layers.append(nn.ReLU(inplace=True))
                input_channel = self.fc_dim[i]
            else:
                fc_layers.append(nn.Identity())  # 恒等映射，不进行任何操作

        return nn.Sequential(*fc_layers), input_channel

    # 初始化xavier_uniform_权重
    def _initialize(self):
        for m in self.fcs:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
        for m in self.fc_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

    # 这里x为img(i think)
    def forward(self, x, labels=None):
        BATCH_SIZE = x.shape[0]

        feat = self.preact(x)

        _, _, f_h, f_w = feat.shape
        feat = self.avg_pool(feat).reshape(BATCH_SIZE, -1)  # 将preact输出的特征向量变为batch_size * channels，这里channels根据resnet50，为2048

        out_coord = self.fc_coord(feat).reshape(BATCH_SIZE, self.num_joints, 2)  # 进linear，并且完成归一化和添加bias,再reshape为batch_size * num_joints * 2，预测坐标
        assert out_coord.shape[2] == 2

        out_sigma = self.fc_sigma(feat).reshape(BATCH_SIZE, self.num_joints, -1) # 预测sigma, torch.Size([64, 17, 2])

        # (B, N, 2)
        pred_jts = out_coord.reshape(BATCH_SIZE, self.num_joints, 2)
        sigma = out_sigma.reshape(BATCH_SIZE, self.num_joints, -1).sigmoid() # 使用sigmoid确保数值在01之间
        scores = 1 - sigma

        scores = torch.mean(scores, dim=2, keepdim=True) # torch.Size([64, 17, 1])

        if self.training and labels is not None:
            # modi2
            # gt_uv = labels['target_uv'].reshape(pred_jts.shape)
            # print("debug regression: ", pred_jts)
            gt_uv = labels['target_uv'].reshape(pred_jts.shape)  # torch.Size([32, 17, 2])
            # print(f"modi2: get_uv size = {gt_uv.shape}")
            bar_mu = (pred_jts - gt_uv) / sigma
            # (B, K, 2)
            log_phi = self.flow.log_prob(bar_mu.reshape(-1, 2)).reshape(BATCH_SIZE, self.num_joints, 1)

            nf_loss = torch.log(sigma) - log_phi
        else:
            nf_loss = None

        output = EasyDict(
            pred_jts=pred_jts,
            sigma=sigma,
            maxvals=scores.float(),
            nf_loss=nf_loss
        )
        return output  # output为关键点坐标pred_jts, 预测分布的标准差sigma，关键点坐标最大值（？）maxvals，与NF loss。
