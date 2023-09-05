from .coco_det import Mscoco_det
from .custom import CustomDataset
from .mscoco import Mscoco
from .lxd_freihand import Freihand_CustomDataset
# from .h36m import H36m
# from .h36m_mpii import H36mMpii

__all__ = ['CustomDataset', 'Mscoco', 'Mscoco_det', 'H36m', 'H36mMpii', 'Freihand_CustomDataset']
