"""Validation script."""
import torch
import torch.multiprocessing as mp
from rlepose.models import builder
from rlepose.opt import cfg, opt
from rlepose.trainer import validate, validate_gt, validate_gt_3d
from rlepose.utils.env import init_dist
from rlepose.utils.transforms import get_coord

num_gpu = torch.cuda.device_count()



def main():
    if opt.launcher in ['none', 'slurm']:
        main_worker(None, opt, cfg)
    else:
        ngpus_per_node = torch.cuda.device_count()
        opt.ngpus_per_node = ngpus_per_node
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(opt, cfg))


def main_worker(gpu, opt, cfg):

    if gpu is not None:
        opt.gpu = gpu

    init_dist(opt) # 初始化分布式训练环境，用不上

    torch.backends.cudnn.benchmark = True

    m = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)  # 根据cfg的配置信息构建模型

    print(f'Loading model from {opt.checkpoint}...')
    # print('Loading model from {}...'.format(opt.checkpoint))
    m.load_state_dict(torch.load(opt.checkpoint, map_location='cpu'), strict=True)  # 加载权重

    m.cuda(opt.gpu)  # 把模型放到gpu中

    m = torch.nn.parallel.DistributedDataParallel(m, device_ids=[opt.gpu])  # 多卡并行，用不上

    output_3d = cfg.DATA_PRESET.get('OUT_3D', False) # 获得yaml中的OUT_3D字段，如果为空则返回false

    heatmap_to_coord = get_coord(cfg, cfg.DATA_PRESET.HEATMAP_SIZE, output_3d)  # 这里return preds, pred_scores，怎么跟hm扯上关系了？？-- 只是做坐标转换的，跟heatmap没有关系


    with torch.no_grad():
        # 输出3d坐标
        if output_3d:
            err = validate_gt_3d(m, opt, cfg, heatmap_to_coord, opt.valid_batch)

            if opt.log:
                print('##### results: {} #####'.format(err))
        else:
            gt_AP = validate_gt(m, opt, cfg, heatmap_to_coord, opt.valid_batch)
            detbox_AP = validate(m, opt, cfg, heatmap_to_coord, opt.valid_batch)
            # modi7: draw & save output
            from rlepose.utils.lxd_draw_output import draw_paint
            print(pose_coords.shape, inps.shape)

            if opt.log:
                print('##### gt box: {} mAP | det box: {} mAP #####'.format(gt_AP, detbox_AP))


if __name__ == "__main__":

    if opt.world_size > num_gpu:
        #print(f'Wrong world size. Changing it from {opt.world_size} to {num_gpu}.')
        print('Wrong world size. Changing it from {} to {}.'.format(opt.world_size, num_gpu))
        opt.world_size = num_gpu
    main()
