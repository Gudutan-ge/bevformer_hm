import torch
import time
import argparse
from mmcv import Config
import os.path as osp
from mmcv.parallel import MMDataParallel
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmcv.runner import load_checkpoint, get_dist_info


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    args = parser.parse_args()
    return args

def main():
    '''
    args = SimpleNamespace(
        config = './projects/configs/bevformer/bevformer_tiny.py',  # 模型配置
        checkpoint = 'work_dirs/bevformer_tiny/latest.pth',         # 权重
        eval = ['bbox'],    # 评估指标
        show = False,       # 是否保存可视化结果
        show_dir = None,    # 可视化结果保存路径
    ) # 运行参数设定
    '''
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if cfg.get('cudnn_benchmark', False): # 设置 CUDNN 的 Benchmark 模式
        torch.backends.cudnn.benchmark = True
    
    if cfg.get('close_tf32', False): # TF32 运算控制
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    
    cfg.model.pretrained = None #  取消预训练权重加载
    # in case the test dataset is concatenated
    samples_per_gpu = 1 # 每 GPU 加载的样本数
    
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    dataset = build_dataset(cfg.data.test) # 构建数据集
    data_loader = build_dataloader( # 构建数据加载器
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg')) # 构建模型
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu') # 加载权重
    
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES'] # this
    else:
        model.CLASSES = dataset.CLASSES
    
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE'] # this
    elif hasattr(dataset, 'PALETTE') and dataset.PALETTE is not None:
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE
    
    model = MMDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)

    rank, _ = get_dist_info()
    if rank == 0:
        kwargs = {}
        kwargs['jsonfile_prefix'] = osp.join('test', args.config.split(
            '/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))

if __name__ == '__main__':
    main()