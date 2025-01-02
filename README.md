# 使用说明

### 0. 大作业说明

本大作业论文翻译自论文[BEVFormer: Learning Bird’s-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers](https://arxiv.org/pdf/2203.17270v2.pdf)，会议时间：ECCV 2022，测试代码位于`tools/test.py`。

### 1. 环境配置

**a. Create a conda virtual environment and activate it.**

```shell
conda create -n open-mmlab python=3.8 -y
conda activate open-mmlab
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**

```shell
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# Recommended torch>=1.9

```

**c. Install gcc>=5 in conda env (optional).**

```shell
conda install -c omgarcia gcc-6 # gcc-6.2
```

**c. Install mmcv-full.**

```shell
pip install mmcv-full==1.4.0
mim install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0 index.html
```

**d. Install mmdet and mmseg.**

```shell
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

**e. Install mmdet3d from source code.**

```shell
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 # Other versions may not be compatible.
python setup.py install
```

**f. Install Detectron2 and Timm.**

```shell
pip install einops fvcore seaborn iopath==0.1.9 timm==0.6.13  typing-extensions==4.5.0 pylint ipython==8.12  numpy==1.19.5 matplotlib==3.5.2 numba==0.48.0 pandas==1.4.4 scikit-image==0.19.3 setuptools==59.5.0
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

**g. Prepare pretrained models.**

```shell
cd bevformer
mkdir ckpts

cd ckpts & wget https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth
```

note: this pretrained model is the same model used in [detr3d](https://github.com/WangYueFt/detr3d) ./work_dirs/bevformer_tiny/latest.pth

### 2.数据集

Download nuScenes V1.0 full dataset data  and CAN bus expansion data [HERE](https://www.nuscenes.org/download). Prepare nuscenes data by running


**Download CAN bus expansion**

```
# download 'can_bus.zip'
unzip can_bus.zip 
# move can_bus to data dir
```

**Prepare nuScenes data**

*We genetate custom annotation files which are different from mmdet3d's*

```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0 --canbus ./data
```

Using the above code will generate `nuscenes_infos_temporal_{train,val}.pkl`.

**Folder structure**

```
bevformer
├── projects/
├── tools/
├── configs/
├── ckpts/
│   ├── r101_dcn_fcos3d_pretrain.pth
├── data/
│   ├── can_bus/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── nuscenes_infos_temporal_train.pkl
|   |   ├── nuscenes_infos_temporal_val.pkl
```

### 3.测试

代码的复现工作在个人电脑上完成，因此只能使用1个GPU，因此只能够复现出有限条件下的测试。本人复现的测试代码是基于bevformer_tiny架构下的1个GPU的结果。

可以通过以下代码在项目目录下运行测试代码：

```
./tools/dist_test.sh ./projects/configs/bevformer/bevformer_tiny.py ./path/to/ckpts.pth
```

提供可以选择的参数有：

- eval：评估指标
- show = False：是否保存可视化结果
- show_dir = None：可视化结果保存路径

分别使用提供的权重`bevformer_tiny_epoch_24.pth`和本人在mini nuscenes数据集2个epoch的训练权重的测试结果如下：

| 模型           | checkpoint    | 数据集       | NDS↑   | mAP↑   | mATE↓  | mASE↓  | mAOE↓  | mAVE↓  | mAAE↓  |
| -------------- | ------------- | ------------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| bevformer_tiny | tiny_epoch_24 | nuscene_full | 0.3014 | 0.2440 | 0.8706 | 0.4690 | 0.7892 | 0.7528 | 0.3245 |
| bevformer_tiny | tiny_epoch_2  | nuscene_mini | 0.0335 | 0.0009 | 1.1129 | 0.8802 | 1.1530 | 0.9236 | 0.8658 |

可以看到两个tiny模型表现相较于论文中的结果差距十分明显，而本人自己在mini数据集，2个epoch下的训练结果与提供的权重效果差距明显。
