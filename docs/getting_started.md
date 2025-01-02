# Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

# Train and Test

Train BEVFormer with 8 GPUs 
```
./tools/dist_train.sh ./projects/configs/bevformer/bevformer_tiny.py 1
```

Eval BEVFormer with 8 GPUs
```
./tools/dist_test.sh ./projects/configs/bevformer/bevformer_base.py ./path/to/ckpts.pth 8

export PYTHONPATH=$(pwd)/projects:$PYTHONPATH
python ./tools/test.py ./projects/configs/bevformer/bevformer_tiny.py work_dirs/bevformer_tiny/latest.pth --eval bbox
```
Note: using 1 GPU to eval can obtain slightly higher performance because continuous video may be truncated with multiple GPUs. By default we report the score evaled with 8 GPUs.



# Using FP16 to train the model.
The above training script can not support FP16 training, 
and we provide another script to train BEVFormer with FP16.

```
./tools/fp16/dist_train.sh ./projects/configs/bevformer_fp16/bevformer_tiny_fp16.py 8
```


# Visualization 

see [visual.py](../tools/analysis_tools/visual.py)