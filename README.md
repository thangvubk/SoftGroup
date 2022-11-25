# SoftGroup
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/softgroup-for-3d-instance-segmentation-on/3d-instance-segmentation-on-scannetv2)](https://paperswithcode.com/sota/3d-instance-segmentation-on-scannetv2?p=softgroup-for-3d-instance-segmentation-on) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/softgroup-for-3d-instance-segmentation-on/3d-instance-segmentation-on-s3dis)](https://paperswithcode.com/sota/3d-instance-segmentation-on-s3dis?p=softgroup-for-3d-instance-segmentation-on) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/softgroup-for-3d-instance-segmentation-on/3d-object-detection-on-scannetv2)](https://paperswithcode.com/sota/3d-object-detection-on-scannetv2?p=softgroup-for-3d-instance-segmentation-on)
![Architecture](./docs/architecture.png)

We provide code for reproducing results of two papers 

[**SoftGroup for 3D Instance Segmentation on Point Clouds**](https://arxiv.org/abs/2203.01509)\
Thang Vu, Kookhoi Kim, Tung M. Luu, Thanh Nguyen, and Chang D. Yoo.\
**CVPR 2022 (Oral)**.

[**Scalable SoftGroup for 3D Instance Segmentation on Point Clouds**](https://arxiv.org/abs/2209.08263)\
Thang Vu, Kookhoi Kim, Tung M. Luu, Thanh Nguyen, Junyeong Kim, and Chang D. Yoo.\
**arXiv preprint 2022**.

## Update
- 25/Nov/2022: Support [SoftGroup++](https://arxiv.org/abs/2209.08263).
- 12/Sep/2022: Support panoptic segmentation on SemanticKITTI dataset.
- 28/Jun/2022: Support STPLS3D dataset. Add custom dataset guideline.
- 16/Apr/2022: The code base is refactored. Coding is more extendable, readable, and consistent. The following features are supported:
  - Support up-to-date pytorch 1.11 and spconv 2.1.
  - Support distributed and mix precision training. Training time on ScanNet v2 (on 4GPUs) reduces from 4 day to 10 hours.
  - Faster inference speed, which requires only 288 ms per ScanNet scan on single Titan X.

## Introduction

Existing state-of-the-art 3D instance segmentation methods perform semantic segmentation followed by grouping. The hard predictions are made when performing semantic segmentation such that each point is associated with a single class. However, the errors stemming from hard decision propagate into grouping that results in (1) low overlaps between the predicted instance with the ground truth and (2) substantial false positives. To address the aforementioned problems, this paper proposes a 3D instance segmentation method referred to as SoftGroup by performing bottom-up soft grouping followed by top-down refinement. SoftGroup allows each point to be associated with multiple classes to mitigate the problems stemming from semantic prediction errors and suppresses false positive instances by learning to categorize them as background. Experimental results on different datasets and multiple evaluation metrics demonstrate the efficacy of SoftGroup. Its performance surpasses the strongest prior method by a significant margin of +6.2% on the ScanNet v2 hidden test set and +6.8% on S3DIS Area 5 of AP_50.

![Learderboard](./docs/leaderboard.png)

## Feature
* State of the art performance on the [ScanNet benchmark](http://kaldir.vc.in.tum.de/scannet_benchmark/semantic_instance_3d) and S3DIS dataset (3/Mar/2022).
* High speed of 345 ms per scan on ScanNet dataset, which is comparable with the existing fastest methods ([HAIS](https://github.com/hustvl/HAIS)). Our refactored implementation (this code) further reduce the inference time to 288 ms per scan.
* Support multiple datasets: ScanNet, S3DIS, STPLS3D, SemanticKITTI.

## Installation
Please refer to [installation guide](docs/installation.md).

## Data Preparation
Please refer to [data preparation](dataset/README.md).

## Pretrained models

### Instance segmentation

|   Dataset  |   Model     |   AP  | AP_50 | AP_25 |                                           Download                                         |
|:----------:|:-----------:|:----:|:-----:|:-----:|:-------------------------------------------------------------------------------------------:|
|    S3DIS   | SoftGroup   | 51.4 |  66.5 |  75.4 | [model](https://drive.google.com/file/d/1-f7I6-eIma4OilBON928N6mVcYbhiUFP/view?usp=sharing) |
|    S3DIS   | SoftGroup++ | 50.9 |  67.8 |  76.0 | [model](https://drive.google.com/file/d/1OLbC8lmWkAQbqYAjiFj84egLQmJr-PmQ/view?usp=sharing) |
| ScanNet v2 | SoftGroup   | 45.8 |  67.4 |  79.1 | [model](https://drive.google.com/file/d/1XUNRfred9QAEUY__VdmSgZxGQ7peG5ms/view?usp=sharing) |
| ScanNet v2 | SoftGroup++ | 45.9 |  67.9 |  79.4 | above |
|  STPLS3D   | SoftGroup   | 47.3 |  63.1 |  71.4 | [model](https://drive.google.com/file/d/1xCkKLTCYtQmSjXYH_sSg21M_6dcAskd8/view?usp=sharing) |
|  STPLS3D   | SoftGroup++ | 46.5 |  62.9 |  71.8 | above |

> **_NOTE:_**  SoftGroup and SoftGroup++ use can use same trained model for inference on ScanNet v2 and STPLS3D.

### Panoptic segmentation

|    Dataset    |  PQ  | Config | Model |
|:-------------:|:----:|:------:|:-----:|
| SemanticKITTI | 60.2 | [config](https://github.com/thangvubk/SoftGroup/blob/main/configs/softgroup_kitti.yaml) | [model](https://drive.google.com/file/d/10Ln-xLfl8Z3DX3G3lnO_RruJtYUYDfI7/view?usp=sharing)     |

## Training
We use the checkpoint of [HAIS](https://github.com/hustvl/HAIS) as pretrained backbone. **We have already converted the checkpoint to work on ``spconv2.x``**. Download the pretrained HAIS-spconv2 model and put it in ``SoftGroup/`` directory.

Converted hais checkpoint: [model](https://drive.google.com/file/d/1FABsCUnxfO_VlItAzDYAwurdfcdK-scs/view?usp=sharing)

Noted that for fair comparison with implementation in STPLS3D paper, we train SoftGroup on this dataset from scratch without pretrained backbone.
### Training S3DIS dataset
The default configs suppose training on 4 GPU. If you use smaller number of GPUs, you should reduce the learning rate linearly. 

First, finetune the pretrained HAIS point-wise prediction network (backbone) on S3DIS.
```
./tools/dist_train.sh configs/softgroup_s3dis_backbone_fold5.yaml 4
```
Then, train model from frozen backbone.
```
./tools/dist_train.sh configs/softgroup_s3dis_fold5.yaml 4
```

### Training ScanNet V2 dataset
Training on ScanNet doesnot require finetuning the backbone. Just freeze pretrained backbone and train the model.
```
./tools/dist_train.sh configs/softgroup_scannet.yaml 4
```

### Training STPLS3D dataset
```
./tools/dist_train.sh configs/softgroup_stpls3d_backbone.yaml 4
./tools/dist_train.sh configs/softgroup_stpls3d.yaml 4
```

## Inference
```
./tools/dist_test.sh $CONFIG_FILE $CHECKPOINT $NUM_GPU
```

### Inference without label
For example, on scannet test split, just change [``prefix``](https://github.com/thangvubk/SoftGroup/blob/cf88d9be41ae83a70f9100856a3ca15ee4ddcee9/configs/softgroup_scannet.yaml#L49) to ``test`` and [``with_label``](https://github.com/thangvubk/SoftGroup/blob/cf88d9be41ae83a70f9100856a3ca15ee4ddcee9/configs/softgroup_scannet.yaml#L52) to ``False`` before running inference. 

### Bounding box evaluation of ScanNet V2 dataset.
We provide script to evaluate detection performance on axis-aligned boxes from predicted/ground-truth instance.
- Step 1: Change ``save_instance`` to ``True`` in [config file](https://github.com/thangvubk/SoftGroup/blob/99ffb9756e553e0edfb2c43e2ab6a6f646892bb5/config/softgroup_default_scannet.yaml#L72).
- Step 2: Run evaluation code.
```
CUDA_VISIBLE_DEVICES=0 python test.py --config config/softgroup_default_scannet.yaml --pretrain $PATH_TO_PRETRAIN_MODEL$
```
- Step 3: Evaluate detection performance.
```
python eval_det.py
```

## Visualization
Please refer to [visualization guide](docs/visualization.md) for visualizing ScanNet and S3DIS results.

## Custom dataset
Please refer to [custom dataset guide](docs/custom_dataset.md).

## Citation
If you find our work helpful for your research. Please consider citing our paper.

```
@inproceedings{vu2022softgroup,
  title={SoftGroup for 3D Instance Segmentation on 3D Point Clouds},
  author={Vu, Thang and Kim, Kookhoi and Luu, Tung M. and Nguyen, Xuan Thanh and Yoo, Chang D.},
  booktitle={CVPR},
  year={2022}
}
```
## Acknowledgements
Code is built based on [HAIS](https://github.com/hustvl/HAIS), [PointGroup](https://github.com/dvlab-research/PointGroup), and [spconv](https://github.com/traveller59/spconv)

This work was partly supported by Institute for Information communications Technology Planning Evaluation (IITP) grant funded by the Korea government (MSIT) (2021-0-01381, Development of Causal AI through Video Understanding, and partly supported by Institute of Information \& Communications Technology Planning \& Evaluation (IITP) grant funded by the Korea government (MSIT) (No. 2019-0-01371, Development of brain-inspired AI with human-like intelligence).
