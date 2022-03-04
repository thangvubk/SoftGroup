
# SoftGroup

![Architecture](./docs/architecture.png)

We provide code for reproducing results of the paper [SoftGroup for 3D Instance Segmentation on Point Clouds (CVPR 2022)](https://arxiv.org/abs/2203.01509)

Author: Thang Vu, Kookhoi Kim, Tung M. Luu, Xuan Thanh Nguyen, and Chang D. Yoo.


## Introduction

Existing state-of-the-art 3D instance segmentation methods perform semantic segmentation followed by grouping. The hard predictions are made when performing semantic segmentation such that each point is associated with a single class. However, the errors stemming from hard decision propagate into grouping that results in (1) low overlaps between the predicted instance with the ground truth and (2) substantial false positives. To address the aforementioned problems, this paper proposes a 3D instance segmentation method referred to as SoftGroup by performing bottom-up soft grouping followed by top-down refinement. SoftGroup allows each point to be associated with multiple classes to mitigate the problems stemming from semantic prediction errors and suppresses false positive instances by learning to categorize them as background. Experimental results on different datasets and multiple evaluation metrics demonstrate the efficacy of SoftGroup. Its performance surpasses the strongest prior method by a significant margin of +6.2% on the ScanNet v2 hidden test set and +6.8% on S3DIS Area 5 of AP_50.
  
![Learderboard](./docs/leaderboard.png)


## Feature

* State of the art performance on the [ScanNet benchmark](http://kaldir.vc.in.tum.de/scannet_benchmark/semantic_instance_3d) and S3DIS dataset (3/Mar/2022).
* High speed of 345 ms per scan on ScanNet dataset, which is comparable with the existing fastest methods ([HAIS](https://github.com/hustvl/HAIS)).
* Reproducibility code for both ScanNet and S3DIS datasets.



## Installation

1\) Environment

* Python 3.x
* Pytorch 1.1 or higher
* CUDA 9.2 or higher
* gcc-5.4 or higher

Create a conda virtual environment and activate it.
```
conda create -n softgroup python=3.7
conda activate softgroup
```


2\) Clone the repository.
```
git clone https://github.com/thangvubk/SoftGroup.git --recursive
```

  
3\) Install the requirements.
```
cd SoftGroup
pip install -r requirements.txt
conda install -c bioconda google-sparsehash 
```

4\) Install spconv 


*  Install the dependencies.
```
sudo apt-get install libboost-all-dev
sudo apt-get install libsparsehash-dev

```

* Compile the spconv library.
```
cd SoftGroup/lib/spconv
python setup.py bdist_wheel
pip install dist/{WHEEL_FILE_NAME}.whl
```


5\) Compile the external C++ and CUDA ops.
```
cd SoftGroup/lib/softgroup_ops
export CPLUS_INCLUDE_PATH={conda_env_path}/softgroup/include:$CPLUS_INCLUDE_PATH
python setup.py build_ext develop
```
{conda_env_path} is the location of the created conda environment, e.g., `/anaconda3/envs`.



## Data Preparation

1\) Download the [ScanNet](http://www.scan-net.org/) v2 dataset.

2\) Put the downloaded ``scans`` and ``scans_test`` folder as follows.

```
SoftGroup
├── dataset
│   ├── scannetv2
│   │   ├── scans
│   │   ├── scans_test
```

3\) Split and preprocess data
```
cd SoftGroup/dataset/scannetv2
bash prepare_data.sh
```

The script data into train/val/test folder and preprocess the data. After running the script the scannet dataset structure should look like below.
```
SoftGroup
├── dataset
│   ├── scannetv2
│   │   ├── scans
│   │   ├── scans_test
│   │   ├── train
│   │   ├── val
│   │   ├── test
│   │   ├── val_gt
```

## Pretrained models

|   Dataset  |  AP  | AP_50 | AP_25 |                                           Download                                          |
|:----------:|:----:|:-----:|:-----:|:-------------------------------------------------------------------------------------------:|
|    S3DIS   |      |       |       | [model](https://drive.google.com/file/d/1RodfMTUC-0YWs47kx8lj-i0jbDyM9PO6/view?usp=sharing) |
| ScanNet v2 | 46.0 |  67.6 |  78.9 | [model](https://drive.google.com/file/d/1Gt1JUXXB-sBtAeuot29crAUnBwcXW7rN/view?usp=sharing) |

## Training
```
CUDA_VISIBLE_DEVICES=0 python train.py --config config/softgroup_default_scannet.yaml 
```


## Inference

1\) To evaluate on validation set, 

* prepare the `.txt` instance ground-truth files as the following.
```
cd dataset/scannetv2
python prepare_data_inst_gttxt.py
```

* set `split` and `eval` in the config file as `val` and `True`. 
  
* Run the inference and evaluation code.
```
CUDA_VISIBLE_DEVICES=0 python test.py --config config/softgroup_default_scannet.yaml --pretrain $PATH_TO_PRETRAIN_MODEL$
```


## Visualization
We provide visualization tools based on Open3D (tested on Open3D 0.8.0).
```
pip install open3D==0.8.0
python visualize_open3d.py --data_path {} --prediction_path {} --data_split {} --room_name {} --task {}
```
Please refer to `visualize_open3d.py` for more details.



## Citation
If you find our work helpful for your research. Please cite our paper.

```
@inproceedings{vu2019softgroup,
  title={SoftGroup for 3D Instance Segmentation on 3D Point Clouds},
  author={Vu, Thang and Kim, Kookhoi and Luu, Tung M. and Nguyen, Xuan Thanh and Yoo, Chang D.},
  booktitle={CVPR},
  year={2022}
}
```
