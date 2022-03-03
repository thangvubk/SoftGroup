
# SoftGroup

![Architecture](./docs/architecture.png)

We provide code for reproducing results of the paper **SoftGroup for 3D Instance Segmentation on Point Clouds (CVPR 2022)**

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

*  Verify the version of spconv.
  
      spconv 1.0, compatible with CUDA < 11 and pytorch < 1.5, is already recursively cloned in `SoftGroup/lib/spconv` in step 2) by default. 

      For higher version CUDA and pytorch, spconv 1.2 is suggested. Replace `SoftGroup/lib/spconv` with this fork of spconv.

```
git clone https://github.com/outsidercsy/spconv.git --recursive
```

      Note:  In the provided spconv 1.0 and 1.2, spconv\spconv\functional.py is modified to make grad_output contiguous. Make sure you use the modified spconv but not the original one. Or there would be some bugs of optimization.


*  Install the dependent libraries.
```
conda install libboost
conda install -c daleydeng gcc-5 # (optional, install gcc-5.4 in conda env)
```

* Compile the spconv library.
```
cd SoftGroup/lib/spconv
python setup.py bdist_wheel
```

* Intall the generated .whl file.
```
cd SoftGroup/lib/spconv/dist
pip install {wheel_file_name}.whl
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

2\) Put the data in the corresponding folders. 
* Copy the files `[scene_id]_vh_clean_2.ply`,  `[scene_id]_vh_clean_2.labels.ply`,  `[scene_id]_vh_clean_2.0.010000.segs.json`  and `[scene_id].aggregation.json`  into the `dataset/scannetv2/train` and `dataset/scannetv2/val` folders according to the ScanNet v2 train/val [split](https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark). 

* Copy the files `[scene_id]_vh_clean_2.ply` into the `dataset/scannetv2/test` folder according to the ScanNet v2 test [split](https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark). 

* Put the file `scannetv2-labels.combined.tsv` in the `dataset/scannetv2` folder.

The dataset files are organized as follows.
```
SoftGroup
├── dataset
│   ├── scannetv2
│   │   ├── train
│   │   │   ├── [scene_id]_vh_clean_2.ply & [scene_id]_vh_clean_2.labels.ply & [scene_id]_vh_clean_2.0.010000.segs.json & [scene_id].aggregation.json
│   │   ├── val
│   │   │   ├── [scene_id]_vh_clean_2.ply & [scene_id]_vh_clean_2.labels.ply & [scene_id]_vh_clean_2.0.010000.segs.json & [scene_id].aggregation.json
│   │   ├── test
│   │   │   ├── [scene_id]_vh_clean_2.ply 
│   │   ├── scannetv2-labels.combined.tsv
```

3\) Generate input files `[scene_id]_inst_nostuff.pth` for instance segmentation.
```
cd SoftGroup/dataset/scannetv2
python prepare_data_inst.py --data_split train
python prepare_data_inst.py --data_split val
python prepare_data_inst.py --data_split test
```

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
