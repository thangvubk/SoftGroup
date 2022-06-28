# Tips for custom dataset.

## Data preparation.
- Step 1: split your data to ``train/val`` folder
- Step 2: for each scene construct a ``.pth`` file that contains:
  -  ``point XYZ coordinates: shape of (N, 3)``
  -  ``colors RGB: shape of (N, 3)``
  -  ``semantic labels: shape of (N, )``
  -  ``insance labels: shape(N, )``

Noted that colors should be normalized in range [-1, 1], see [here](https://github.com/thangvubk/SoftGroup/blob/fc1dcbca82b3b329215279c179cd6be2a63215db/dataset/scannetv2/prepare_data_inst.py#L56). 

## Config
The following configs may be modified for custom dataset.
- ``semantic_classes``: the number of class for semantic segmentation
- ``instance_classes``: the number of semantic classes considered for instance segmentation. For example, in ScanNet dataset [config](https://github.com/thangvubk/SoftGroup/blob/fc1dcbca82b3b329215279c179cd6be2a63215db/configs/softgroup_scannet.yaml#L4-L5), ``wall`` and ``floor`` is not considered for instance segmentation. So that ``instance_classes = semantic_classes - 2``.
- ``sem2ins_classes ``: use this when you directly use semantic segmentation results as instance segmentation results for specified classes. For example, in S3DIS dataset, class ``floor`` and ``ceil`` (index [0, 1]) are specified since most of the cases, each scene has only one floor and one ceil.
- ``class_numpoint_mean``: the number of points for each instance per class. shape of ``(semantic_classes, )``
- ``scale``: the point coordinates are scaled up for voxelization. From ``scale``, we can infer ``voxel_size = 1 / scale``. Indoor datasets often use scale = 50 (voxel_size = 0.02m). In outdoor datasets, the voxelize should be larger due to higher spasity. For example, in STPLS3D dataset, ``scale`` is set to 3 (voxel_size = 0.33m). Ablation may be needed to figure out which ``scale`` is most suitable to your dataset.
- ``grouping_cfg.radius``: The radius for grouping. This value is related to voxel_size. When the voxelize is higher, the radius should be also higher.
- ``grouping_cfg.ignore_classes``: the semantic class indices that are not considered for grouping.

For further information, you can compare the configs of [STPLS3D](https://github.com/thangvubk/SoftGroup/blob/main/configs/softgroup_stpls3d.yaml) and [ScanNet](https://github.com/thangvubk/SoftGroup/blob/main/configs/softgroup_scannet.yaml).
