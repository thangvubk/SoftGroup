```
model:
  channels: 32  # number of base channel for the backbone network
  num_blocks: 7  # number of backbone blocks
  semantic_classes: 20  # number of semantic classes
  instance_classes: 18  # number of instance classes (semantic class can be less than instance classes)
  sem2ins_classes: []  # class index to get instance directly from semantic. This trick is applied on S3DIS floor and ceil since 1 scene ussualy contain only 1 floor and 1 ceil.
  semantic_only: False  # If true, only the point-wise networks (semantic and offset) are trained. The top-down refinement stage is not taken into consideration. Set it to True ussualy using for pretraining the backbone. 
  ignore_label: -100  # Ignore label while training
  grouping_cfg:
    score_thr: 0.2  # soft score threshold using in softgroup.
    radius: 0.04  # the search radius of K-nearest neighbor using for grouping.
    mean_active: 300  # using to constrain the total size after K-NN
    class_numpoint_mean: [-1., -1., 3917., 12056., 2303.,
                          8331., 3948., 3166., 5629., 11719.,
                          1003., 3317., 4912., 10221., 3889.,
                          4136., 2120., 945., 3967., 2589.]  # the mean number of point per instance for each class.
    npoint_thr: 0.05  # absolute if class_numpoint == -1, relative if class_numpoint != -1  #  when grouping, if num_point[i] > npoint_thr * class_numpoint_mean[i], the cluster is consider as an instance, otherwise discarded. 
    ignore_classes: [0, 1]  # classes will be ignored while perform grouping.
  instance_voxel_cfg:
    scale: 50  # scaling factor, voxel size = 1 / scale. In this case voxel_size = 1/50 = 0.02m
    spatial_shape: 20  # the dimension of instance in terms of voxels, i.e., H, W, D of instance will be 20 voxels. 
  train_cfg:
    max_proposal_num: 200  # if number of proposals > max_proposal_num while training, the number of propsoals will be truncated to reduce memory usage.
    pos_iou_thr: 0.5  # intersection over union threshold to identify positive and negative samples.
  test_cfg:
    x4_split: False  # whether divide the scene into 4 part then merge the results. This is used for S3DIS dataset since the scene is very big.
    cls_score_thr: 0.001  # score threshold for postprocessing
    mask_score_thr: -0.5  # threshold to classify background and foreground in segmentation
    min_npoint: 100  # min number of points for each instance
  fixed_modules: ['input_conv', 'unet', 'output_layer', 'semantic_linear', 'offset_linear']  # These module will not have gradient updates while training. 

data:
  train:
    type: 'scannetv2'  # dataset type
    data_root: 'dataset/scannetv2' # root path to your data
    prefix: 'train'  # data prefix
    suffix: '_inst_nostuff.pth'  # data suffix
    training: True  # training mode
    repeat: 4  # repeat factor for the data. In case the dataset is small, using repeat to avoid data loading every epoch -> reduce loading time.
    voxel_cfg:
      scale: 50  # scaling factor, voxel size = 1 / scale. In this case voxel_size = 1/50 = 0.02m
      spatial_shape: [128, 512]  # min and max spatial shape of the whole scene after random crop
      max_npoint: 250000  # max number of points after random crop
      min_npoint: 5000  # min number of points after random crop
  test:
    type: 'scannetv2'  # test data type
    data_root: 'dataset/scannetv2' # test data root
    prefix: 'val'  # data prefix
    suffix: '_inst_nostuff.pth'  # data suffix
    training: False  # test mode
    voxel_cfg:
      scale: 50 # scaling factor, voxel size = 1 / scale. In this case voxel_size = 1/50 = 0.02m
      spatial_shape: [128, 512]  # no effect during testing
      max_npoint: 250000  # no effect during testing
      min_npoint: 5000  # no effect during testing

dataloader:
  train:
    batch_size: 4  # train batch size
    num_workers: 4  # train number of processes to load data
  test:
    batch_size: 1  # test batch size
    num_workers: 1  # test number of processes to load data

optimizer:
  type: 'Adam'  # optimizer type
  lr: 0.004  # learning rate

save_cfg:
  semantic: True  # weather saving semantic while evaluation
  offset: True  # weather saving offset while evaluation
  instance: True  # weather saving instance while evaluation

fp16: False  # Mix precision training
epochs: 128  # Number of total epochs
step_epoch: 50  # Epoch to step learning rate
save_freq: 4  # frequency to save model and perform validation
pretrain: './hais_ckpt_spconv2.pth'  # pretrain model path
work_dir: ''  # directory to save model and log
```
