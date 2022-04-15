## Visualization

Before visualization, you need to change ``save_semantic``, ``save_pt_offsets``, ``save_instance`` to True in the config file and run the inference to write the output predictions.

There are two options for visualization:

- Visualization using through a pop-up using open3D (default). Prerequisite: ``pip install open3D==0.8.0``

- Write point clouds to ``.ply`` file then use an visualization application such as [MeshLab](https://www.meshlab.net/) to see the results. Just pass the arg ``--out YOUR_FILE.ply`` to enable this option.

After inference, run visualization by execute the following command

```
python visualization.py --dataset {} --prediction_path --split {} --scene_name {} --task {} --out {}

usage: visualization.py [-h] [--dataset {scannet,s3dis}]
                        [--prediction_path PREDICTION_PATH]
                        [--data_split DATA_SPLIT] [--room_name ROOM_NAME]
                        [--task TASK] [--out OUT]

optional arguments:
  --dataset {scannet,s3dis}
                        dataset for visualization
  --prediction_path PREDICTION_PATH
                        path to the prediction results
  --data_split DATA_SPLIT
                        train/val/test for scannet or Area_ID for s3dis
  --room_name ROOM_NAME
                        room_name
  --task TASK           input / semantic_gt / semantic_pred /
                        offset_semantic_pred / instance_gt / instance_pred
  --out OUT             output point cloud file in FILE.ply format
```
