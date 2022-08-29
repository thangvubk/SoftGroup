## Visualization

Before visualization, you need to write the output results of inference, for example:

```
./tools/dist_test.sh $CONFIG $CHECKPOINT $NUM_GPUs --out results/
```

There are two options for visualization:

- Visualization using through a pop-up using open3D (default). Prerequisite: ``pip install open3D==0.8.0``

- Write point clouds to ``.ply`` file then use an visualization application such as [MeshLab](https://www.meshlab.net/) to see the results. Just pass the arg ``--out YOUR_FILE.ply`` to enable this option.

After inference, run visualization by execute the following command. (given that predictions are saved in ``results/`` directory.

```
python visualization.py --prediction_path results/ --room_name {} --task {} --out {}

usage: visualization.py [-h] [--prediction_path PREDICTION_PATH]
                        [--room_name ROOM_NAME] [--task TASK] [--out OUT]

optional arguments:
  -h, --help            show this help message and exit
  --prediction_path PREDICTION_PATH
                        path to the prediction results
  --room_name ROOM_NAME
                        room_name
  --task TASK           input/semantic_gt/semantic_pred/offset_semantic_pred/i
                        nstance_gt/instance_pred
  --out OUT             output point cloud file in FILE.ply format

```
