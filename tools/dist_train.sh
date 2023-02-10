#!/usr/bin/env bash
#CONFIG=$1
#GPUS=$2
#PORT=${PORT:-29500}
#RESUME=$/home/lino/Documentos/GITHUB/SoftGroup_Multiespectral/work_dirs/softgroup_stpls3d_backbone/epoch_20.pth
#OMP_NUM_THREADS=1 torchrun --nproc_per_node=$GPUS --master_port=$PORT $(dirname "$0")/train.py --resume=$RESUME --dist $CONFIG ${@:3}

# De aquí para abajo hecho por mi:

CONFIG='/home/lino/Documentos/GITHUB/SoftGroup_Multiespectral/configs/softgroup/softgroup_stpls3d_backbone.yaml'
GPUS=$2
PORT=${PORT:-29500}
RESUME='/home/lino/Documentos/GITHUB/SoftGroup_Multiespectral/work_dirs/softgroup_stpls3d_backbone/epoch_1.pth'
WORK_DIR='/home/lino/Documentos/GITHUB/SoftGroup_Multiespectral/work_dirs/prueba'


# Si quiero continuar desde un checkpoint hago así:
OMP_NUM_THREADS=1 torchrun --master_port=$PORT $(dirname "$0")/train.py $CONFIG --dist --resume $RESUME --work_dir $WORK_DIR ${@:3}

# Si empezamos de cero hacemos así:
#OMP_NUM_THREADS=1 torchrun --master_port=$PORT $(dirname "$0")/train.py $CONFIG --dist --work_dir $WORK_DIR ${@:3}
