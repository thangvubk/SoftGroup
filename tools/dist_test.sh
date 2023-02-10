#!/usr/bin/env bash
#CONFIG=$1
#CHECK_POINT=$2
#GPUS=$3
#PORT=${PORT:-29501}
#GUARDADO='/home/lino/Documentos/GITHUB/SoftGroup_Multiespectral/work_dirs/softgroup_stpls3d_backbone'

CONFIG='/home/lino/Documentos/GITHUB/SoftGroup_Multiespectral/configs/softgroup/softgroup_stpls3d_backbone.yaml'
CHECK_POINT='/home/lino/Documentos/GITHUB/SoftGroup_Multiespectral/work_dirs/prueba/epoch_9.pth'
GPUS=$3
PORT=${PORT:-29501}
GUARDADO='/home/lino/Documentos/GITHUB/SoftGroup_Multiespectral/work_dirs/prueba/predicciones'

#OMP_NUM_THREADS=1 torchrun --nproc_per_node=$GPUS --master_port=$PORT  $(dirname "$0")/test.py $CONFIG $CHECK_POINT --dist ${@:4}
OMP_NUM_THREADS=1 torchrun --master_port=$PORT  $(dirname "$0")/test.py $CONFIG $CHECK_POINT --dist ${@:4} --out $GUARDADO
