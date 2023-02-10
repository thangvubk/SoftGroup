# Script de entrenamiento - testeo -visualización

# Antes de lanzar revisar si los archivos (dist_train.sh, train.py), (dist_train,
# test.py) y visualization.py están con los parámetros y rutas adecuados.

# Entrenamiento:
./../tools/dist_train.sh

# Testeo:
./../tools/dist_test.sh

# Visualización:
python ../tools/visualization.py --prediction_path '/home/lino/Documentos/GITHUB/SoftGroup_Multiespectral/work_dirs/prueba/predicciones' --room_name '25_points_GTv3_03' --task 'semantic_pred' --out 'nube_segmentada.ply'

