import glob
import math
import os

import numpy as np
import torch

data_folder = os.path.join(
    os.path.dirname(os.getcwd()), 'dataset', 'Synthetic_v3_InstanceSegmentation', 'train')
files = sorted(glob.glob(data_folder + '/*.pth'))
numclass = 15
semanticIDs = []
for i in range(numclass):
    semanticIDs.append(i)

class_numpoint_mean_dict = {}
class_radius_mean = {}
for semanticID in semanticIDs:
    class_numpoint_mean_dict[semanticID] = []
    class_radius_mean[semanticID] = []
num_points_semantic = np.array([0 for i in range(numclass)])

for file in files:
    coords, colors, sem_labels, instance_labels = torch.load(file)
    points = np.concatenate(
        [coords, colors, sem_labels[:, None].astype(int), instance_labels[:, None].astype(int)],
        axis=1)
    for semanticID in semanticIDs:
        singleSemantic = points[np.where(points[:, 6] == semanticID)]
        uniqueInstances, counts = np.unique(singleSemantic[:, 7], return_counts=True)
        for count in counts:
            class_numpoint_mean_dict[semanticID].append(count)
        allRadius = []
        for uniqueInstance in uniqueInstances:
            eachInstance = singleSemantic[np.where(singleSemantic[:, 7] == uniqueInstance)]
            radius = (np.max(eachInstance, axis=0) - np.min(eachInstance, axis=0)) / 2
            radius = math.sqrt(radius[0]**2 + radius[1]**2 + radius[2]**2)
            class_radius_mean[semanticID].append(radius)

    uniqueSemantic, semanticCount = np.unique(points[:, 6], return_counts=True)
    uniqueSemanticCount = np.array([0 for i in range(numclass)])
    uniqueSemantic = uniqueSemantic.astype(int)
    indexOf100 = np.where(uniqueSemantic == -100)
    semanticCount = np.delete(semanticCount, indexOf100)
    uniqueSemantic = np.delete(uniqueSemantic, indexOf100)
    uniqueSemanticCount[uniqueSemantic] = semanticCount
    num_points_semantic += uniqueSemanticCount

class_numpoint_mean_list = []
class_radius_mean_list = []
for semanticID in semanticIDs:
    class_numpoint_mean_list.append(
        sum(class_numpoint_mean_dict[semanticID]) * 1.0 / len(class_numpoint_mean_dict[semanticID]))
    class_radius_mean_list.append(
        sum(class_radius_mean[semanticID]) / len(class_radius_mean[semanticID]))

print('Using the printed list in hierarchical_aggregation.cpp for class_numpoint_mean_dict: ')
print([1.0] + [float('{0:0.0f}'.format(i)) for i in class_numpoint_mean_list][1:], sep=',')
print('Using the printed list in hierarchical_aggregation.cu for class_radius_mean: ')
print([1.0] + [float('{0:0.2f}'.format(i)) for i in class_radius_mean_list][1:], sep='')

# make ground to 1 the make building to 1
maxSemantic = np.max(num_points_semantic)
num_points_semantic = maxSemantic / num_points_semantic
num_points_semantic = num_points_semantic / num_points_semantic[1]
print('Using the printed list in hais_run_stpls3d.yaml for class_weight')
print([1.0, 1.0] + [float('{0:0.2f}'.format(i)) for i in num_points_semantic][2:], sep='')
