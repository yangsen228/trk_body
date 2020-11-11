import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D

from utils import *

view = 'side'
mode = 'test'
N_STEPS = 96
STEP_SIZE = 2
model = '1026'

# Load data
processed_data_path = 'data/processed/ITOP'
indices = np.load(os.path.join(processed_data_path, 'ITOP_{}_{}_indices.npy'.format(view, mode)))
depth_images_full = np.load(os.path.join(processed_data_path, 'ITOP_{}_{}_depth_map_full.npy'.format(view, mode)))
joints = np.load(os.path.join(processed_data_path, 'ITOP_{}_{}_labels.npy'.format(view, mode)))                # (x,y)pixel, (z)m
joints_3d = np.load(os.path.join(processed_data_path, 'ITOP_{}_{}_labels_3d.npy'.format(view, mode)))          # (x,y,z)m
# preds_full = np.load(os.path.join(processed_data_path, 'ITOP_{}_{}_labels_a2j.npy'.format(view, mode)))        # (x,y,z)m
preds_full = np.load('output/preds/{}_{}_{}_{}_track_2020{}.npy'.format(view, mode, N_STEPS, STEP_SIZE, model))  # (x,y)pixel, (z)m
# preds_full = np.load('output/preds/tmp_train.npy')[0]
# preds_full = np.load('output/preds/{}_{}_{}_{}_a2j-track_2020{}_cnn_rf27.npy'.format(view, mode, N_STEPS, STEP_SIZE, model))
print('data shape: {}, {}'.format(joints.shape, preds_full.shape))
n_data = len(preds_full)
print(np.max(joints[:,:,2]), np.min(joints[:,:,2]))

# Extract data
if 1:
    preds_3d = [pixel2world(preds_full[i]) for i in indices]
else:
    preds_3d = [preds_full[i] for i in indices]
preds_3d = np.array(preds_3d)

# Evaluation
logger.debug('Evaluating results')

# Save gt-a2j labels
# for i, j in enumerate(indices):
#     preds_full[j] = joints_3d[i]
# np.save('output/preds/{}_{}_gt-a2j.npy'.format(view, mode), preds_full)

# preds_3d[:,:,-1] = joints_3d[:,:,-1]

# Calculate mean distance error
total_dist, total_mAP = [], []
for joint_id in range(N_JOINTS):
    dist = np.linalg.norm((joints_3d[:,joint_id,:] - preds_3d[:,joint_id,:]), axis=1)
    mAP_list = [ele for ele in dist if ele <= 0.1]
    mAP = len(mAP_list)/len(dist)
    mean_dist = dist.mean()
    total_dist.append(mean_dist)
    total_mAP.append(mAP)
    print('{} {} error: {}           10cm mAP: {}'.format(joint_id, JOINT_IDX[joint_id], mean_dist, mAP))
print('Overall error: {}     Overall 10cm mAP: {}'.format(np.array(total_dist).mean(), np.array(total_mAP).mean()))

# Visualization
# for image_id in range(n_data):
#     if image_id % 1000 == 0:
#         logger.debug('draw ({}/{}) image...'.format(image_id, n_data))
#     png_path = os.path.join('output/png/ITOP/{}_{}_{}_{}_track_2020{}'.format(view, mode, N_STEPS, STEP_SIZE, model), str(image_id) + '.png')
#     draw_preds(depth_images_full[image_id], preds_full[image_id], png_path)

