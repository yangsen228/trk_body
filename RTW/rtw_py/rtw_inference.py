import os
import copy
import joblib
import numpy as np

from feature_engineering import get_feat_offset, feature_generation
from utils import *

skeleton = [(8,8),(10,8),(12,10),(14,12),(9,8),(11,9),(13,11),(1,8),(0,1),(2,1),(4,2),(6,4),(3,1),(5,3),(7,5)]

def random_walk(regressor, L, img, start_point, torso_z, feats, n_steps, step_size):
    qm = np.zeros((n_steps + 1, 3))
    qm[0] = start_point
    joint_pred = np.zeros(3)

    for i in range(n_steps):
        f = feature_generation(img, qm[i], torso_z, feats, mode='test').reshape(1,-1)
        leaf_id = regressor.apply(f)[0]

        idx = np.random.choice(L[leaf_id][0].shape[0], p=L[leaf_id][0])  # L[leaf_id][0] = weights
        u = L[leaf_id][1][idx]  # L[leaf_id][1] = centers

        qm[i+1] = qm[i] + u * step_size
        qm[i+1][0] = np.clip(qm[i+1][0], 0, W-1) # limit x between 0 and W
        qm[i+1][1] = np.clip(qm[i+1][1], 0, H-1) # limit y between 0 and H
        qm[i+1][2] = img[int(qm[i+1][1]), int(qm[i+1][0])] / 1000.0
        joint_pred += qm[i+1]

    joint_pred = joint_pred / n_steps
    if joint_pred[-1] > 3.8 or joint_pred[-1] < 1.8:
        joint_pred[-1] = torso_z
    return joint_pred

def rtw_tracking(regressors, Ls, img, start_point, n_steps, step_size):
    preds = np.zeros((N_JOINTS, 3))
    feats = get_feat_offset()
    for joint_id in range(N_JOINTS):
        preds[joint_id] = random_walk(regressors[joint_id], Ls[joint_id], img, start_point[joint_id], start_point[8][-1], feats, n_steps, step_size)
    return preds