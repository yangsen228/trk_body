import os
import h5py
import numpy as np
import pandas as pd
from utils import *

N_FEATS = 500
MAX_FEAT_OFFSET = 150
N_SAMPLES = 300
MAX_XY_OFFSET = 40
MAX_Z_OFFSET = 0.5


def get_feat_offset(n_feats=N_FEATS, max_feat_offset=MAX_FEAT_OFFSET):
    np.random.seed(0)
    feats = np.random.randint(-max_feat_offset, max_feat_offset + 1, (4, n_feats)) # feats.shape = (4, N_FEATS)
    return feats

def feature_generation(img, sample_coord, torso_z, feats, mode='train'):
    coord = sample_coord[:2][::-1] # reverse x, y
    if (coord[0] < 0 or coord[0] > H-1 or coord[1] < 0 or coord[1] > W-1) and mode == 'train':
        return np.zeros((1,N_FEATS))
    if mode == 'test':
        coord[0] = np.clip(coord[0], 0, H-1) 
        coord[1] = np.clip(coord[1], 0, W-1) 
    coord = np.rint(coord).astype(int) 

    z = torso_z if (coord[-1] > 4.5) or (coord[-1] < 1.0) else coord[-1] # avoid z = NaN

    x1 = np.clip(coord[1] + feats[0] / z, 0, W-1).astype(int)
    x2 = np.clip(coord[1] + feats[2] / z, 0, W-1).astype(int)
    y1 = np.clip(coord[0] + feats[1] / z, 0, H-1).astype(int)
    y2 = np.clip(coord[0] + feats[3] / z, 0, H-1).astype(int)

    feature = img[y1, x1] - img[y2, x2] # feature.shape = (1, N_FEATS)
    return feature


def get_sample_offset(max_xy_offset=MAX_XY_OFFSET, max_z_offset=MAX_Z_OFFSET):
    offset_xy = np.random.randint(-max_xy_offset, max_xy_offset + 1, 2)
    offset_z = np.random.uniform(-max_z_offset, max_z_offset, 1)
    offset = np.concatenate((offset_xy, offset_z))
    return offset


def sample_generation(depth_images, joints, joint_id, n_data, n_samples=N_SAMPLES, max_xy_offset=MAX_XY_OFFSET):
    X = np.zeros((n_data, N_SAMPLES, N_FEATS), dtype=np.float64)
    y = np.zeros((n_data, N_SAMPLES, 3), dtype=np.float64)
    feats = get_feat_offset()
    print(feats[:,:4])

    for image_id in range(n_data): # traverse each depth image
        img = depth_images[image_id]
        coord = joints[image_id][joint_id]  # get x, y, z
        torso_z = joints[image_id][8][-1]   # get z of Torso(8)

        for sample_id in range(n_samples): 
            offset = get_sample_offset()
            unit_offset = 0 if np.linalg.norm(offset) == 0 else (-offset / np.linalg.norm(offset))
            X[image_id, sample_id] = feature_generation(img, coord + offset, torso_z, feats)
            y[image_id, sample_id] = unit_offset

    X_reshape = X.reshape(X.shape[0]*X.shape[1], X.shape[2])
    y_reshape = y.reshape(y.shape[0]*y.shape[1], y.shape[2])
    return X_reshape, y_reshape

def main():
    # ITOP data
    processed_data_path = 'data/processed/ITOP'
    depth_images = np.load(os.path.join(processed_data_path, 'ITOP_side_train_depth_map.npy'))
    joints = np.load(os.path.join(processed_data_path, 'ITOP_side_train_labels.npy'))
    folder_path = '/home/dl-box/MDisk/features_40_05_partial'

    print("Loaded data\'s shape: ", depth_images.shape)
    print("Loaded data\'s shape: ", joints.shape)
    n_data = len(depth_images)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for joint_id in range(N_JOINTS): # traverse each joint
        if joint_id in [0,1,2,3,8,9,10]:
            continue
        logger.debug('------------------- %s -------------------', JOINT_IDX[joint_id])
        X, y = sample_generation(depth_images, joints, joint_id, n_data)
        print(X.shape)
        print(y.shape)
        path = os.path.join(folder_path, '%d_data.h5' % joint_id)
        f = h5py.File(path, 'w')
        f["X"] = X
        f["y"] = y
        f.close()

if __name__ == "__main__":
    main()