import os
import copy
import joblib
import multiprocessing
import numpy as np
from time import time

from feature_engineering import get_feat_offset, feature_generation
from utils import *

view = 'side'
mode = 'train'
N_STEPS = 16
STEP_SIZE = 2
model = '1026'
start_idx = 0

skeleton = [(8,8),(10,8),(12,10),(14,12),(9,8),(11,9),(13,11),(1,8),(0,1),(2,1),(4,2),(6,4),(3,1),(5,3),(7,5)]

DEPTH_IMAGES = None

def random_walk(regressor, L, img, start_point, torso_z, feats, n_steps=N_STEPS, step_size=STEP_SIZE):
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

# Detection
def detection(depth_images, joints, regressors, Ls, n_data):
    preds = np.zeros((n_data, N_JOINTS, 3))
    feats = get_feat_offset()
    for image_id in range(n_data):
        if image_id % 1000 == 0:
            logger.debug('Testing ({}/{})'.format(image_id, n_data))
        for (child, parent) in skeleton:
            start_point = joints[image_id][8] if child == 8 else preds[image_id][parent]
            preds[image_id][child] = random_walk(regressors[child], Ls[child], depth_images[image_id], start_point, joints[image_id][8][-1], feats)
    return preds

# Tracking
def tracking(indices, depth_images, joints, joints_a2j_2d, regressors, Ls, n_data):
    preds = np.zeros((n_data, N_JOINTS, 3))
    feats = get_feat_offset()
    for joint_id in range(N_JOINTS):
        logger.debug('Testing %s(%d)', JOINT_IDX[joint_id], joint_id)
        for image_id in range(n_data):
            if image_id < start_idx:
                continue
            if image_id % 6000 == 0:
                print(image_id)
            start_point = joints_a2j_2d[image_id][joint_id] if (image_id == 32 or image_id == 2428 or image_id == 3505 or image_id == 3552) else preds[image_id-1][joint_id]
            preds[image_id][joint_id] = random_walk(regressors[joint_id], Ls[joint_id], depth_images[image_id], start_point, joints_a2j_2d[image_id][8][-1], feats)
    return preds

# Parallel tracking
def subprocess(joint_id, joints_a2j_2d, regressor, L, n_data, feats):
    subpreds = np.zeros((n_data, 3))
    logger.debug('Testing %s(%d)', JOINT_IDX[joint_id], joint_id)
    for image_id in range(n_data):
        if image_id < start_idx:
            continue
        if image_id % 6000 == 0:
            print(image_id)
        start_point = joints_a2j_2d[image_id][joint_id] # if (image_id == 32 or image_id == 2428 or image_id == 3505 or image_id == 3552) else subpreds[image_id-1]
        subpreds[image_id] = random_walk(regressor, L, DEPTH_IMAGES[image_id], start_point, joints_a2j_2d[image_id][8][-1], feats)
    return joint_id, subpreds

def parallel_tracking(indices, joints, joints_a2j_2d, regressors, Ls, n_data):
    preds = np.zeros((n_data, N_JOINTS, 3))
    feats = get_feat_offset()
    pool = multiprocessing.Pool(8)
    results = []
    for i in range(N_JOINTS):
        results.append(pool.apply_async(subprocess, (i, joints_a2j_2d, regressors[i], Ls[i], n_data, feats)))
    pool.close()
    pool.join()
    for res in results:
        joint_id, subpreds = res.get()[0], res.get()[1]
        preds[:,joint_id,:] = subpreds
    return preds


def main():
    global DEPTH_IMAGES
    # Load data
    processed_data_path = 'data/processed/ITOP'
    logger.debug('Loading data from directory %s', processed_data_path)
    indices = np.load(os.path.join(processed_data_path, 'ITOP_{}_{}_indices.npy'.format(view, mode)))
    DEPTH_IMAGES = np.load(os.path.join(processed_data_path, 'ITOP_{}_{}_depth_map_full.npy'.format(view, mode)))
    joints = np.load(os.path.join(processed_data_path, 'ITOP_{}_{}_labels.npy'.format(view, mode)))
    joints_a2j_3d = np.load(os.path.join(processed_data_path, 'ITOP_{}_{}_labels_a2j.npy'.format(view, mode)))
    # Convert world to pixel coordinates
    n_data = len(DEPTH_IMAGES)
    joints_a2j_2d = []
    for image_id in range(n_data):
        joints_a2j_2d.append(world2pixel(joints_a2j_3d[image_id]))
    joints_a2j_2d = np.array(joints_a2j_2d)
    print(DEPTH_IMAGES.shape, joints_a2j_2d.shape)
    
    # Load model
    model_path = 'models/ITOP/regression_tree_3d_{}'.format(model)
    logger.debug('Loading model from directory %s', model_path)
    regressors, Ls = {}, {}
    for joint_id in range(N_JOINTS):
        regressor_path = os.path.join(model_path, 'regressor_%d.pkl' % joint_id)
        regressors[joint_id] = joblib.load(regressor_path)
        L_path = os.path.join(model_path, 'L_%d.pkl' % joint_id)
        Ls[joint_id] = joblib.load(L_path)
    
    # Test
    start = time()
    # preds = detection(depth_images, joints, regressors, Ls, n_data)
    # preds = tracking(indices, depth_images, joints, joints_a2j_2d, regressors, Ls, n_data)
    preds = parallel_tracking(indices, joints, joints_a2j_2d, regressors, Ls, n_data)
    end = time()
    print('running time: {}'.format((end - start) / (n_data - start_idx)))
    np.save('output/preds/{}_{}_{}_{}_a2j-track_2020{}.npy'.format(view, mode, N_STEPS, STEP_SIZE, model), preds)


if __name__ == "__main__":
    main()