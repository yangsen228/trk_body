import os
import sys
import copy
import torch
import joblib
import numpy as np
from time import time

sys.path.append('./A2J/src/')
sys.path.append('./RTW/rtw_py/')
from A2J.src.a2j_inference import a2j_detection
from RTW.rtw_py.rtw_inference import rtw_tracking
from RTW.rtw_py.utils import *
from VideoPose3D.cnn_inference import cnn_recovery
from VideoPose3D.common.model import *

N_JOINTS = 15

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='real-time inference')
    # General arguments
    parser.add_argument('-v', '--view', type=str, default='side', help='which view of ITOP dataset')
    parser.add_argument('-m', '--mode', type=str, default='test', help='train or test')
    parser.add_argument('-i', '--start_idx', type=int, default=32, help='start from xth frame')
    # Model arguments
    parser.add_argument('--rtw', type=str, default='1026', help='which rtw model to use')
    parser.add_argument('--cnn', type=str, default='20201026', help='which cnn model to use')
    # Experimental arguments
    parser.add_argument('-n', '--n-steps', type=int, default=96, help='num of steps in rtw')
    parser.add_argument('-s', '--step-size', type=int, default=2, help='step size in rtw')
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--channels', type=int, default=1024)
    args = parser.parse_args()

    return args

args = parse_args()
print(args)

def load_data():
    processed_data_path = 'RTW/rtw_py/data/processed/ITOP'
    depth_images = np.load(os.path.join(processed_data_path, 'ITOP_{}_{}_depth_map_full.npy'.format(args.view, args.mode)))
    joints = np.load(os.path.join(processed_data_path, 'ITOP_{}_{}_labels.npy'.format(args.view, args.mode)))                # (x,y)pixel, (z)m
    joints_3d = np.load(os.path.join(processed_data_path, 'ITOP_{}_{}_labels_3d.npy'.format(args.view, args.mode)))          # (x,y,z)m

    return depth_images, joints, joints_3d

def load_models():
    # RTW
    rtw_path = 'RTW/rtw_py/models/ITOP/regression_tree_3d_{}'.format(args.rtw)
    print('Loading rtw models', rtw_path)
    regressors, Ls = {}, {}
    for joint_id in range(N_JOINTS):
        regressor_path = os.path.join(rtw_path, 'regressor_%d.pkl' % joint_id)
        regressors[joint_id] = joblib.load(regressor_path)
        L_path = os.path.join(rtw_path, 'L_%d.pkl' % joint_id)
        Ls[joint_id] = joblib.load(L_path)

    # Dilated CNN
    model_pos = TemporalModel(15, 3, 15, filter_widths=[3,3], causal=True, dropout=args.dropout, channels=args.channels, dense=False)
    if torch.cuda.is_available():
        model_pos = model_pos.cuda()
    chk_filename = '/home/dl-box/yang/3d_HPE_depth/VideoPose3D/checkpoint/{}/epoch_80.bin'.format(args.cnn)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))
    model_pos.load_state_dict(checkpoint['model_pos'])
    
    return regressors, Ls, model_pos

def main():
    # Load data
    depth_images, joints, joints_3d = load_data()
    n_data = len(depth_images)
    print('depth images shape: {}'.format(depth_images.shape))

    # Load models
    regressors, Ls, model_pos = load_models()

    # Initialization
    print('>>> initialization...')
    init_point = a2j_detection(args.start_idx)[0]
    draw_preds(depth_images[args.start_idx], init_point, './init.png')
    refine_point = a2j_detection(2428)[0]
    
    # Running
    print('>>> start running')
    start_time = time()
    preds_2d, preds_3d = np.zeros((n_data, N_JOINTS, 3)), np.zeros((n_data, N_JOINTS, 3))
    preds_2d[args.start_idx], preds_3d[args.start_idx] = init_point, pixel2world(init_point)
    for image_id in range(args.start_idx + 1, n_data):
        if image_id % 1000 == 0:
            print(image_id)
        img = depth_images[image_id]
        # RTW tracking
        start_point = world2pixel(preds_3d[image_id - 1]) if image_id != 2428 else refine_point
        preds_2d[image_id] = rtw_tracking(regressors, Ls, img, start_point, args.n_steps, args.step_size)
        # Ditaled temporal CNN
        preds_3d[image_id] = cnn_recovery(model_pos, preds_2d[args.start_idx:image_id+1])
    elapsed = (time() - start_time) / (n_data - args.start_idx)
    print('average running time: {}s'.format(elapsed))

    # Save results
    np.save('RTW/rtw_py/output/preds/{}_{}_{}_{}_a2j-track_2020{}_cnn_rf{}.npy'.format(args.view, args.mode, args.n_steps, args.step_size, args.rtw, model_pos.receptive_field()), preds_3d)


if __name__ == '__main__':
    main()