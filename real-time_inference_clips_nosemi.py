import os
import sys
import copy
import torch
import joblib
import numpy as np
# from time import time
import time

sys.path.append('./A2J/src/')
sys.path.append('./RTW/rtw_py/')
sys.path.append('./VideoPose3D/')
from A2J.src.a2j_inference import a2j_detection
from RTW.rtw_py.rtw_inference import rtw_tracking
from RTW.rtw_py.utils import *
from VideoPose3D.cnn_inference import cnn_recovery_nosemi
from VideoPose3D.common.model import *

N_JOINTS = 15

from arguments import parse_args
args = parse_args()
print(args)

clip_list_train = [[0, 3212], 
                   [3264, 17880],
                   [17901, 19560],
                   [19658, 31008],
                   [31015, 31325],
                   [31422, 33421],
                   [33422, 35073],
                   [35074, 37228],
                   [37294, 37590],
                   [37591, 39795]]

clip_list_test = [[0, 2409],
                 [2428, 3467],
                 [3557, 4126],
                 [4126, 10116],
                 [10308, 10501]]

def load_data():
    processed_data_path = 'RTW/rtw_py/data/processed/ITOP'
    depth_images = np.load(os.path.join(processed_data_path, 'ITOP_{}_{}_depth_map_full.npy'.format(args.view, args.mode)))
    joints = np.load(os.path.join(processed_data_path, 'ITOP_{}_{}_labels.npy'.format(args.view, args.mode)))                # (x,y)pixel, (z)m
    joints_3d = np.load(os.path.join(processed_data_path, 'ITOP_{}_{}_labels_3d.npy'.format(args.view, args.mode)))          # (x,y,z)m
    indices = np.load(os.path.join(processed_data_path, 'ITOP_{}_{}_indices.npy'.format(args.view, args.mode)))

    return depth_images, joints, joints_3d, indices

def load_models(filter_widths):
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
    model_pos = TemporalModel(15, 3, 15, filter_widths=filter_widths, causal=True, dropout=args.dropout, channels=args.channels, dense=False)
    if torch.cuda.is_available():
        model_pos = model_pos.cuda()
    chk_filename = '/home/dl-box/yang/3d_HPE_depth/VideoPose3D/checkpoint/{}/epoch_{}.bin'.format(args.cnn, args.epoch)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))
    model_pos.load_state_dict(checkpoint['model_pos'])
    
    return regressors, Ls, model_pos

def main():
    # Load data
    depth_images, joints, joints_3d, indices = load_data()
    n_data = len(depth_images)
    print('depth images shape: {}'.format(depth_images.shape))

    new_indices = []
    tmp_indices = []
    pairs = dict()
    l = 0
    for i, j in enumerate(indices):
        if i - l == j - indices[l]:
            tmp_indices.append(j)
            pairs[j] = i
            if j == indices[-1]:
                new_indices.append(tmp_indices)
        else:
            new_indices.append(tmp_indices)
            tmp_indices = [j]
            pairs[j] = i
            l = i
    # for start, end in clip_list_train:
    #     new_indices.append(np.arange(start,end))

    total_len = 0
    for i in range(len(new_indices)):
        total_len += len(new_indices[i])
        # if 512 in new_indices[i] or 3469 in new_indices[i] or 4130 in new_indices[i] or 10119 in new_indices[i] or 10181 in new_indices[i] or 8665 in new_indices[i]:
        #     print(new_indices[i])
    print(total_len, len(indices))

    # Load models
    filter_widths = [int(x) for x in args.architecture.split(',')]
    regressors, Ls, model_pos = load_models(filter_widths)

    total_preds_3d, total_idx = [], 0
    for clip in new_indices:
        # Initialization
        # print('>>> initialization...')
        init_point = a2j_detection(clip[0])[0]
        draw_preds(depth_images[clip[0]], init_point, './init.png')
        # refine_point = a2j_detection(2428)[0]
        
        # Running
        # print('>>> start running')
        # start_time = time.time()
        preds_2d, preds_3d = np.zeros((len(clip), N_JOINTS, 3)), np.zeros((len(clip), N_JOINTS, 3))
        preds_2d[0], preds_3d[0] = init_point, pixel2world(init_point)
        for image_id in range(1, len(clip)):
            if image_id % 50 == 0:
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '({}/{})'.format(total_idx+image_id, n_data))
            img = depth_images[clip[image_id]]
            # RTW tracking
            start_point = world2pixel(preds_3d[image_id - 1]) # if image_id != 2428 else refine_point
            preds_2d[image_id] = rtw_tracking(regressors, Ls, img, start_point, args.n_steps, args.step_size)
            # Ditaled temporal CNN
            preds_3d[image_id] = cnn_recovery_nosemi(model_pos, preds_2d[:image_id+1])
        # elapsed = (time.time() - start_time) / (n_data - args.start_idx)
        # print('average running time: {}s'.format(elapsed))
        total_preds_3d.append(preds_3d)
        total_idx += len(clip)
    
    output_preds_3d = []
    for clip_idx in range(len(total_preds_3d)):
        for frame_idx in range(len(total_preds_3d[clip_idx])):
            output_preds_3d.append(total_preds_3d[clip_idx][frame_idx])
    output_preds_3d = np.array(output_preds_3d)

    # Save results
    np.save('RTW/rtw_py/output/preds/{}_{}_{}_{}_a2j-track_2020{}_cnn_{}_clips.npy'.format(args.view, args.mode, args.n_steps, args.step_size, \
                                                                                        args.rtw, args.cnn), output_preds_3d)


if __name__ == '__main__':
    main()