# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np

from common.arguments import parse_args
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno

from common.camera import *
from common.model import *
from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from time import time
from common.utils import deterministic_random

args = parse_args()
print(args)

try:
    # Create checkpoint directory if it does not exist
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

print('Preparing data...')

kps_left, kps_right = [9,11,13,2,4,6], [10,12,14,3,5,7]
joints_left, joints_right = [9,11,13,2,4,6], [10,12,14,3,5,7]

action_filter = None if args.actions == '*' else args.actions.split(',')
if action_filter is not None:
    print('Selected actions:', action_filter)
    
print('Loading 2D detections...')
cameras_valid = None
preds_3d = np.load('data/rtw/side_train_16_2_a2j-track_20201026.npy')
poses_valid_2d = np.array([preds_3d])
joints = np.load('data/rtw/side_train_gt-a2j.npy')
poses_valid = np.array([joints])

print(len(poses_valid), poses_valid[0].shape)
print(len(poses_valid_2d), poses_valid_2d[0].shape)

filter_widths = [int(x) for x in args.architecture.split(',')]
model_pos = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], 15,
                            filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels,
                            dense=args.dense)
print(args.causal, args.dropout, args.channels, args.dense)
receptive_field = model_pos.receptive_field()
print('INFO: Receptive field: {} frames'.format(receptive_field))
pad = (receptive_field - 1) // 2 # Padding on each side
if args.causal:
    print('INFO: Using causal convolutions')
    causal_shift = pad
else:
    causal_shift = 0

model_params = 0
for parameter in model_pos.parameters():
    model_params += parameter.numel()
print('INFO: Trainable parameter count:', model_params)

if torch.cuda.is_available():
    model_pos = model_pos.cuda()
        
chk_filename = os.path.join(args.checkpoint, 'normal', args.resume if args.resume else args.evaluate)
print('Loading checkpoint', chk_filename)
checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
print('This model was trained for {} epochs'.format(checkpoint['epoch']))
model_pos.load_state_dict(checkpoint['model_pos'])

test_generator = UnchunkedGenerator(cameras_valid, poses_valid, poses_valid_2d,
                                    pad=pad, causal_shift=causal_shift, augment=False,
                                    kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
print('INFO: Testing on {} frames'.format(test_generator.num_frames()))

# Inference
losses_3d_valid = []
        
print('** Note: reported losses are averaged over all frames and test-time augmentation is not used here.')
print('** The final evaluation will be carried out after the last training epoch.')

# Pos model only
start_time = time()
N = 0
# End-of-epoch evaluation
with torch.no_grad():
    model_pos.eval()

    epoch_loss_3d_valid = 0
    epoch_loss_traj_valid = 0
    epoch_loss_2d_valid = 0
    N = 0
    
    # Evaluate on test set
    for cam, batch, batch_2d in test_generator.next_epoch():
        inputs_3d = torch.from_numpy(batch.astype('float32'))     # shape = (1, 39795, 15, 3)
        inputs_2d = torch.from_numpy(batch_2d.astype('float32'))  # shape = (1, 39795+13*2 = 39821, 15, 2)
        if torch.cuda.is_available():
            inputs_3d = inputs_3d.cuda()
            inputs_2d = inputs_2d.cuda()
        # inputs_3d[:, :, 0] = 0

        # Predict 3D poses
        predicted_3d_pos = model_pos(inputs_2d)
        loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
        epoch_loss_3d_valid += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_3d_pos.item()
        N += inputs_3d.shape[0]*inputs_3d.shape[1]

    # np.save('../RTW/rtw_py/output/preds/tmp_train.npy', predicted_3d_pos.cpu().numpy())
    losses_3d_valid.append(epoch_loss_3d_valid / N)
    print(losses_3d_valid)

elapsed = (time() - start_time) / poses_valid[0].shape[0]
print(elapsed)