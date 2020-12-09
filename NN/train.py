import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from model.model import Linear
from data.itop_data import ITOP

# sys.path.append('/home/yxs/titech/utils')
# from depth_utils import *

N_INPUT = 15 * 3
N_HIDDEN = 1024
N_OUTPUT = 15 * 3

EPOCH = 100
BATCH_SIZE = 64
LR = 0.0003

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(comment='-NN')

view = 'side'
DATA_PATH = '/home/dl-box/yang/3d_HPE_depth/VideoPose3D/data/rtw'

def main():
    # create model
    print(">>> creating model")
    model = Linear(N_INPUT, N_HIDDEN, N_OUTPUT)
    model = model.to(device)
    # if isinstance(model, nn.Linear):
    #     nn.init.kaiming_normal(model.weight)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,30,50,80], gamma=0.3)
    # loss_func = nn.MSELoss(size_average=True).cuda()
    loss_func = nn.SmoothL1Loss(size_average=True).cuda()

    # load dataset
    print(">>> loading data")
    train_loader = DataLoader(
            dataset=ITOP(DATA_PATH, view, 'train'),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True)
    eval_loader = DataLoader(
            dataset=ITOP(DATA_PATH, view, 'test'),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True)
    print(">>> data loaded")

    # start training
    for epoch in range(EPOCH):
        print("===========================")
        print(">>> epoch: {} | lr: {:.5f}".format(epoch+1, optimizer.state_dict()['param_groups'][0]['lr']))
        writer.add_scalar('learning_rate', optimizer.state_dict()['param_groups'][0]['lr'], epoch)

        train_loss = train(train_loader, model, optimizer, loss_func)
        start = time.time()
        eval_loss, mAP_loss = evaluate(eval_loader, model, loss_func)
        end = time.time()
        print(">>> runtime: %f" % (end-start))

        scheduler.step(epoch)
        writer.add_scalars('loss_scalars', {'train_loss' : train_loss,
                                            'eval_loss' : eval_loss}, epoch)
        writer.add_scalars('mAP_scalars', {' 10cm mAP' : mAP_loss}, epoch)
        print('>>> ({:03d}/{:03d}) | eval_loss: {:.4f} | 10cm mAP: {:.4f}'.format(len(eval_loader), len(eval_loader), eval_loss, mAP_loss))

    print(">>> training finished")

    # save net parameters
    torch.save(model.state_dict(), 'model/model_parameters_nn_s_{}_{}.pkl'.format(BATCH_SIZE, EPOCH))

def train(train_loader, model, optimizer, loss_func):
    model.train()
    loss_sum = 0
    for step, (x, y) in enumerate(train_loader):
        b_x = x.to(device)
        b_y = y.to(device)

        output = model(b_x)
        loss = loss_func(output, b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        if (step+1) % 280 == 0:
            print(">>> ({:03d}/{:03d}) | train_loss: {:.4f}".format(step+1, len(train_loader), loss_sum/(step+1)))
    
    return loss_sum / len(train_loader)


def evaluate(eval_loader, model, loss_func):
    model.eval()
    loss_sum = 0
    mAP_sum, joints_sum = 0, 0
    with torch.no_grad():
        for step, (x, y) in enumerate(eval_loader):
            b_x = x.to(device)
            b_y = y.to(device)

            batch_joints = b_y.shape[0] * 15

            output = model(b_x)
            loss = loss_func(output, b_y)
            mAP_cnt = eval_10cm(output.detach().cpu().numpy().reshape(-1,15,3), b_y.detach().cpu().numpy().reshape(-1,15,3)) * batch_joints
            
            loss_sum += loss.item()
            mAP_sum += mAP_cnt
            joints_sum += batch_joints
            # if (step+1) % 35 == 0:
            #     print(">>> ({:03d}/{:03d}) | eval_loss: {:.4f} | 10cm mAP: {:.4f}".format(step+1, len(eval_loader), loss_sum/(step+1), mAP_sum/joints_sum))

    return loss_sum / len(eval_loader), mAP_sum / joints_sum
    

def eval_10cm(a, b):
    dist = np.linalg.norm(a-b, axis=2).flatten()
    cnt_10cm = len([i for i in dist if i <= 0.1])
    return cnt_10cm / len(dist)
                
if __name__ == '__main__':
    main()
