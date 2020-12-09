import os
import torch
import numpy as np
from torch.utils.data import Dataset
import joblib

def normalize_screen_coordinates(X, w, h): 
    assert X.shape[-1] == 2
    
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X/w*2 - [1, h/w]

class ITOP(Dataset):
    '''
    x_train.shape -> (n x 15 x 3)
    y_train.shape -> (n x 15 x 3)
    x_test.shape -> (m x 15 x 3)
    y_test.shape -> (m x 15 x 3)

    joints.shape -> (45 x 1)
    labels.shape -> (45 x 1)
    '''
    def __init__(self, data_path, view, mode):
        self.data_path = data_path
        self.view = view
        self.mode = mode
        self.x, self.y, self.indices = [], [], []
        
        # loading data
        x_tmp = np.load(os.path.join(data_path, '{}_{}_16_2_a2j-track_20201026.npy'.format(self.view, self.mode)))
        y_tmp = np.load(os.path.join(data_path, '{}_{}_gt-a2j.npy'.format(self.view, self.mode)))
        self.indices = np.load(os.path.join(data_path, 'ITOP_{}_{}_indices.npy'.format(self.view, self.mode)))

        # normalization
        x_tmp[..., :2] = normalize_screen_coordinates(x_tmp[..., :2], w=320, h=240)

        # clip valid data
        for i in self.indices:
            # if i - 1 in self.indices:
            #     self.x.append(x_tmp[i-1])
            # else:
            #     self.x.append(x_tmp[i])
            self.x.append(x_tmp[i])
            self.y.append(y_tmp[i])
        self.x, self.y = np.array(self.x), np.array(self.y)
        
        print('data shape: ', self.x.shape, self.y.shape)


    def __getitem__(self, index):
        joints = torch.from_numpy(self.x[index].flatten()).float()
        labels = torch.from_numpy(self.y[index].flatten()).float()

        return joints, labels


    def __len__(self):
        return len(self.y)
