import os
import h5py
import joblib
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.cluster import MiniBatchKMeans
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils import *

K = 20

def load_split_data(feat_path):
    logger.debug('Loading features from directory %s', feat_path)
    # Load data
    f = h5py.File(feat_path, 'r')
    data_X = f["X"]
    data_y = f["y"]
    print('original shape: ', data_X.shape, data_y.shape)
    valid_X, valid_y = [], []
    for i in range(len(data_X)):
        if not (data_X[i] == np.zeros(500)).all():
            valid_X.append(data_X[i])
            valid_y.append(data_y[i])
    valid_X, valid_y = np.array(valid_X), np.array(valid_y)
    print(valid_X.shape, valid_y.shape)
    return valid_X, valid_y

def train(X_train, y_train, max_depth, min_samples_leaf):
    logger.debug('Start training...')
    model = DecisionTreeRegressor(min_samples_leaf=min_samples_leaf)
    #model = RandomForestRegressor(n_estimators=60, max_depth=max_depth, min_samples_split=25, min_samples_leaf=min_samples_leaf, max_features=125, n_jobs=8, random_state=0)
    print("X_train.shape: ", X_train.shape)
    print("y_train.shape: ", y_train.shape)
    model.fit(X_train, y_train)

    return model

def cluster(model, X_train, y_train):
    L = {}
    indices = model.apply(X_train) # leaf id of each sample
    leaf_ids = np.unique(indices)  # array of unique leaf ids

    logger.debug("Running stochastic K-means...")
    print("leaf_node: ", leaf_ids.shape)
    for idx, leaf_id in enumerate(leaf_ids):
        kmeans = MiniBatchKMeans(n_clusters=K, batch_size=1000)

        cur_indices = np.arange(len(indices))[indices == leaf_id]
        labels = kmeans.fit_predict(y_train[cur_indices])
        weights = np.bincount(labels).astype(float) / labels.shape[0]
        # if weights.shape[0] != 20:
        #     print(labels)
        #     print(np.amax(labels))
        #     print(labels.shape)
        #     print(y_train[indices == leaf_id].shape)

        # Normalize the centers
        centers = kmeans.cluster_centers_    # shape: (20, 2)
        norm = np.linalg.norm(centers, axis=1)[:, np.newaxis]   # shape: (20, 1)
        norm[norm == 0] = 1    # avoid L2 norm = 0
        centers /= norm

        L[leaf_id] = (weights, centers)

        if idx % 3000 == 0:
            print("==> {}".format(idx))

    return L

def main():
    train_path = '/home/dl-box/MDisk/features_40_05_partial'

    for joint_id in range(N_JOINTS):
        if joint_id in [0,1,2,3,4,5,6,7,8,9,10,11]:
            continue
        logger.debug('------------------- Train&Test %s(%d) -------------------', JOINT_IDX[joint_id], joint_id)
        train_feat_path = os.path.join(train_path, '%d_data.h5' % joint_id)
        X_train, y_train = load_split_data(train_feat_path)
        # Train tree-based model
        model = train(X_train, y_train, 20, 400)
        model_path = os.path.join('models/ITOP/regression_tree_3d_1105/regressor_%d.pkl' % joint_id)
        joblib.dump(model, model_path)
        # Evaluation
        y_output = model.predict(X_train)
        error = np.mean(np.linalg.norm((y_train - y_output), axis=1))
        print('training error: {}'.format(error))
        # Train kmeans
        L = cluster(model, X_train, y_train)
        L_path = os.path.join('models/ITOP/regression_tree_3d_1105/L_%d.pkl' % joint_id)
        joblib.dump(L, L_path)


if __name__ == "__main__":
    main()
     