import os
import h5py
import numpy as np

view = 'side'
mode = 'test'

images = h5py.File("/home/dl-box/yang/datasets/ITOP/ITOP_{}_{}_depth_map.h5".format(view,mode), 'r')
labels = h5py.File("/home/dl-box/yang/datasets/ITOP/ITOP_{}_{}_labels.h5".format(view,mode), 'r')

save_path = 'data/processed/ITOP'

def main():
    # Get valid indices
    is_valid = labels["is_valid"]
    valid_list = [i for i in range(len(is_valid)) if is_valid[i] == 1]
    print("number of valid indices = %d\n" % len(valid_list))
    np.save(os.path.join(save_path, 'ITOP_{}_{}_indices.npy'.format(view, mode)), valid_list)

    # Get 3d labels (x,y in pixel; z in meter)
    x_y = np.array([labels["image_coordinates"][i] for i in valid_list])
    z = np.array([labels["real_world_coordinates"][i] for i in valid_list])[:,:,-1:]
    x_y_z = np.concatenate((x_y, z), axis=2)
    # x_y_z = np.array([labels["real_world_coordinates"][i] for i in valid_list])
    print('x_y_z.shape =', x_y_z.shape)
    print(x_y_z[0][0])
    np.save(os.path.join(save_path, 'ITOP_{}_{}_labels_3d.npy'.format(view, mode)), x_y_z)

    # Get depth image
    # depth_image = np.array([images["data"][i] for i in valid_list])
    # depth_image = depth_image * 1000
    # print('depth_image.shape =', depth_image.shape)
    # np.save(os.path.join(save_path, 'ITOP_{}_{}_depth_map_full.npy'.format(view, mode)), depth_image)

if __name__ == "__main__":
    main()