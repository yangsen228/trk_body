import os
import cv2
import numpy as np

N_JOINTS = 15

def read_labels(label_path):
    joints = []
    with open(label_path) as f:
        content = [line.rstrip() for line in f.readlines()]
        for line in content:
            fields = line.split(',')
            assert len(fields) == 4*N_JOINTS+2, 'Actual length is: ' + str(len(fields))
            joint_coords = []
            for joint_id in range(N_JOINTS):
                x = float(fields[2+4*joint_id])
                y = float(fields[3+4*joint_id])
                z = float(fields[4+4*joint_id])
                joint_coords.append([x,y,z])
            joints.append(joint_coords)
    return joints

def read_images(image_path):
    fs = cv2.FileStorage(image_path, cv2.FileStorage_READ)
    depth_image = fs.getNode('bufferMat').mat()
    fs.release()
    return depth_image

def background_removal(image_array):
    for idx in range(len(image_array)):
        img = image_array[idx]
        #img[img > 3000] = 10000
        #img[img == 0] = 10000
        #kernel = np.ones((5, 5), np.uint8)
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        img_morph = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        image_array[idx] = img_morph 
    return image_array

def main():
    data_path = 'data/original'
    joints = []
    depth_images = []
    for root, _, files in os.walk(data_path, topdown=False):
        files.sort(key=lambda x:int(x[:-4]))
        for fname in files:
            if fname.endswith('.txt'):
                joints = read_labels(os.path.join(root, fname))
            elif fname.endswith('.xml'):
                idx = int(fname[:-4])
                if idx % 100 == 0:
                    print('processing: %d' % idx)
                depth_images.append(read_images(os.path.join(root, fname)))
    
    joints = np.array(joints)
    depth_images = np.array(depth_images)
    depth_images = background_removal(depth_images)

    np.save('data/processed/kinect_depth_images_071.npy', depth_images)
    np.save('data/processed/kinect_joints_071.npy', joints)

    # Check the preprocessed data
    print(depth_images.shape)
    print(joints.shape)
    print(joints[0])
    img = depth_images[255]
    print(np.amax(img))
    print(np.amin(img))
    img = (img-np.amin(img))*255.0/(np.amax(img)-np.amin(img))
    img = img.astype(np.uint8)
    cv2.imshow('test', img)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()