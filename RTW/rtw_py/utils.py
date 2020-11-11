import cv2
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

#np.set_printoptions(threshold=np.nan)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

fh = logging.FileHandler('log.txt')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

palette = [(34, 88, 226), (34, 69, 101), (0, 195, 243), (146, 86, 135),
           (38, 61, 43), (241, 202, 161), (50, 0, 190), (128, 178, 194),
           (23, 45, 136), (0, 211, 220), (172, 143, 230), (108, 68, 179),
           (121, 147, 249), (151, 78, 96), (0, 166, 246), (165, 103, 0),
           (86, 136, 0), (130, 132, 132), (0, 182, 141), (0, 132, 243)] # BGR

# JOINT_IDX = {
#     0 :'Torso',
#     1 :'Neck',
#     2 :'Head',
#     3 :'Left Shoulder',
#     4 :'Left Elbow',
#     5 :'Left Hand',
#     6 :'Right Shoulder',
#     7 :'Right Elbow',
#     8 :'Right Hand',
#     9 :'Left Hip',
#     10:'Left Knee',
#     11:'Left Foot',
#     12:'Right Hip',
#     13:'Right Knee',
#     14:'Right Foot',
# }

JOINT_IDX = {
    0 :'Head',
    1 :'Neck',
    2 :'R Shoulder',
    3 :'L Shoulder',
    4 :'R Elbow',
    5 :'L Elbow',
    6 :'R Hand',
    7 :'L Hand',
    8 :'Torso',
    9 :'R Hip',
    10:'L Hip',
    11:'R Knee',
    12:'L Knee',
    13:'R Foot',
    14:'L Foot',
}

# H, W = 424, 512
H, W = 240, 320
N_JOINTS = 15

def pixel2world(pixel):
    # fx, fy, cx, cy = 366.391, 366.391, 255.16, 204.694	
    world = np.empty(pixel.shape)
    world[:, 0] = (pixel[:, 0] - 160) * pixel[:, 2] * 0.0035
    world[:, 1] = (120 - pixel[:, 1]) * pixel[:, 2] * 0.0035
    world[:, 2] = pixel[:, 2]
    return world

def world2pixel(world):
    pixel = np.empty(world.shape)
    pixel[:,0] = 160.0 + world[:,0] / (0.0035 * world[:,2])
    pixel[:,1] = 120.0 - world[:,1] / (0.0035 * world[:,2])
    pixel[:,2] = world[:,2]
    return pixel

def mean_distance(y, preds):
    distances = []
    for i in range(len(preds)):
        distances.append(np.linalg.norm(y[i]-preds[i]))
    return np.array(distances).mean()

def draw_scatter_animation(x, y, N, w, i, path):
    ims = []
    fig = plt.figure()
    for i in range(N):
        im = plt.scatter(x[i], y[i], linewidths=w).findobj()
        ims.append(im)
    ani = ArtistAnimation(fig, ims, interval=i)
    ani.save(path, writer='pillow')

def draw_preds(img, joints, filename):
    img = (img-np.amin(img))*255.0/(np.amax(img)-np.amin(img))
    img = img.astype(np.uint8)
    img = cv2.equalizeHist(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)

    if joints is not None:
        joints_copy = joints.copy()
        for i, joint in enumerate(joints_copy):
            cv2.circle(img, tuple(joint[:2].astype(np.uint16)), 4, palette[i], -1)

    cv2.imwrite(filename, img)