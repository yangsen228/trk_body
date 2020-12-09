import numpy as np
import torch

from data.itop_data import normalize_screen_coordinates

def nn_recovery(model, preds_2d):
    # Padding
    receptive_field = 2
    pad = max(0, receptive_field - len(preds_2d))
    batch_2d = np.pad(preds_2d, ((pad, 0), (0, 0), (0, 0)), 'edge')[-receptive_field:,:,:]
    batch_2d[..., :2] = normalize_screen_coordinates(batch_2d[..., :2], w=320, h=240)

    # Inference
    with torch.no_grad():
        model.eval()
        inputs_2d = torch.from_numpy(np.expand_dims(batch_2d.flatten(), axis=0).astype('float32'))
        if torch.cuda.is_available():
            inputs_2d = inputs_2d.cuda()
        # Predict 3D poses
        predicted_3d_pos = model(inputs_2d)
    
    return predicted_3d_pos.cpu().data.numpy().reshape((15,3))