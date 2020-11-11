import torch
import numpy as np

def cnn_recovery(model_pos, preds_2d):
    # Padding
    receptive_field = model_pos.receptive_field()
    pad = max(0, receptive_field - len(preds_2d))
    batch_2d = np.expand_dims(np.pad(preds_2d, ((pad, 0), (0, 0), (0, 0)), 'edge'), axis=0)[:,-receptive_field:,:,:]

    # Inference
    with torch.no_grad():
        model_pos.eval()
        # Evaluate on test set
        inputs_2d = torch.from_numpy(batch_2d.astype('float32'))  # shape = (1, receptive_filed, 15, 3)
        if torch.cuda.is_available():
            inputs_2d = inputs_2d.cuda()
        # Predict 3D poses
        predicted_3d_pos = model_pos(inputs_2d)

    return predicted_3d_pos.cpu().data.numpy()[0]