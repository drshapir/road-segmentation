#!/usr/bin/env python3
"""
Created on 2019-04-17

@author: dillonshapiro

Uses best dilated UNet model to generate prediction masks for validation dataset
and convert them to RLE format.
"""
from pathlib import Path

import numpy as np
from keras.optimizers import RMSprop

from dilated_unet import DilatedUNet
from utils import create_submission, get_img_id, Dataset, bce_dice_loss


def predict(val_data, model):
    batch_n = 10
    val_gen = val_data.image_batch_generator(batch_n)
    y_preds = model.predict_generator(val_gen, steps=np.ceil(len(val_data)/batch_n))
    y_preds = y_preds.reshape(len(val_data), 512, 512)
    y_preds[y_preds > 0.5] = 1
    y_preds[y_preds <= 0.5] = 0
    return y_preds


if __name__ == '__main__':
    # load best model weights
    weights_fn = 'dilated-unet_best.hdf5'
    model_params = {'n_blocks': 3, 'filters': 32, 'input_shape': (512, 512, 3),
                    'lr': 2e-4, 'loss': bce_dice_loss, 'optimizer': RMSprop}
    unet = DilatedUNet(**model_params).compile_model()
    unet.load_weights(weights_fn)

    val_dir = 'val/'
    val_paths = sorted(list(Path(val_dir).glob('*_sat.jpg')))
    val_data = Dataset(val_paths)

    predictions = predict(val_data, unet)
    img_ids = [get_img_id(path) for path in val_data.img_paths]
    create_submission('dilated-unet_predictions.csv', predictions, img_ids)
