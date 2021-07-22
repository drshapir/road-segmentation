#!/usr/bin/env python3
"""
Created on 2019-04-17

@author: dillonshapiro

For training the Dilated UNet model.
"""
from pathlib import Path

import numpy as np
from imgaug import augmenters as iaa
from keras.callbacks import (TensorBoard, ModelCheckpoint, EarlyStopping,
                             ReduceLROnPlateau)
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

from .dilated_unet import DilatedUNet
from .utils import Dataset, bce_dice_loss


def load_data(img_dir, test_size=.2):
    """
    Loads and preprocesses the image/mask pairs

    Args:
        img_dir (Path-like): Location of all training image/mask pairs.
        test_size (float): Percentage of training data to keep aside as test data.

    Returns:
        train_dataset (Dataset)
        test_dataset (Dataset)
    """
    img_paths = sorted(list(Path(img_dir).glob('*_sat.jpg')))
    mask_paths = sorted(list(Path(img_dir).glob('*_msk.png')))

    X_train, X_test, y_train, y_test = train_test_split(img_paths, mask_paths,
                                                        test_size=test_size,
                                                        random_state=13)
    train_dataset = Dataset(X_train, y_train)
    test_dataset = Dataset(X_test, y_test)
    return train_dataset, test_dataset


def train(train_data, test_data, model_params):
    """
    Trains the Dilated UNet model, given a dictionary of model parameters.

    Args:
        train_data (Dataset): Object containing the training data.
        test_data (Dataset): Object containing the test data.
        model_params (dict): Contains all hyperparameters for initializing the Dilated UNet model. See the DilatedUNet
            class definition for info on parameter inputs.
    """
    model = DilatedUNet(**model_params).compile_model()
    model.summary()

    board = TensorBoard()
    stop = EarlyStopping(monitor='val_dice_coef', mode='max', patience=10)
    save_chkpt = ModelCheckpoint(
        'dilated-unet_best.hdf5', save_best_only=True,
        mode='max', verbose=1, save_weights_only=True, monitor='val_dice_coef'
    )
    reduce_lr = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.2,
                                  patience=5, verbose=1, epsilon=1e-4,
                                  mode='max')
    aug = iaa.OneOf([iaa.Flipud(0.5), iaa.Fliplr(0.5)])

    batch_n = 12
    train_gener = train_data.image_batch_generator(batchsize=batch_n,
                                                   augmenter=aug)
    test_gener = test_data.image_batch_generator(batchsize=batch_n)

    model.fit_generator(
        train_gener, steps_per_epoch=np.ceil(len(train_data) / batch_n),
        epochs=100, verbose=1, validation_data=test_gener,
        validation_steps=np.ceil(len(test_data) / batch_n),
        callbacks=[save_chkpt, board, stop, reduce_lr],
        use_multiprocessing=True
    )


if __name__ == '__main__':
    # replace train_dir with filepath to training data to run as script
    train_dir = 'train/'
    train_data, test_data = load_data(train_dir, test_size=.2)

    # parameters from our best model (can also replace as needed)
    model_params = {'n_blocks': 3, 'filters': 32, 'input_shape': (512, 512, 3),
                    'lr': 2e-4, 'loss': bce_dice_loss, 'optimizer': RMSprop}

    train(train_data, test_data, model_params)
