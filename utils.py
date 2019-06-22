#!/usr/bin/env python3
"""
Created on 2019-04-17

@author: dillonshapiro

utility functions/classes
"""
import os

import keras.backend as K
import numpy as np
import pandas as pd
from keras.losses import binary_crossentropy
from skimage.io import imread


def dice_coef(y_true, y_pred):
    smooth = 1e-9
    y_true_f = K.flatten(y_true)
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * (K.sum(intersection) + smooth) / (K.sum(y_true_f) +
                                                   K.sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - ((2*intersection + 1)/(K.sum(y_true_f)+K.sum(y_pred_f)+1))


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def create_submission(csv_name, predictions, image_ids):
    """
    csv_name -> string for csv ("XXXXXXX.csv")
    predictions -> numpyarray of size (num_examples, height, width)
                In this case (num_examples, 512, 512)
    image_ids -> numpyarray or list of size (num_examples,)

    predictions[i] should be the prediction of road for image_id[i]
    """
    sub = pd.DataFrame()
    sub['ImageId'] = image_ids
    encodings = []
    num_images = len(image_ids)
    for i in range(num_images):
        if (i + 1) % (num_images // 10) == 0:
            print(i, num_images)
        encodings.append(rle_encoding(predictions[i]))
    sub['EncodedPixels'] = encodings
    sub.to_csv(csv_name, index=False)


#
def rle_encoding(x):
    """
    Run-length encoding stolen
    from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python

    x = numpyarray of size (height, width) representing the mask of an image
    if x[i,j] == 0:
        image[i,j] is not a road pixel
    if x[i,j] != 0:
        image[i,j] is a road pixel
    """
    dots = np.where(x.T.flatten() != 0)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def get_img_id(img_path):
    img_basename = os.path.basename(img_path)
    img_id = os.path.splitext(img_basename)[0][:-len('_sat')]
    return img_id


class Dataset(object):
    def __init__(self, img_paths, mask_paths=None):
        self.img_paths = sorted(img_paths)
        if mask_paths:
            self.mask_paths = sorted(mask_paths)
        else:
            self.mask_paths = None

    def __len__(self):
        return len(self.img_paths)

    def image_gen(self):
        # Iterate over all the image paths
        for i, img_path in enumerate(self.img_paths):
            # Load the image and scale it to 0-1 range
            img = imread(img_path) / 255
            if self.mask_paths:
                mask = imread(self.mask_paths[i])[:, :, 0] / 255
                mask = np.expand_dims(mask, axis=-1)
                yield img, mask
            else:
                yield img, None

    @staticmethod
    def _augment_pairs(augmenter, batch_img, batch_mask):
        aug_det = augmenter.to_deterministic()
        aug_imgs = aug_det.augment_images(batch_img)
        aug_masks = aug_det.augment_images(batch_mask)
        return aug_imgs, aug_masks

    def image_batch_generator(self, batchsize=32, augmenter=None):
        while True:
            ig = self.image_gen()
            batch_img, batch_mask = [], []
            for img, mask in ig:
                # Add the image and mask to the batch
                batch_img.append(img)
                batch_mask.append(mask)
                # If we've reached our batchsize, yield the batch and reset
                if len(batch_img) == batchsize:
                    batch_img = np.stack(batch_img, axis=0)
                    batch_mask = np.stack(batch_mask, axis=0)
                    if self.mask_paths:
                        if augmenter:
                            batch_img, batch_mask = self._augment_pairs(
                                augmenter, batch_img, batch_mask)
                        yield batch_img, batch_mask
                    else:
                        yield batch_img
                    batch_img, batch_mask = [], []
            # If we have an nonempty batch left, yield it out and reset
            if len(batch_img) != 0:
                batch_img = np.stack(batch_img, axis=0)
                batch_mask = np.stack(batch_mask, axis=0)
                if self.mask_paths:
                    if augmenter:
                        batch_img, batch_mask = self._augment_pairs(augmenter,
                                                                    batch_img,
                                                                    batch_mask)
                    yield batch_img, batch_mask
                else:
                    yield batch_img
                batch_img, batch_mask = [], []
