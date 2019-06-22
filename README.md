# COMP540 Road Segmentation Project

The final task of the COMP540 (Statistical Machine Learning) class was to build a model that could accurately segment roads within a set of Google Earth images. We were given approximately 11,000 images with corresponding hand-drawn masks to act as a training set, along with a validation set containing ~2,000 unlabeled images. Student groups were scored (by the Dice coefficient metric) and ranked according to the accuracy of predictions on this validation set.

This repository serves as a record of our group's work, where we utilized a convolutional neural network architecture (called a Dilated U-Net) to achieve a final Dice coefficient score of 0.73273.

## Dilated U-Net Architecture

Our final model was a blending of two parts:
- A simple U-Net architecture, comprised of successive encoder blocks, corresponding decoder blocks, and a bottleneck in between.
- A series of dilated convolutional layers, which were then summed together and passed into the decoding blocks.

![Diagram of Dilated U-Net](network.png)

### U-Net



### Dilated Convolutions



### Training Process



## Citations

1. Ronneberger, O., Fischer, P., Brox, T.(2015). U-Net: Convolutional Networks for Biomedical ImageSegmentation.  In N. Navab, J. Hornegger, W. M. Wells, & A. F. Frangi (Eds.), Medical Image Computingand Computer-Assisted Intervention – MICCAI 2015 (pp.  234–241).  Cham:  Springer International Pub-lishing
2. Yu, F., and Koltun, V. (2015). Multi-Scale Context Aggregation by Dilated Convolutions.
3. Dilated U-Net github reference: https://github.com/lyakaap/Kaggle-Carvana-3rd-place-solution
