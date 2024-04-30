# Grape Bunch Phenotyping

This repository contains code to perform various analyses on grape bunches using RGB images. It is organized into three main directories:

- **Instance Segmentation**: This directory includes PyTorch and Detectron2 code for training Mask R-CNN models on RGB images to perform instance segmentation of grape bunches. It also includes code for Surgical Fine Tuning of specific blocks of the Mask R-CNN model.

- **Tracking**: Here, you'll find code for Multi-Object Tracking of grape bunches using either the fast and online SORT algorithm or the slower but more accurate graph-based algorithm solved through the muSSP solver.

- **Volume Prediction**: This code is designed to predict grape bunch volumes from RGB-D image data.
