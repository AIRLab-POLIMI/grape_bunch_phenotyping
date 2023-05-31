import os
import re
import cv2
from pycocotools import mask as maskUtils
import numpy as np
import json
import math
import torch
import torchvision.transforms as T
from torchvision.io import read_image
from torchvision.transforms.functional import crop
from torch.utils.data import Dataset


class VinePlantsDataset(Dataset):
    def __init__(self, annotations_file, img_dir, img_size, crop_size,
                 depth_dir=None, apply_mask=False, transform=None,
                 depth_transform=None, target_transform=None,
                 target_scaling=None, horizontal_flip=False,
                 not_occluded=False):

        with open(annotations_file) as dictionary_file:
            json_dictionary = json.load(dictionary_file)

        self.img_info = json_dictionary['images']
        self.img_dir = img_dir
        self.fixed_img_size = img_size      # img_size expressed as (height, width)
        self.crop_size = crop_size          # crop_size expressed as (height, width)
        self.depth_dir = depth_dir
        self.apply_mask = apply_mask        # whether to isolate the single bunches with their masks
        self.transform = transform
        self.depth_transform = depth_transform
        self.target_transform = target_transform
        self.target_scaling = target_scaling
        self.horizontal_flip = horizontal_flip

        filtered_ann = []
        img_id = 0
        for ann in json_dictionary['annotations']:
            if ann['image_id'] != img_id:
                img_id = ann['image_id']
                img_width, img_height = 0, 0
                for img in json_dictionary['images']:
                    if img['id'] == img_id:
                        img_width, img_height = img['width'], img['height']
                        img_filename = img['file_name']
                        break
