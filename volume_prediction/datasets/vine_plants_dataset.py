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

        self.json_imgs = json_dictionary['images']
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
        self.anns_dict = {}

        for ann in json_dictionary['annotations']:
            if self.anns_dict[ann['image_id']]:
                self.anns_dict[ann['image_id']].append(ann)
            else:
                self.anns_dict[ann['image_id']] = [ann]

    def __len__(self):
        return len(self.json_imgs)
    
    def __getitem__(self, idx):
        # get the json img
        img = self.json_imgs[idx]
        img_size = (img['height'], img['width'])

        # load the image
        img_path = os.path.join(self.img_dir, img['file_name'])
        image = read_image(img_path)

        # create the target (sum of the volumes of the bunches)
        label = 0
        for ann in self.anns_dict[img['id']]:
            if ann['attributes']['is_cut'] or not (ann['attributes']['volume'] > 0.0 and ann['attributes']['weight'] > 0.0):
                # cut out the grape bunch from the image
                # and replace it with black pixels
                segmentation_mask = ann['segmentation']
                rles = maskUtils.frPyObjects(segmentation_mask, img_size[0], img_size[1])
                rle = maskUtils.merge(rles)
                mask = maskUtils.decode(rle)
                mask = np.array(mask, dtype=np.float32)
                # mask = ! mask negation of the mask
                # convert the mask to a Torch tensor
                mask = torch.from_numpy(mask)
                # apply the mask to the image
                image = image * mask
            else:
                label += ann['attributes']['volume']





