import os
import re
import cv2
from pycocotools import mask as maskUtils
import numpy as np
import json
import torch
import torchvision.transforms as T
from torchvision.io import read_image
from torch.utils.data import Dataset


class VinePlantsDataset(Dataset):
    def __init__(self, annotations_file, target, img_dir, img_size,
                 depth_dir=None, color_transform=None,
                 color_depth_transform=None, target_transform=None,
                 target_scaling=True, horizontal_flip=False,
                 not_occluded=False):

        with open(annotations_file) as dictionary_file:
            json_dictionary = json.load(dictionary_file)

        self.json_imgs = json_dictionary['images']
        self.target = target
        self.img_dir = img_dir
        self.fixed_img_size = img_size      # img_size expressed as (height, width)
        self.depth_dir = depth_dir
        self.color_transform = color_transform
        self.color_depth_transform = color_depth_transform
        self.target_transform = target_transform
        self.target_scaling = target_scaling
        self.horizontal_flip = horizontal_flip # TODO: NO NEED TO BE HERE IN THIS CASE, ALSO ADD ROTATION AND TRANSLATION
        self.anns_dict = {}
        self.img_labels = []
        self.min_max_target = None

        # create a dictionary with the annotations for each image
        for ann in json_dictionary['annotations']:
            img_id = ann['image_id']
            if img_id not in self.anns_dict:
                self.anns_dict[img_id] = []
            self.anns_dict[img_id].append(ann)

        # remove from json_imgs the images that do not have at least a valid annotation
        new_json_imgs = []
        for img in self.json_imgs:
            # FILTER OUT IMAGES OF SEPTEMBER BECAUSE OF INVALID DEPTH
            # TODO: REMOVE THIS FILTER AND DO IT DIRECTLY ON THE DATASET
            # img_filename = img['file_name']
            # img_number = int(re.findall(r'\d+', img_filename)[0])
            # filter out images with img_id greater than
            # if img_number > 49:
            #     continue
            # FILTER OUT IMAGES OF SEPTEMBER BECAUSE OF INVALID DEPTH
            if any([not ann['attributes']['cut'] and
                    ann['attributes'][self.target] > 0.0
                    for ann in self.anns_dict[img['id']]]):
                new_json_imgs.append(img)
        self.json_imgs = new_json_imgs

        # loop through all the images to create the target
        for img in self.json_imgs:
            label = 0
            for ann in self.anns_dict[img['id']]:
                label += ann['attributes'][self.target]
            self.img_labels.append(label)

        # if target_scaling is a boolean, then compute the min and max target values
        if isinstance(self.target_scaling, bool):
            self.min_max_target = (min(self.img_labels), max(self.img_labels))
        # if target_scaling is a tuple, then use the values in the tuple as min and max target values
        elif isinstance(self.target_scaling, tuple):
            self.min_max_target = self.target_scaling

    def __len__(self):
        return len(self.json_imgs)

    def __getitem__(self, idx):
        # get the json img
        img = self.json_imgs[idx]
        img_size = (img['height'], img['width'])
        img_filename = img['file_name']

        # load the image
        img_path = os.path.join(self.img_dir, img['file_name'])
        image = read_image(img_path)
        # convert into float32 and scale into [0,1]
        image = T.ConvertImageDtype(torch.float32)(image)

        # put a black pixels mask on all invalid annotations
        for ann in self.anns_dict[img['id']]:
            if ann['attributes']['cut'] or ann['attributes'][self.target] <= 0.0:
                segmentation_mask = ann['segmentation']
                # create a binary mask where the pixels inside the segmentation
                # mask are 1 and the pixels outside the mask are 0
                rles = maskUtils.frPyObjects(segmentation_mask, img_size[0], img_size[1])
                rle = maskUtils.merge(rles)
                mask = maskUtils.decode(rle)
                np.logical_not(mask, out=mask)  # element-wise negation to delete invalid bunch pixels
                mask = np.array(mask, dtype=np.float32)
                # convert the mask to a Torch tensor
                mask = torch.from_numpy(mask)
                # apply the mask to the image
                image = image * mask

        if self.color_transform:
            image = self.color_transform(image)

        # load depth images
        if self.depth_dir:
            # extract the number from img_filename with regex (r'\d+')
            depth_filename = 'depth' + str(re.findall(r'\d+', img_filename)[0]) + '.png'
            depth_path = os.path.join(self.depth_dir, depth_filename)
            # read image using cv2 and passthorugh as encoding
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            # scale image in the range [0, 1.0]
            depth = depth / 6000.0
            # convert to Torch tensor
            depth = torch.from_numpy(depth)
            # add a dimension to the depth image
            depth = depth.unsqueeze(0)
            depth = T.ConvertImageDtype(torch.float32)(depth)
            # concatenate depth image to RGB image
            image = torch.cat((image, depth), 0)

        # resize the image if needed
        if self.fixed_img_size[0] != img_size[0] or self.fixed_img_size[1] != img_size[1]:
            image = T.Resize(size=self.fixed_img_size, antialias=True)(image)

        if self.color_depth_transform:
            image = self.color_depth_transform(image)

        # get the target
        label = self.img_labels[idx]

        # scale the target if needed
        if self.target_scaling:
            min = self.min_max_target[0]
            max = self.min_max_target[1]
            label = (label - min) / (max - min)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label
