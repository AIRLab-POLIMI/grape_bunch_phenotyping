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
        

class GrapeBunchesDataset(Dataset):

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
        self.apply_mask = apply_mask        # whether to isolate the single bunch with its mask
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

            # FILTER OUT IMAGES OF SEPTEMBER BECAUSE OF INVALID DEPTH
            # TODO: REMOVE THIS FILTER AND DO IT DIRECTLY ON THE DATASET
            img_number = int(re.findall(r'\d+', img_filename)[0])
            # filter out images with img_id greater than
            if img_number > 49:
                continue
            # FILTER OUT IMAGES OF SEPTEMBER BECAUSE OF INVALID DEPTH

            if ann['attributes']['tagged']:
                # skip occluded bunches if the not_occluded parameter is True
                if not_occluded:
                    if not ann['attributes']['not_occluded']:
                        continue
                # skip bunches with volume/weight value <= 0.0
                if ann['attributes']['volume'] > 0.0 and ann['attributes']['weight'] > 0.0:
                    half_crop_width = math.ceil(crop_size[1]/2)
                    half_crop_height = math.ceil(crop_size[0]/2)
                    # check whether bboxes are distant from borders at least half of corresponding crop size
                    # rescale half_crop_{width,height} for img_size
                    x_scale, y_scale = self.x_y_scale(img_width, img_height)
                    if x_scale != 1.0:
                        half_crop_width /= x_scale
                    if y_scale != 1.0:
                        half_crop_height /= y_scale
                    from_left = ann['bbox'][0] >= half_crop_width
                    from_top = ann['bbox'][1] >= half_crop_height
                    from_right = img_width-(ann['bbox'][0]+ann['bbox'][2]/2) >= half_crop_width
                    from_bottom = img_height-(ann['bbox'][1]+ann['bbox'][3]/2) >= half_crop_height
                    if from_left and from_top and from_right and from_bottom:
                        filtered_ann.append(ann)
        # we only add filtered annotations, that is, grapes which have been
        # tagged, grapes with a volume/weight value > 0.0, and grapes that
        # are distant from all image borders at least half of crop_size.
        self.img_labels = filtered_ann

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # get the annotation for the idx-th image
        ann = self.img_labels[idx]
        label = ann['attributes']['volume']
        # scale the label if required
        if self.target_scaling:
            min = self.target_scaling[0]
            max = self.target_scaling[1]
            label = (label - min) / (max - min)

        img_id = ann['image_id']
        img_filename = []
        img_size = []
        for img in self.img_info:
            if img['id'] == img_id:
                img_filename = img['file_name']
                img_size = [img['height'], img['width']]
                break

        # load RGB images
        img_path = os.path.join(self.img_dir, img_filename)
        image = read_image(img_path)
        bbox = ann['bbox']                  # bbox format is [x,y,width,height]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

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
            if self.depth_transform:
                depth = self.depth_transform(depth)
            # concatenate depth image to RGB image
            image = torch.cat((image, depth), 0)

        # apply segmentation mask if required
        if self.apply_mask:
            segmentation_mask = ann['segmentation']
            # create a binary mask where the pixels inside the segmentation
            # mask are 1 and the pixels outside the mask are 0
            rles = maskUtils.frPyObjects(segmentation_mask, img_size[0], img_size[1])
            rle = maskUtils.merge(rles)
            mask = maskUtils.decode(rle)
            mask = np.array(mask, dtype=np.float32)
            # convert the mask to a Torch tensor
            mask = torch.from_numpy(mask)
            # apply the mask to the image
            image = image * mask

        # resize the image if needed
        if self.fixed_img_size[0] != img_size[0] or self.fixed_img_size[1] != img_size[1]:
            image = T.Resize(size=self.fixed_img_size, antialias=True)(image)
            # Calculate the scaling factor for the bounding box
            x_scale, y_scale = self.x_y_scale(img_size[1], img_size[0])
            # Resize the bounding box
            bbox = [
                bbox[0] * x_scale,
                bbox[1] * y_scale,
                bbox[2] * x_scale,
                bbox[3] * y_scale
                ]
        # crop the image with a fixed custom bbox around current bbox center
        bbox_center = (bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2)        # center coordinates format is (x,y)
        custom_x = round(bbox_center[0] - self.crop_size[1]/2)      
        custom_y = round(bbox_center[1] - self.crop_size[0]/2)     
        assert custom_x >= 0.0
        assert custom_y >= 0.0
        assert custom_x+self.crop_size[1] <= self.fixed_img_size[1]
        assert custom_y+self.crop_size[0] <= self.fixed_img_size[0]

        custom_bbox = (custom_x, custom_y, self.crop_size[1], self.crop_size[0])
        img_crop = crop(image, custom_bbox[1], custom_bbox[0], custom_bbox[3], custom_bbox[2])

        if self.horizontal_flip:
            img_crop = T.RandomHorizontalFlip(p=0.5)(img_crop)

        return img_crop, label

    def x_y_scale(self, img_width, img_height):
        x_scale = self.fixed_img_size[1] / img_width
        y_scale = self.fixed_img_size[0] / img_height

        return x_scale, y_scale
