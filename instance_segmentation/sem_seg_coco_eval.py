"""
This script is for evaluating semantic segmentation
predictions given in COCO json file comparing them
with the ground truth in COCO json file.
!!! The script can currently handle only single class
annotations !!!
"""
import argparse
import random
import json
import cv2
import numpy as np
from pycocotools import mask as maskUtils


def load_coco_annotations(annotations_file):
    with open(annotations_file, 'r') as f:
        coco_annotations = json.load(f)
    return coco_annotations


def combine_instance_masks(images_dict, annotations):
    combined_masks = {}  # Dictionary to store combined masks for each image
    for annotation in annotations:
        image_id = annotation['image_id']
        image = images_dict[image_id]
        h, w = image['height'], image['width']
        segm = annotation['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, h, w)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = segm
        mask = maskUtils.decode(rle)
        if image_id not in combined_masks:
            combined_masks[image_id] = mask  # TODO: assign a list of masks for multi-class support
        else:
            combined_masks[image_id] = np.logical_or(combined_masks[image_id], mask)
    return combined_masks


def compute_metrics(gt_masks, pred_masks):
    precision_list = []
    recall_list = []
    f1_score_list = []

    for image_id, gt_mask in gt_masks.items():
        pred_mask = pred_masks[image_id]

        intersection = np.logical_and(gt_mask, pred_mask)
        # union = np.logical_or(gt_mask, pred_mask)

        precision = np.sum(intersection) / (np.sum(pred_mask) + 1e-6)   # Avoid division by zero
        recall = np.sum(intersection) / (np.sum(gt_mask) + 1e-6)        # Avoid division by zero

        if precision + recall == 0:
            f1_score = 0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_score_list.append(f1_score)

    average_precision = np.mean(precision_list)
    average_recall = np.mean(recall_list)
    average_f1_score = np.mean(f1_score_list)

    return average_precision, average_recall, average_f1_score


def get_masks_frm_coco(ground_truth_file, inference_file):
    # Load Coco-style annotations from the JSON files
    ground_truth_annotations = load_coco_annotations(ground_truth_file)
    inference_annotations = load_coco_annotations(inference_file)

    images = ground_truth_annotations['images']
    images_dict = {}
    for image in images:
        images_dict[image['id']] = image

    # Combine instance segmentation masks for each image
    gt_masks = combine_instance_masks(images_dict, ground_truth_annotations['annotations'])
    pred_masks = combine_instance_masks(images_dict, inference_annotations)

    return gt_masks, pred_masks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute semantic segmentation metrics with COCO annotations.")
    parser.add_argument("ground_truth_file", help="Path to ground truth COCO json annotations", type=str)
    parser.add_argument("inference_file", help="Path to inference COCO json annotations", type=str)
    parser.add_argument("--debug", help="Activate debug mode", type=bool, default=False)
    args = parser.parse_args()

    gt_masks, pred_masks = get_masks_frm_coco(args.ground_truth_file, args.inference_file)

    if args.debug:
        smpl_gt_masks = random.sample(gt_masks.items(), 10)
        for smpl in smpl_gt_masks:
            image_id, gt_mask = smpl
            cv2.imwrite(f'gt_mask_{image_id}.png', gt_mask * 255)
            cv2.imwrite(f'pred_mask_{image_id}.png', pred_masks[image_id] * 255)

    # Compute evaluation metrics
    average_precision, average_recall, average_f1_score = compute_metrics(gt_masks, pred_masks)

    print(f'Average Precision: {average_precision}')
    print(f'Average Recall: {average_recall}')
    print(f'Average F1-Score: {average_f1_score}')
