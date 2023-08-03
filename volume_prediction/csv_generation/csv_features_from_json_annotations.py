"""
Create a csv file where each row is a grape bunch and each column is a feature.
The features are:
- bunch_id: unique identifier of the grape bunch
- west: boolean value that indicates if the grape bunch is on the west side of the vine
- cut: boolean value that indicates if the grape bunch is cut
- date (optional): date of the image acquisition for the year 2021
- area: area of the grape bunch in the image
- avg_depth (optional): average depth (distance) of the grape bunch in the image
- not_occluded: boolean value that indicates if the grape bunch is occluded or not
"""

import csv
import json
import numpy as np
import cv2
import os
import pycocotools.mask as mask_util
import argparse
import re


def retrieve_date_from_img_id(img_id):

    if img_id <= 26:
        img_date = "2021-07-27"
    elif img_id > 26 and img_id <= 49:
        img_date = "2021-08-23"
    else:
        img_date = "2021-09-06"

    return img_date


def find_imagesInfo_from_id(img_id, images_json):
    filename = ""
    width = 0
    height = 0

    for img in images_json:
        if img["id"] == img_id:
            filename = img["file_name"]
            width = img["width"]
            height = img["height"]

    return filename, width, height


def bin_mask_frPoly(annotation, height, width):
    """
    This function returns a binary mask (numpy array) from the JSON segmentation annotation (list of polygon vertices).
    """

    poly = annotation["segmentation"]                   # it is a list of lists because a segmentation of a single grape bunch it can be composed of multiple polygons
    rle = mask_util.frPyObjects(poly, height, width)    # run lenght encoding of the polygons
    binary_mask = mask_util.decode(rle)                 # binary mask with as many channels as polygon components
    binary_mask_merged = np.bitwise_or.reduce(binary_mask, axis=2)  # bitwise OR between the single components masks

    return binary_mask_merged


def area(annotation, height, width):
    """
    This function returns the area of a segmentation annotation.
    """

    binary_mask = bin_mask_frPoly(annotation, height, width)
    area = np.sum(binary_mask)

    return area


def compute_avgDepth(depth_img, binary_mask):
    """
    This function returns the avg depth of the image wrt the segmentation mask where pixels are not 0.
    """

    depth_img_annot = depth_img * binary_mask

    # TODO: Plot depth img values to see if there are some outliers

    avg_depth = depth_img_annot.sum() / np.count_nonzero(depth_img_annot)

    return avg_depth


def parse_json_annotation(annotation, images_json, depth_folder, include_date):

    img_id = annotation["image_id"]
    # Find image info corresponding to image ID
    rgb_file_name, width, height = find_imagesInfo_from_id(img_id, images_json)

    # Compute area of the annotation
    area = annotation["area"]               # segmentation aera from JSON file annotation
    area_norm = area / (height * width)     # normalize by the total image number of pixels to account for different resolutions (but same camera and optic)

    # Retrieve date from image ID
    if include_date:
        date = retrieve_date_from_img_id(int(re.sub("\D", "", rgb_file_name)))

    # Compute avg depth of the annotation if depth images are available
    if depth_folder:
        # Extract the binary mask corresponding to the polygon annotations with pycocotools
        binary_mask = bin_mask_frPoly(annotation, height, width)
        depth_file_name = rgb_file_name.replace("rgb", "depth")
        depth_img = cv2.imread(os.path.join(depth_folder, depth_file_name), cv2.IMREAD_ANYDEPTH)
        avg_depth = compute_avgDepth(depth_img, binary_mask)
    
    ann_dict = {}
    ann_dict["bunch_id"] = "g" + str(int(annotation["attributes"]["bunch_no"])) + "p" + str(int(annotation["attributes"]["plant_no"]))
    ann_dict["west"] = annotation["attributes"]["west"]
    ann_dict["cut"] = annotation["attributes"]["cut"]
    if include_date:
        ann_dict["date"] = date
    ann_dict["area"] = area_norm
    if depth_folder:
        ann_dict["avg_depth"] = avg_depth
    ann_dict["not_occluded"] = annotation["attributes"]["not_occluded"]

    return ann_dict


def write_and_save_to_csv(ann_dict, csv_file, writer, depth_folder, include_date):

    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        body = [ann_dict["bunch_id"], ann_dict["west"], ann_dict["cut"], ann_dict["area"], ann_dict["not_occluded"]]
        if include_date:
            body += [ann_dict["date"]]
        if depth_folder:
            body += [ann_dict["avg_depth"]]
        writer.writerow(body)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", help="path of the json annotations file", default="/home/user/red_globe_2021_datasets/red_globe_2021_07-27_09-06/annotations/red_globe_2021_07-27_09-06.json")
    parser.add_argument("--depth_folder", help="path of the depth images folder", default=None)
    parser.add_argument("--csv_file", help="path of the output raw csv file that will be created", default="/home/user/red_globe_2021_datasets/volume_regression_csv/redglobe_2021_features.csv")
    parser.add_argument("--include_date", help="whether to include the dates of data collection for the year 2021", default=False)

    args = vars(parser.parse_args())
    json_path = args["json_path"]
    depth_folder = args["depth_folder"]
    csv_file = args["csv_file"]
    include_date = args["include_date"]

    with open(json_path) as f:
        json_data = json.load(f)

    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        header = ["bunch_id", "west", "cut", "area", "not_occluded"]
        if include_date:
            header += ["date"]
        if depth_folder is not None:
            header += ["avg_depth"]
        writer.writerow(header)

    for annotation in json_data["annotations"]:
        if annotation.get("attributes").get("tagged"):
            ann_parsed_dict = parse_json_annotation(annotation, json_data["images"], depth_folder, include_date)
            write_and_save_to_csv(ann_parsed_dict, csv_file, writer, depth_folder, include_date)
