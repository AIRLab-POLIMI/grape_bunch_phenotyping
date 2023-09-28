"""
This script takes as input COCO inference detections
saved in a JSON file and converts them into MOT Challenge
detections format saved into a txt file.
"""
import json
import argparse
import os


def coco_to_mot(coco_json_file, output_dir):

    with open(coco_json_file, 'r') as json_file:
        coco_data = json.load(json_file)

    # Initialize a dictionary to store MOT detections
    mot_detections = {}

    for detection in coco_data:
        # Extract relevant information from the COCO format
        frame_id = detection['image_id']
        bbox = detection['bbox']
        confidence = detection['score']

        # MOT format: frame_id, track_id, bb_left, bb_top, bb_width, bb_height, confidence, x, y, z
        mot_line = f"{frame_id},-1,{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{confidence},-1,-1,-1"

        # Append the MOT line to the corresponding frame's entry in the dictionary
        if frame_id in mot_detections:
            mot_detections[frame_id].append(mot_line)
        else:
            mot_detections[frame_id] = [mot_line]

    mot_detections_path = os.path.join(output_dir, 'det.txt')
    # Write MOT detections to the output text file
    with open(mot_detections_path, 'w') as txt_file:
        for frame_id, detections in mot_detections.items():
            txt_file.write('\n'.join(detections) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("coco_inference", help="path of the COCO JSON file", type=str)
    parser.add_argument("output_dir", help="path of the output directory", type=str)
    args = vars(parser.parse_args())

    coco_inference = args["coco_inference"]
    output_dir = args["output_dir"]

    coco_to_mot(coco_inference, output_dir)
