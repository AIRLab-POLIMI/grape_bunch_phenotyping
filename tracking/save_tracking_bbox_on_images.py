"""
This script reads images and tracking detections in MOT
format and write new images with overlayed bounding boxes
and tracking IDs.
"""
import cv2
import os
import argparse
import random


def get_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def process_images(image_dir, image_ext, detections_file, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load detections from your MOT text file into a dictionary
    detections = {}  # Use a dictionary to store detections by frame reference
    with open(detections_file, "r") as file:
        for line in file:
            frame_no, object_id, bb_left, bb_top, bb_width, bb_height, _, _, _, _ = map(float, line.strip().split(','))
            frame_no = int(frame_no)
            object_id = int(object_id)
            bbox = [int(bb_left), int(bb_top), int(bb_left + bb_width), int(bb_top + bb_height)]

            # Store the detection information in the dictionary
            if frame_no not in detections:
                detections[frame_no] = []
            detections[frame_no].append({'bbox': bbox, 'tracking_id': object_id})

    # Create a dictionary to store random colors for each tracking ID
    id_colors = {}

    # Loop through the images in the directory
    for frame_no in range(1, len(detections) + 1):
        image_path = os.path.join(image_dir, f"image{frame_no}.{image_ext}")  # Assuming image filenames are like "image1.ext", "image2.ext", etc.
        image = cv2.imread(image_path)

        # Check if there are detections for this frame
        if frame_no in detections:
            frame_detections = detections[frame_no]

            # Draw bounding boxes and tracking IDs
            for detection in frame_detections:
                bbox = detection['bbox']
                tracking_id = detection['tracking_id']

                # Generate a random color for this tracking ID (and store it for consistency)
                if tracking_id not in id_colors:
                    id_colors[tracking_id] = get_random_color()
                color = id_colors[tracking_id]

                # Draw bounding box
                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

                # Write tracking ID
                cv2.putText(image, str(tracking_id), ((bbox[0]+bbox[2])//2-10, (bbox[1]+bbox[3])//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Save the image with bounding boxes and tracking IDs in the output directory
        output_path = os.path.join(output_dir, f"track_image{frame_no}.{image_ext}")
        cv2.imwrite(output_path, image)

    print(f"Saved {len(detections)} images with detections in '{output_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save images with MOT detections")
    parser.add_argument("image_dir", help="Directory containing images", type=str)
    parser.add_argument("image_ext", help="Extension of the images", type=str)
    parser.add_argument("detections_file", help="Path to the MOT format detections file", type=str)
    parser.add_argument("output_dir", help="Output directory for processed images", type=str)
    args = parser.parse_args()

    process_images(args.image_dir, args.image_ext, args.detections_file, args.output_dir)
