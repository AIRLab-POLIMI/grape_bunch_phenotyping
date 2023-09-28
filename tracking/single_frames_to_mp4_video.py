"""
This script takes as input single frames and creates
an mp4 video out of them.
The frames are supposed to be named with a specific pattern:
string+counter+extension.
The counter is an integer id to order the frames.
The extension is given as parameter.
"""
import cv2
import os
import argparse
import re


def create_video(input_folder, output_file, frame_rate, frame_extension):

    image_filenames = os.listdir(input_folder)

    # Filter and sort the file list based on numeric part of the filename
    pattern = re.compile(r'(\d+)')  # Regular expression to match one or more digits
    frame_files = sorted(
        [filename for filename in image_filenames if filename.endswith(frame_extension)],
        key=lambda x: int(pattern.search(x).group()) if pattern.search(x) else float('inf')
    )

    if not frame_files:
        print(f"No {frame_extension} files found in the input folder.")
        return

    # Load the first image to get frame dimensions
    first_frame = cv2.imread(os.path.join(input_folder, frame_files[0]))
    height, width, _ = first_frame.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for AVI format
    out = cv2.VideoWriter(output_file, fourcc, frame_rate, (width, height))

    # Iterate through frames and write them to the video
    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(input_folder, frame_file))
        out.write(frame)

    # Release the VideoWriter and close the OpenCV windows
    out.release()
    # cv2.destroyAllWindows()  # comment out if you are running the script with GUI enabled

    print(f"Video '{output_file}' created successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an MP4 video from image frames.")
    parser.add_argument("input_folder", type=str, help="Path to the folder containing image frames.")
    parser.add_argument("output_file", type=str, help="Name (complete path) of the output video file.")
    parser.add_argument("--frame_rate", type=int, default=3, help="Frame rate of the output video.")
    parser.add_argument("--frame_extension", type=str, default=".png", help="Extension of the image frames (e.g., .png, .jpg).")

    args = parser.parse_args()

    create_video(args.input_folder, args.output_file, args.frame_rate, args.frame_extension)
