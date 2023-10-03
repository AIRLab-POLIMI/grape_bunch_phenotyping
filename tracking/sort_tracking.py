import os
from sort import Sort
import time
import numpy as np
import argparse


def track(mot_detections_path, output_dir, max_age=1, min_hits=3, iou_threshold=0.3):

    total_time = 0.0
    frame = 0
    mot_tracker = Sort(max_age, min_hits, iou_threshold)
    seq_dets = np.loadtxt(mot_detections_path, delimiter=',')

    with open(os.path.join(output_dir, 'output_det.txt'), 'w') as out_file:
        for frame in range(int(seq_dets[:, 0].max())):
            frame += 1  # detection and frame numbers begin at 1
            dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
            dets[:, 2:4] += dets[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]

            start_time = time.time()
            trackers = mot_tracker.update(dets)
            cycle_time = time.time() - start_time
            total_time += cycle_time

            for d in trackers:
                print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4], d[0], d[1], d[2]-d[0], d[3]-d[1]), file=out_file)

    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, frame, frame / total_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mot_detections_path", help="path of the mot detections txt file", type=str)
    parser.add_argument("output_dir", help="path of the output directory", type=str)
    parser.add_argument("--max_age", help="The time that can pass without the id assignment.", type=int, default=15)
    parser.add_argument("--min_hits", help="The minimum value of hits in a track such that it gets displayed in the outputs.", type=int, default=15)
    parser.add_argument("--iou_threshold", help="iou threshold", type=float, default=0.3)
    args = vars(parser.parse_args())

    mot_detections_path = args["mot_detections_path"]
    output_dir = args["output_dir"]
    max_age = int(args["max_age"])
    min_hits = int(args["min_hits"])
    iou_threshold = float(args["iou_threshold"])

    track(mot_detections_path, output_dir, max_age, min_hits, iou_threshold)
