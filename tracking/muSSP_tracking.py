import numpy as np
import os
import mot3d
import mot3d.weight_functions as wf
import argparse


def muSSP_tracking(detections, ):
    """
    Functions to build the tracking graph and solve it
    with the muSSP solver.

    Args:
        - detections (list): list of lists of tuples. Each element of the list
                            is a list related to a single frame. Each frame-list
                            is a list of tuples, where each tuple is a position
                            of a detection in the form (x,y), where x and y are
                            the 2D pixel coordinates of the bbox center
                            containing the detected object.

    Returns:
        - trajectories: ?

    """

    start = 0
    stop = len(detections)

    mot3d_dets = []
    for index in range(start, stop):
        for bbox in detections[index]:
            center_x = int((bbox[0]+bbox[2]) / 2)
            center_y = int((bbox[1]+bbox[3]) / 2)
            mot3d_dets.append(mot3d.Detection2D(index+1, (center_x, center_y), confidence=1, bbox=bbox))   # sum 1 to index to follow MOT convention

    print("Building the graph...")

    weight_distance = lambda d1, d2: wf.weight_distance_detections_2d(d1, d2,
                                                                      sigma_jump=1, sigma_distance=10,
                                                                      max_distance=15,
                                                                      use_color_histogram=False, use_bbox=False)

    weight_confidence = lambda d: wf.weight_confidence_detections_2d(d, mul=1, bias=0)

    g = mot3d.build_graph(mot3d_dets, weight_source_sink=20,
                          max_jump=10, verbose=True,
                          weight_confidence=weight_confidence,
                          weight_distance=weight_distance)
    if g is None:
        raise RuntimeError("There is not a single path between sink and source nodes!")

    print("-"*30)
    print("Solving the graph with muSSP...")
    print("-"*30)
    muSSP_trajectories = mot3d.solve_graph(g, verbose=True, method='muSSP')            

    # print("-"*30)
    # print("Solving the graph with ILP...")
    # print("-"*30)
    # ILP_trajectories = mot3d.solve_graph(g, verbose=True, method='ILP')

    return muSSP_trajectories


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='muSSP tracking')
    parser.add_argument("seq_path", help="Path to detections in MOT format.", type=str)
    parser.add_argument("output_dir", help="Path of the output directory.", type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # all train
    args = parse_args()

    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading detections...")
    seq_dets = np.loadtxt(args.seq_path, delimiter=',')     # detections are expected to be in MOT format
                                                            # with frame index starting at 1
    detections = []
    for i, det in enumerate(seq_dets):
        # MOT format: frame_id, track_id, bb_left, bb_top, bb_width, bb_height, confidence, x, y, z
        # MOT bbox convention is (bb_left, bb_top, bb_width, bb_height)
        # muSSP bbox convention is: (xmin, ymin, xmax, ymax)
        xmin = det[2]
        ymin = det[3]
        xmax = det[2] + det[4]
        ymax = det[3] + det[5]
        if (int(det[0])-1) < len(detections):
            detections[int(det[0])-1].append((xmin, ymin, xmax, ymax))
        else:
            detections.append([(xmin, ymin, xmax, ymax)])

    with open(os.path.join(output_dir, 'muSSP_output_det.txt'), 'w') as out_file:
        trajectories = muSSP_tracking(detections)

        for track in trajectories:
            for detection_2d in track:
                # a track is a single trajectory, so each detection in a track
                # must have the same track_id

                # TODO: fill a list of tracks and then order based on frame id ...
                frame_id = detection_2d.index
                bbox = detection_2d.bbox
                print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame_id, id, bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]), file=out_file)
