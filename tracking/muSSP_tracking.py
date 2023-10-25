import numpy as np
import os
import mot3d
import mot3d.weight_functions as wf
import argparse


def muSSP_first_pass_tracking(detections):
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
        - trajectories: list of lists of Detection2D. Each list of detections is a single track.

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
                                                                      sigma_jump=1, sigma_distance=20,
                                                                      max_distance=60,
                                                                      use_color_histogram=False, use_bbox=False)

    weight_confidence = lambda d: wf.weight_confidence_detections_2d(d, mul=1, bias=0)

    g = mot3d.build_graph(mot3d_dets, weight_source_sink=10,
                          max_jump=4, verbose=True,
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


def muSSP_second_pass_tracking(trajectories):

    tracklets = []
    for traj in trajectories:
        tracklets += mot3d.split_trajectory_modulo(traj, length=3)

    tracklets = mot3d.remove_short_trajectories(tracklets, th_length=1)

    print("Building graph on tracklets...")

    detections_tracklets = [mot3d.DetectionTracklet2D(tracklet) for tracklet in tracklets]

    weight_distance_t = lambda t1, t2: wf.weight_distance_tracklets_2d(t1, t2, max_distance=None,
                                                                       sigma_motion=50,
                                                                       cutoff_motion=0.1,
                                                                       use_color_histogram=False, debug=False)

    weight_confidence_t = lambda t: wf.weight_confidence_tracklets_2d(t, mul=1, bias=0)

    g = mot3d.build_graph(detections_tracklets, weight_source_sink=0.5,
                          max_jump=20, verbose=True,
                          weight_confidence=weight_confidence_t,
                          weight_distance=weight_distance_t)
    if g is None:
        raise RuntimeError("There is not a single path between sink and source nodes!")

    print("-"*30)
    print("Solver second pass on tracklets...")
    print("-"*30)
    muSSP_trajectories = mot3d.solve_graph(g, verbose=True, method='muSSP')

    return muSSP_trajectories


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='muSSP tracking')
    parser.add_argument("seq_path", help="Path to detections in MOT format.", type=str)
    parser.add_argument("output_dir", help="Path of the output directory.", type=str)
    parser.add_argument("--only_first_pass", help="Whether to only perform a single pass.", type=bool, default=False)
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
        trajectories = muSSP_first_pass_tracking(detections)
        if not args.only_first_pass:
            trajectories = muSSP_second_pass_tracking(trajectories)
        out_detections = []
        track_id = 0
        for track in trajectories:
            track_id += 1
            for detection_2d in track:
                if args.only_first_pass:
                    # a track is a single trajectory, so each detection in a track
                    # must have the same track_id
                    bbox = detection_2d.bbox
                    out_detections.append([detection_2d.index, track_id, bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1], 1, -1, -1, -1])
                else:
                    bbox = detection_2d.head.bbox
                    width = bbox[2]-bbox[0]
                    height = bbox[3]-bbox[1]
                    for position, index in zip(detection_2d.head.positions, detection_2d.head.indexes):
                        out_detections.append([index, track_id, position[0]-width/2, position[1]-height/2, width, height, 1, -1, -1, -1])

        out_detections.sort(key=lambda x: x[0])
        for det in out_detections:
            print('%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%d,%d,%d' % (*det,), file=out_file)
