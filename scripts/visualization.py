import fiftyone as fo
import fiftyone.utils.coco as fouc
import argparse


def main(args):

    images_dir_path = args.imgs_dir
    pred_file_path = args.pred_json
    gt_file_path = args.gt_json

    # Create a FiftyOne dataset from the JSON dictionary
    coco_dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=images_dir_path,
        labels_path=gt_file_path,
        include_id=True,
        label_field="ground_truth",
    )

    classes = coco_dataset.default_classes
    # And add model predictions
    fouc.add_coco_labels(
        coco_dataset,
        "predictions",
        pred_file_path,
        classes,
        label_type='segmentations',
        coco_id_field="ground_truth_coco_id",
    )

    # Launch the view in a web interface
    session = fo.launch_app(coco_dataset)

    # waiting for the user to be done, then cleaning up
    input("Press enter when you're done inspecting the data:\n")
    session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("imgs_dir", help="path of the images directory", type=str)
    parser.add_argument("pred_json", help="path of the predictions JSON file in COCO format", type=str)
    parser.add_argument("gt_json", help="path of the ground truth JSON file in COCO format", type=str)
    args = parser.parse_args()
    main(args)
