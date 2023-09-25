import torch
from detectron2.evaluation import inference_on_dataset, DatasetEvaluators, COCOEvaluator
import os
import detectron2
from detectron2.data import build_detection_test_loader
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.config import get_cfg
from configs.cstm_cfg import get_cstm_cfg_defaults
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import load_coco_json


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    # evaluator_list.append(
    #     SemSegEvaluator(
    #         dataset_name,
    #         distributed=True,
    #         output_dir=output_folder,
    #     )
    # )
    evaluator_list.append(COCOEvaluator(dataset_name,
                                        output_dir=output_folder,
                                        allow_cached_coco=False,        # our dataset is small, so we do not need caching
                                        use_fast_impl=False))           # use a fast but unofficial implementation to compute AP.
                                                                        # compute results with the official API for use in papers

    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {}".format(dataset_name)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_predict(cfg, args):

    dataset_name = cfg.DATASETS.TEST[0]

    test_data_loader = build_detection_test_loader(cfg, dataset_name)

    evaluator = get_evaluator(
        cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
    )

    model = build_model(cfg)

    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )

    return inference_on_dataset(model, test_data_loader, evaluator)


def get_dataset_dicts(cfg, split_name: str):
    if split_name == 'TRAIN':
        return load_coco_json(cfg.DATASET.ANNOTATIONS_PATH_TRAIN,
                              cfg.DATASET.IMAGES_PATH_TRAIN, cfg.DATASET.NAME+"_TRAIN")
    elif split_name == 'TEST':
        return load_coco_json(cfg.DATASET.ANNOTATIONS_PATH_TEST,
                              cfg.DATASET.IMAGES_PATH_TEST, cfg.DATASET.NAME+"_TEST")
    else:
        return None


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()                         # obtain detectron2's default config
    cfg.merge_from_file(args.config_file)   # load values from a file
    cfg.merge_from_list(args.opts)          # load values from a list of str

    # cfg.freeze()                          # I cannot freeze cfg for hyperparam tuning
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


def main(args):
    # ------ CONFIGURATIONS ------

    cfg = setup(args)               # load detectron2 configurations

    # custom configurations
    cstm_cfg = get_cstm_cfg_defaults()
    cstm_cfg.merge_from_file("./configs/cstm_test_cfg.yaml")
    cstm_cfg.freeze()

    DatasetCatalog.register(cstm_cfg.DATASET.NAME+"_TEST",
                            lambda: get_dataset_dicts(cstm_cfg, "TEST"))
    MetadataCatalog.get(cstm_cfg.DATASET.NAME+"_TEST").thing_colors = [(255, 0, 0)]

    return do_predict(cfg, args)


if __name__ == "__main__":
    TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
    CUDA_VERSION = torch.__version__.split("+")[-1]
    print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
    print("detectron2:", detectron2.__version__)

    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
