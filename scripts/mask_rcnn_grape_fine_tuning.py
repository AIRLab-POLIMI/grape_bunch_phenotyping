from configs.dataset_cfg import get_dataset_cfg_defaults
import torch, detectron2
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json

def get_coco_dataset_dicts(cfg):
  
  return load_coco_json(cfg.DATASET.ANNOTATIONS_PATH, cfg.DATASET.IMAGES_PATH, cfg.DATASET.NAME)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()                         # obtain detectron2's default config
    cfg.merge_from_file(args.config_file)   # load values from a file
    cfg.merge_from_list(args.opts)          # load values from a list of str

    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg

def main(args):
    cfg = setup(args)                       # load configurations

    # custom configurations
    dataset_cfg = get_dataset_cfg_defaults()
    dataset_cfg.merge_from_file("configs/dataset_cfg.yaml")
    dataset_cfg.freeze()
    
    DatasetCatalog.register(dataset_cfg.DATASET.NAME, lambda: get_coco_dataset_dicts(dataset_cfg))      # register the dataset
    MetadataCatalog.get(dataset_cfg.DATASET.NAME).thing_colors = [(255,0,0)]                            # add color metadata for bunches


    # model = build_model(cfg)
    # logger.info("Model:\n{}".format(model))
    # if args.eval_only:
    #     DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    #         cfg.MODEL.WEIGHTS, resume=args.resume
    #     )
    #     return do_test(cfg, model)

    # distributed = comm.get_world_size() > 1
    # if distributed:
    #     model = DistributedDataParallel(
    #         model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
    #     )

    # do_train(cfg, model, resume=args.resume)
    # return do_test(cfg, model)

    return 0

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