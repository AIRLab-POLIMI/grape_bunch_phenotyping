import logging
import functools
import os
from configs.dataset_cfg import get_dataset_cfg_defaults
import torch
import detectron2
from detectron2.engine import default_argument_parser, default_setup, launch, default_writers
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.data import transforms as T
from detectron2.data import DatasetMapper   # the default mapper
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data import get_detection_dataset_dicts
from detectron2.data.samplers import InferenceSampler
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
import detectron2.utils.comm as comm
from torch.nn.parallel import DistributedDataParallel
from detectron2.solver import build_optimizer, build_lr_scheduler
from detectron2.utils.events import EventStorage
from collections import OrderedDict
from detectron2.evaluation import inference_on_dataset, print_csv_format, DatasetEvaluators, COCOEvaluator
from fvcore.common.param_scheduler import ExponentialParamScheduler
from detectron2.config import CfgNode
try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from detectron2.solver import WarmupParamScheduler, LRMultiplier

# Logging metadata with Neptune
import neptune.new as neptune

run = neptune.init_run(project='AIRLab/grape-bunch-phenotyping',
                       mode='async',        # use 'debug' to turn off logging, 'async' otherwise
                       name='freeze_at_1',
                       tags=[])


logger = logging.getLogger("detectron2")


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
                                        allow_cached_coco=False,    # our dataset is small, so we do not need caching
                                        use_fast_impl=True))        # use a fast but unofficial implementation to compute AP.
                                                                    # compute results with the official API for use in papers

    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def build_test_loss_loader(cfg):
    dataset = get_detection_dataset_dicts(
            cfg.DATASETS.TEST,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )
    mapper = DatasetMapper(cfg, True)
    test_sampler = InferenceSampler(len(dataset))
    return build_detection_train_loader(dataset=dataset,
                                        mapper=mapper,
                                        sampler=test_sampler,
                                        total_batch_size=1,
                                        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
                                        num_workers=cfg.DATALOADER.NUM_WORKERS)


def do_test(model, test_data_loader, evaluator, test_loss_data_loader=None):
    results = inference_on_dataset(model, test_data_loader, evaluator)
    if comm.is_main_process():
        logger.info("Evaluation results for test dataset in csv format:")
        print_csv_format(results)
        # AP logging with Neptune
        run['metrics/AP_segm_test'].log(results['segm']['AP'])

    if test_loss_data_loader is not None:
        with torch.no_grad():
            batches_no = 0
            loss_sum = 0.0
            for data in test_loss_data_loader:
                loss_dict = model(data)
                losses = sum(loss_dict.values())
                assert torch.isfinite(losses).all(), loss_dict
                
                loss_dict_reduced = {"test_" + k: v.item()
                                    for k, v in comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                loss_sum += losses_reduced 
                batches_no += 1
            
            avg_loss = loss_sum / batches_no

            if comm.is_main_process():
                # Test loss logging with Neptune
                run['metrics/total_test_loss'].log(avg_loss)
    
    return results['segm']['AP']


def get_dataset_dicts(cfg, split_name: str):
    if split_name == 'TRAIN':
        return load_coco_json(cfg.DATASET.ANNOTATIONS_PATH_TRAIN,
                              cfg.DATASET.IMAGES_PATH_TRAIN, cfg.DATASET.NAME+"_TRAIN")
    elif split_name == 'TEST':
        return load_coco_json(cfg.DATASET.ANNOTATIONS_PATH_TEST,
                              cfg.DATASET.IMAGES_PATH_TEST, cfg.DATASET.NAME+"_TEST")
    else:
        return None


def build_exp_lr_scheduler(cfg: CfgNode, optimizer: torch.optim.Optimizer) -> LRScheduler:
    """
    Build an exponential LR scheduler.
    """
    
    sched = ExponentialParamScheduler(
        start_value=cfg.SOLVER.BASE_LR,
        decay=cfg.SOLVER.GAMMA
        )

    sched = WarmupParamScheduler(
        sched,
        cfg.SOLVER.WARMUP_FACTOR,
        min(cfg.SOLVER.WARMUP_ITERS / cfg.SOLVER.MAX_ITER, 1.0),
        cfg.SOLVER.WARMUP_METHOD,
        cfg.SOLVER.RESCALE_INTERVAL,
    )
    return LRMultiplier(optimizer, multiplier=sched, max_iter=cfg.SOLVER.MAX_ITER)


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


def do_train_test(cfg, args):

    # ------ DATA LOADERS ------

    dataset_name = dataset_name = cfg.DATASETS.TEST[0]

    # Define a sequence of augmentations:
    augs_list = [
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        T.RandomApply(T.RandomContrast(0.75, 1.25)),        # default probability of RandomApply is 0.5
        T.RandomApply(T.RandomSaturation(0.75, 1.25)),
        T.RandomApply(T.RandomBrightness(0.75, 1.25))
    ]

    train_data_loader = build_detection_train_loader(cfg,
                                                mapper=DatasetMapper(cfg,
                                                                    is_train=True,
                                                                    augmentations=augs_list,
                                                                    image_format=cfg.INPUT.FORMAT,
                                                                    use_instance_mask=True)
                                                )

    test_data_loader = build_detection_test_loader(cfg, dataset_name)

    test_loss_data_loader = build_test_loss_loader(cfg)

    evaluator = get_evaluator(
        cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
    )

    # ------ MODEL ------

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(model, test_data_loader, evaluator)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    # ------ TRAIN ------

    # We tell the model that we are training.
    # This helps to inform layers which are
    # designed to behave differently during
    # training and evaluation (like Dropout
    # and BatchNorm).
    model.train()

    optimizer = build_optimizer(cfg, model)
    # scheduler = build_lr_scheduler(cfg, optimizer)        # default step scheduler
    scheduler = build_exp_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter 
    )

    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter):
        for data, iteration in zip(train_data_loader, range(start_iter, max_iter)):
            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                if comm.is_main_process():
                    # loss logging with Neptune
                    run['metrics/total_train_loss'].log(losses_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            run['scalars/lr'].log(optimizer.param_groups[0]["lr"])  # log learning rate with Neptune
            scheduler.step()

            # ------ TEST ------

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                do_test(model, test_data_loader, evaluator, test_loss_data_loader)
                comm.synchronize()

            periodic_checkpointer.step(iteration)    

    return do_test(model, test_data_loader, evaluator)


def main(args):
    # ------ CONFIGURATIONS ------

    cfg = setup(args)               # load detectron2 configurations

    # custom configurations
    dataset_cfg = get_dataset_cfg_defaults()
    dataset_cfg.merge_from_file("./configs/dataset_test_cfg.yaml")
    dataset_cfg.freeze()

    # ------ NEPTUNE LOGGING ------

    # Log fixed parameters in Neptune
    # Tuned parameters (with Optuna) are commented out
    PARAMS = {'dataset_train': cfg.DATASETS.TRAIN,
              'dataset_test': cfg.DATASETS.TEST,
              'dataloader_num_workers': cfg.DATALOADER.NUM_WORKERS,
              'freeze_at': cfg.MODEL.BACKBONE.FREEZE_AT,
              'batch_size_train': cfg.SOLVER.IMS_PER_BATCH,
              'max_iter': cfg.SOLVER.MAX_ITER,
              'base_lr': cfg.SOLVER.BASE_LR,
              'momentum': cfg.SOLVER.MOMENTUM,
              'weight_decay': cfg.SOLVER.WEIGHT_DECAY,
              'gamma': cfg.SOLVER.GAMMA,
              'steps': cfg.SOLVER.STEPS,
              'warmup_factor': cfg.SOLVER.WARMUP_FACTOR,
              'warmup_iters': cfg.SOLVER.WARMUP_ITERS,
              'eval_period': cfg.TEST.EVAL_PERIOD,
              'optimizer': 'SGD'}

    # Pass parameters to the Neptune run object.
    run['cfg_parameters'] = PARAMS          # This will create a ‘parameters' directory containing the PARAMS dictionary

    # ------ DATASETS ------

    for split_name in ['TRAIN', 'TEST']:
        DatasetCatalog.register(dataset_cfg.DATASET.NAME+"_"+split_name,
                                lambda: get_dataset_dicts(dataset_cfg, split_name))           # register the dataset
        MetadataCatalog.get(dataset_cfg.DATASET.NAME+"_"+split_name).thing_colors = [(255, 0, 0)]              # add color metadata for bunches

    # ------ TRAIN AND TEST WITH OPTUNA ------

    return do_train_test(cfg, args)


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