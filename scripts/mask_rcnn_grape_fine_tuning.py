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

# Hyperparameters optimization with Optuna
import neptune.new.integrations.optuna as optuna_utils
import optuna

run = neptune.init_run(project='AIRLab/grape-bunch-phenotyping',
                       mode='async',        # use 'debug' to turn off logging
                       name='test',
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


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
            # AP50 logging with Neptune
            # run['metrics/AP50_segm_test'].log(results_i['segm']['AP50'])
    
    return results[dataset_name]['segm']['AP50']


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


def objective(trial, cfg, args):

    # Define hyperparameters to tune with Optuna
    cfg.SOLVER.BASE_LR = trial.suggest_float("base_lr", 1e-6, 0.5, log=True)  # If log is true, the value is sampled from the range in the log domain
    cfg.MODEL.BACKBONE.FREEZE_AT = trial.suggest_categorical("freeze_at", [0, 1, 2, 3, 4, 5])
    cfg.SOLVER.WARMUP_ITERS = trial.suggest_int("warmup_iters", 30, 90, step=10)
    cfg.SOLVER.GAMMA = trial.suggest_float("gamma", 0.1, 0.9) 

    # ------ MODEL ------

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

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
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter  # by default is every 5000 iterations
    )

    # writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    # Define a sequence of augmentations:
    augs_list = [
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        T.RandomApply(T.RandomContrast(0.75, 1.25)),        # default probability of RandomApply is 0.5
        T.RandomApply(T.RandomSaturation(0.75, 1.25)),
        T.RandomApply(T.RandomBrightness(0.75, 1.25))
    ]

    data_loader = build_detection_train_loader(cfg,
                                                mapper=DatasetMapper(cfg,
                                                                    is_train=True,
                                                                    augmentations=augs_list,
                                                                    image_format=cfg.INPUT.FORMAT,
                                                                    use_instance_mask=True)
                                                )

    epoch = 0
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter):
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                # loss logging with Neptune
                # run['metrics/total_train_loss'].log(losses_reduced)
                pass

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            # run['scalars/lr'].log(optimizer.param_groups[0]["lr"])  # log learning rate with Neptune
            scheduler.step()

            # ------ TEST ------

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                performance = do_test(cfg, model)
                trial.report(performance, epoch)    # log perfomance after each epoch to enable pruning
                # Handle pruning based on the intermediate value
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                epoch += 1
                comm.synchronize()

            # if iteration - start_iter > 5 and (
            #     (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            # ):
            #     for writer in writers:
            #         writer.write()
            periodic_checkpointer.step(iteration)    

    return do_test(cfg, model)


def main(args):
    # ------ CONFIGURATIONS ------

    cfg = setup(args)               # load detectron2 configurations

    # custom configurations
    dataset_cfg = get_dataset_cfg_defaults()
    dataset_cfg.merge_from_file("./configs/dataset_train_cfg.yaml")
    dataset_cfg.freeze()

    # ------ NEPTUNE LOGGING ------

    # Log fixed parameters in Neptune
    # Tuned parameters (with Optuna) are commented out
    PARAMS = {'dataset_train': cfg.DATASETS.TRAIN,
              'dataset_test': cfg.DATASETS.TEST,
              'dataloader_num_workers': cfg.DATALOADER.NUM_WORKERS,
              # 'freeze_at': cfg.MODEL.BACKBONE.FREEZE_AT,
              'batch_size_train': cfg.SOLVER.IMS_PER_BATCH,
              'max_iter': cfg.SOLVER.MAX_ITER,
              # 'base_lr': cfg.SOLVER.BASE_LR,
              'momentum': cfg.SOLVER.MOMENTUM,
              'weight_decay': cfg.SOLVER.WEIGHT_DECAY,
              'gamma': cfg.SOLVER.GAMMA,
              'steps': cfg.SOLVER.STEPS,
              'warmup_factor': cfg.SOLVER.WARMUP_FACTOR,
              # 'warmup_iters': cfg.SOLVER.WARMUP_ITERS,
              'eval_period': cfg.TEST.EVAL_PERIOD,
              'optimizer': 'SGD'}

    # Pass parameters to the Neptune run object.
    run['parameters'] = PARAMS          # This will create a ‘parameters' directory containing the PARAMS dictionary

    # ------ DATASETS ------

    for split_name in ['TRAIN', 'TEST']:
        DatasetCatalog.register(dataset_cfg.DATASET.NAME+"_"+split_name,
                                lambda: get_dataset_dicts(dataset_cfg, split_name))           # register the dataset
        MetadataCatalog.get(dataset_cfg.DATASET.NAME+"_"+split_name).thing_colors = [(255, 0, 0)]              # add color metadata for bunches

    # ------ TRAIN AND TEST WITH OPTUNA ------

    # Create a NeptuneCallback for Optuna
    neptune_callback = optuna_utils.NeptuneCallback(run)

    # Pass NeptuneCallback to Optuna Study .optimize()
    study = optuna.create_study(direction="maximize", study_name="lr_freeze_warmiters", storage='sqlite:///optuna-db/lr_freeze_warmiters.db')
    study.optimize(functools.partial(objective, cfg=cfg, args=args),  # I use functools to create a new function with the additional arguments
                   n_trials=1,            # The number of trials for each process
                   timeout=None,            # Stop study after the given number of seconds
                   n_jobs=1,               # The number of parallel jobs. If -1, the number is set to CPU count
                   callbacks=[neptune_callback])


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