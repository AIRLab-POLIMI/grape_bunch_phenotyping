from configs.dataset_cfg import get_dataset_cfg_defaults
import torch
import detectron2
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.data import transforms as T
from detectron2.data import DatasetMapper   # the default mapper
from detectron2.data import build_detection_train_loader


def do_train(cfg, model, resume=False):
    # model.train()
    # optimizer = build_optimizer(cfg, model)
    # scheduler = build_lr_scheduler(cfg, optimizer)

    # checkpointer = DetectionCheckpointer(
    #     model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    # )
    # start_iter = (
    #     checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    # )
    # max_iter = cfg.SOLVER.MAX_ITER

    # periodic_checkpointer = PeriodicCheckpointer(
    #     checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    # )

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
                                                                    image_format='RGB',        # TODO: take from config file
                                                                    use_instance_mask=True)
                                                )

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    # logger.info("Starting training from iteration {}".format(start_iter))
    # with EventStorage(start_iter) as storage:
    #     for data, iteration in zip(data_loader, range(start_iter, max_iter)):
    #         storage.iter = iteration

    #         loss_dict = model(data)
    #         losses = sum(loss_dict.values())
    #         assert torch.isfinite(losses).all(), loss_dict

    #         loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
    #         losses_reduced = sum(loss for loss in loss_dict_reduced.values())
    #         if comm.is_main_process():
    #             storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

    #         optimizer.zero_grad()
    #         losses.backward()
    #         optimizer.step()
    #         storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
    #         scheduler.step()

    #         if (
    #             cfg.TEST.EVAL_PERIOD > 0
    #             and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
    #             and iteration != max_iter - 1
    #         ):
    #             do_test(cfg, model)
    #             # Compared to "train_net.py", the test results are not dumped to EventStorage
    #             comm.synchronize()

    #         if iteration - start_iter > 5 and (
    #             (iteration + 1) % 20 == 0 or iteration == max_iter - 1
    #         ):
    #             for writer in writers:
    #                 writer.write()
    #         periodic_checkpointer.step(iteration)


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
    
    for split_name in ['TRAIN', 'TEST']:
        DatasetCatalog.register(dataset_cfg.DATASET.NAME+"_"+split_name,
                                lambda: get_dataset_dicts(dataset_cfg, split_name))           # register the dataset
        MetadataCatalog.get(dataset_cfg.DATASET.NAME+"_"+split_name).thing_colors = [(255, 0, 0)]              # add color metadata for bunches


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