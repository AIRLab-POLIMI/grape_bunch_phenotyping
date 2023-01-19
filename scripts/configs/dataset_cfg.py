from yacs.config import CfgNode as CN

_C = CN()

_C.DATASET = CN()
_C.DATASET.NAME = ""
_C.DATASET.ANNOTATIONS_PATH_TRAIN = ""
_C.DATASET.ANNOTATIONS_PATH_TEST = ""
_C.DATASET.IMAGES_PATH_TRAIN = ""
_C.DATASET.IMAGES_PATH_TEST = ""


def get_dataset_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()