from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.NAME = ""
_C.DATASET.ANNOTATIONS_PATH_TRAIN = ""
_C.DATASET.ANNOTATIONS_PATH_TEST = ""
_C.DATASET.IMAGES_PATH_TRAIN = ""
_C.DATASET.IMAGES_PATH_TEST = ""
_C.DATASET.IMAGE_SIZE = ()
_C.DATASET.CROP_SIZE = ()
_C.DATASET.MASKING = False
_C.DATASET.TARGET_SCALING = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.BATCH_SIZE = 2

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.EPOCHS = 100
_C.SOLVER.BASE_LR = 0.001

def get_base_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()