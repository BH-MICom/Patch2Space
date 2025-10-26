from yacs.config import CfgNode as CN

config = CN()
config.NUM_WORKERS = 5
config.PRINT_FREQ = 10
config.VALIDATION_INTERVAL = 5
config.OUTPUT_DIR = 'experiments'
config.LOG_DIR = 'log'
config.SEED = 3407

config.CUDNN = CN()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

config.DATASET = CN()
config.DATASET.ROOT = 'in-house_MRICT_preprocessed'
config.DATASET.NUM_MODALS = 4
config.DATASET.NUM_MODALS_MR = 3
config.DATASET.NUM_MODALS_CT = 1
config.DATASET.NUM_CLASSES = 2

config.MODEL = CN()
config.MODEL.NAME = 'UNet'
config.MODEL.PRETRAINED = ''
config.MODEL.NUM_DIMS = 3
config.MODEL.EXTRA = CN(new_allowed=True)

config.TRAIN = CN()
config.TRAIN.LR = 1e-2
config.TRAIN.WEIGHT_DECAY = 3e-5
config.TRAIN.BATCH_SIZE = 10
config.TRAIN.PATCH_SIZE = [64, 64, 64]
config.TRAIN.NUM_BATCHES = 250 # 250
config.TRAIN.EPOCH = 201
config.TRAIN.PARALLEL = False
config.TRAIN.DEVICES = [0]

config.INFERENCE = CN()
config.INFERENCE.BATCH_SIZE = 10
config.INFERENCE.PATCH_SIZE = [64, 64, 64]
config.INFERENCE.PATCH_OVERLAP = [16, 16, 16]
