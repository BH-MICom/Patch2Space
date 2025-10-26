import torch
import argparse
import torch.nn as nn

from models import *
from core.config import config
from core.function import inference_brats
from dataset.dataset import get_testset
from utils.utils import determine_device, update_config, create_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='configuration file', required=False, type=str, default='./configs/unet.yaml')
    parser.add_argument('--weights', help='path for pretrained weights', required=False, type=str, default='experiments/model_best.pth')
    parser.add_argument('--mriweights', help='path for pretrained weights', type=str, default='experiments_mri/model_best.pth')
    parser.add_argument('--ctweights', help='path for pretrained weights', type=str, default='experiments_ct/model_best.pth')
    parser.add_argument('--fold', help='which fold to validate', required=False, type=int, default=0)
    args = parser.parse_args()
    return args


def main(args):
    update_config(config, args.cfg)

    net = unet.UNet  # use your network architecture here --> <file_name>.<class_name>
    mri_class = mri_classification.UNet
    ct_class = ct_classification.UNet
    if config.TRAIN.PARALLEL:   # only cuda is supported
        devices = config.TRAIN.DEVICES
        model = net(config)
        model = nn.DataParallel(model, devices).cuda(devices[0])
    else:   # support cuda, mps and ... cpu (really?)
        device = determine_device()
        model = net(config).to(device)
        mri_class = mri_class(config).to(device)
        mri_class.load_state_dict(torch.load(args.mriweights, weights_only=True))
        ct_class = ct_class(config).to(device)
        ct_class.load_state_dict(torch.load(args.ctweights, weights_only=True))
    # load pretrained weights
    model.load_state_dict(torch.load(args.weights))
    # validation dataset
    testset = get_testset(args.fold)

    logger = create_logger(config.LOG_DIR, 'test.log')
    inference_brats(model, mri_class, ct_class, testset, logger, config)


if __name__ == '__main__':
    args = parse_args()
    main(args)
