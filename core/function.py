import torch
import numpy as np
import torch.nn as nn
import torchio as tio
import gc
from core.config import config
from medpy.metric.binary import dc, hd95, sensitivity

from utils.utils import AverageMeter, determine_device

def train(model, mri_class, ct_class, train_generator, optimizer, criterion, logger, config, epoch):
    model.train()
    losses = AverageMeter()
    # if scaler is not supported, it switches to default mode, the training can also continue
    scaler = torch.cuda.amp.GradScaler()
    num_iter = config.TRAIN.NUM_BATCHES  # number of batches per epoch
    for i in range(num_iter):
        data_dict = next(train_generator)
        data = data_dict['data']
        label = data_dict['label']  # list of length 3, each with shape (32, 1, x, y); x, y may differ due to downsampling
        if config.TRAIN.PARALLEL:
            devices = config.TRAIN.DEVICES
            data = data.cuda(devices[0])
            label = [l.cuda(devices[0]) for l in label]
        else:
            device = determine_device()
            data = data.to(device)
            patches = get_patches(data, [64, 64, 64])    # torch.Size([8, 4, 64, 64, 64])
            mri = patches[:, :3, :, :, :]
            mri = mri.to(device)
            flair = patches[:, 3:, :, :, :]
            flair = flair.to(device)
            PE_mri, PE_flair = [], []
            # classify each patch
            with torch.no_grad():
                PE_mri = mri_class(mri)  # output shape (batch_size, num_classes)
                PE_flair = ct_class(flair)  # output shape (batch_size, num_classes)
                PE_data = torch.stack((PE_mri, PE_flair), dim=0)
                PE_data = PE_data.unsqueeze(1)
                image_data = ((mri, flair))
            PE_data = PE_data.to(device)
            label = [l.to(device) for l in label]
        # run training
        with torch.cuda.amp.autocast():
            out = model(image_data, PE_data)  # returns a list of segmentation maps at different levels
            for jk in range(len(out)):
                out[jk] = reconstructed_patches(out[jk], out[jk].shape[-1])
            loss = criterion(out, label)
        losses.update(loss.item(), len(out))
        # back-propagation

        del data_dict, data, label, out
        gc.collect()
        torch.cuda.empty_cache()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
        scaler.step(optimizer)
        scaler.update()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                'Loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                    epoch, i, num_iter,
                    loss = losses,
                )
            logger.info(msg)


def inference(model, mri_class, ct_class, dataset, logger, config):
    model.eval()
    
    num_classes = config.DATASET.NUM_CLASSES
    perfs = [AverageMeter() for _ in range(num_classes)]
    nonline = nn.Softmax(dim=1)
    scores = {}
    for case in dataset:
        dims = config.MODEL.NUM_DIMS
        patch_size = config.INFERENCE.PATCH_SIZE
        patch_overlap = config.INFERENCE.PATCH_OVERLAP
        # torchio does not support 2D slice natively, only as pseudo 3D patches
        if dims == 2:
            patch_size = [1] + patch_size
            patch_overlap = [0] + patch_overlap
        # pad if data smaller than patch size
        target_shape = np.max([patch_size, case['data'].shape[1:]], 0)
        transform = tio.CropOrPad(target_shape)
        case = transform(case)
        # sliding window
        sampler = tio.inference.GridSampler(case, patch_size, patch_overlap)
        loader = torch.utils.data.DataLoader(sampler, config.INFERENCE.BATCH_SIZE)
        aggregator = tio.inference.GridAggregator(sampler, 'average')
        label_aggregator = tio.inference.GridAggregator(sampler, 'average')

        with torch.no_grad():
            for data_dict in loader:
                data = data_dict['data'][tio.DATA]
                label = data_dict['label'][tio.DATA]
                if dims == 2:
                    data = data.squeeze(2)
                    label = label.squeeze(2)
                if config.TRAIN.PARALLEL:
                    devices = config.TRAIN.DEVICES
                    data = data.cuda(devices[0])
                    label = label.cuda(devices[0])
                else:
                    device = determine_device()
                    data = data.to(device)
                    patches = get_patches(data, [64, 64, 64])    # torch.Size([8, 4, 64, 64, 64])
                    mri = patches[:, :3, :, :, :]
                    mri = mri.to(device)
                    flair = patches[:, 3:, :, :, :]
                    flair = flair.to(device)
                    PE_mri, PE_flair = [], []
                    # classify each patch
                    with torch.no_grad():
                        PE_mri = mri_class(mri)
                        PE_flair = ct_class(flair)
                        PE_data = torch.stack((PE_mri, PE_flair), dim=0)
                        PE_data = PE_data.unsqueeze(1)
                        image_data = ((mri, flair))
                    PE_data = PE_data.to(device)
                    label = label.to(device)
                with torch.amp.autocast('cuda'):
                    out = model(data)[0]
                    out = reconstructed_patches(out, out.shape[-1])
                    out = nonline(out)
                locations = data_dict[tio.LOCATION]
                if dims == 2:
                    out = out.unsqueeze(1)
                aggregator.add_batch(out, locations)
                label_aggregator.add_batch(label, locations)
                print(torch.unique(label), label.shape, out.shape)
            # final prediction
            pred = aggregator.get_output_tensor()
            final_label = label_aggregator.get_output_tensor()
            pred = torch.argmax(pred, dim=0).cpu().numpy()
            label = final_label.cpu().numpy()
            name = case['name']
            # quantitative analysis
            scores[name] = {}
            for c in np.unique(label):
                scores[name][int(c)] = dc(pred==c, label==c)
                perfs[int(c)].update(scores[name][c])
        del case
        del data_dict, data, label, out
        gc.collect()
        torch.cuda.empty_cache()
    logger.info('------------ dice scores ------------')
    logger.info(scores)
    for c in range(num_classes):
        logger.info(f'class {c} dice mean: {perfs[c].avg}')
    logger.info('------------------------------------')
    perf = np.mean([perfs[c].avg for c in range(1, num_classes)])
    return perf


def inference_brats(model, mri_class, ct_class, dataset, logger, config):
    model.eval()

    perfs = {
        'WT_dc': AverageMeter(),
        'WT_hd95': AverageMeter(),
        'WT_sensitivity': AverageMeter(),
    }
    nonline = nn.Softmax(dim=1)
    scores = {}
    for case in dataset:
        dims = config.MODEL.NUM_DIMS
        patch_size = config.INFERENCE.PATCH_SIZE
        patch_overlap = config.INFERENCE.PATCH_OVERLAP
        if dims == 2:
            patch_size = [1] + patch_size
            patch_overlap = [0] + patch_overlap
        target_shape = np.max([patch_size, case['data'].shape[1:]], 0)
        transform = tio.CropOrPad(target_shape)
        case = transform(case)
        sampler = tio.inference.GridSampler(case, patch_size, patch_overlap)
        loader = torch.utils.data.DataLoader(sampler, config.INFERENCE.BATCH_SIZE)
        aggregator = tio.inference.GridAggregator(sampler, 'average')

        with torch.no_grad():
            for data_dict in loader:
                data = data_dict['data'][tio.DATA]
                label = data_dict['label'][tio.DATA]
                if dims == 2:
                    data = data.squeeze(2)
                    label = label.squeeze(2)
                if config.TRAIN.PARALLEL:
                    devices = config.TRAIN.DEVICES
                    data = data.cuda(devices[0])
                    label = label.cuda(devices[0])
                else:
                    device = determine_device()
                    data = data.to(device)
                    patches = get_patches(data, [64, 64, 64])
                    mri = patches[:, :3, :, :, :]
                    mri = mri.to(device)
                    flair = patches[:, 3:, :, :, :]
                    flair = flair.to(device)
                    PE_mri, PE_flair = [], []
                    with torch.no_grad():
                        PE_mri = mri_class(mri)
                        PE_flair = ct_class(flair)
                        PE_data = torch.stack((PE_mri, PE_flair), dim=0)
                        PE_data = PE_data.unsqueeze(1)
                        image_data = ((mri, flair))
                    PE_data = PE_data.to(device)
                    label = label.to(device)
                with torch.amp.autocast('cuda'):
                    out = model(data)[0]
                    out = reconstructed_patches(out, out.shape[-1])
                    out = nonline(out)
                locations = data_dict[tio.LOCATION]
                if dims == 2:
                    out = out.unsqueeze(2)
                aggregator.add_batch(out, locations)
            pred = aggregator.get_output_tensor()
            pred = torch.argmax(pred, dim=0).cpu().numpy()
            label = case['label'][tio.DATA][0].numpy()
            name = case['name']
            scores[name] = {}
            # WT metrics
            scores[name]['WT_dc'] = dc(pred>0, label>0)
            perfs['WT_dc'].update(scores[name]['WT_dc'])
            scores[name]['WT_sensitivity'] = sensitivity(pred>0, label>0)
            perfs['WT_sensitivity'].update(scores[name]['WT_sensitivity'])
            if (pred>0).any() and (label>0).any():
                scores[name]['WT_hd95'] = hd95(pred>0, label>0)
                perfs['WT_hd95'].update(scores[name]['WT_hd95'])
        del case
    logger.info('------------ dice scores ------------')
    logger.info(scores)
    for c in perfs.keys():
        logger.info(f'{c} mean: {perfs[c].avg}')
    logger.info('------------------------------------')
    perf = np.mean([perfs[c].avg for c in perfs.keys()])
    return perf


def get_patches(x, patch_size):
    """
    Extract patches from an input tensor.

    Args:
        x: Input tensor, shape [batch, in_channels, img_size[0], img_size[1], img_size[2]].
        patch_size: The patch dimensions.

    Returns:
        patches: Extracted patches of shape [batch, n_patches, in_channels, patch_size[0], patch_size[1], patch_size[2]].
    """
    batch_size, _, *img_size = x.shape
    patches = []

    # Extract patches from each dimension
    for idx in range(0, img_size[0], patch_size[0]):
        for jdx in range(0, img_size[1], patch_size[1]):
            for kdx in range(0, img_size[2], patch_size[2]):
                patch = x[:, :, idx:idx + patch_size[0], jdx:jdx + patch_size[1], kdx:kdx + patch_size[2]]
                patches.append(patch)

    patches = torch.stack(patches)  # [n_patches, batch_size, in_channels, patch_size[0], patch_size[1], patch_size[2]]
    
    if batch_size == 1:
        patches = patches.squeeze(1)

    return patches


def reconstructed_patches(x, num):
    """
    Reconstruct full image from patch predictions.

    Args:
        x: Input tensor, shape [batch, in_channels, patch_size[0], patch_size[1], patch_size[2]].
        num: Number of patches (unused placeholder).

    Returns:
        reconstructed: Reconstructed tensor of original image shape.
    """
    batch_size = config.TRAIN.BATCH_SIZE                          
    _, _, *patch_size = x.shape
    img_size = [patch_size[0]*3, patch_size[1]*4, patch_size[2]*4]                                  

    reconstructed = torch.zeros(batch_size, 2, img_size[0], img_size[1], img_size[2], device=x.device)
    patch_idx = 0
    
    for idx in range(0, img_size[0], patch_size[0]):
        for jdx in range(0, img_size[1], patch_size[1]):
            for kdx in range(0, img_size[2], patch_size[2]):
                reconstructed[:, :, idx:idx + patch_size[0], jdx:jdx + patch_size[1], kdx:kdx + patch_size[2]] = x[patch_idx, :, :, :, :]
                patch_idx += 1

    return reconstructed
