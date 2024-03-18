import errno
import os
import random

import numpy as np
import torch
from easydict import EasyDict
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR


def mkdir_if_missing(directory):
    # Reference: https://github.com/wvangansbeke/Unsupervised-Classification.git
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def accuracy_check(loader, model, device):
    with torch.no_grad():
        model.eval()
        total, num_samples = 0, 0
        for images, labels in loader:
            labels, images = labels.to(device, non_blocking=True), images.to(device, non_blocking=True)
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            total += (preds == labels).sum().item()
            num_samples += labels.size(0)

    return total / num_samples


def accuracy_check_noise(loader, model, device, return_predict=False):
    with torch.no_grad():
        model.eval()
        total, num_samples = 0, 0
        preds = []
        for images, labels, _ in loader:
            labels, images = labels.to(device, non_blocking=True), images.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            preds.extend(predicted.cpu().tolist())
            total += (predicted == labels).sum().item()
            num_samples += labels.size(0)
        if return_predict:
            return total / num_samples, np.array(preds)
    return total / num_samples


def one_hot(x, num_class=None):
    if not isinstance(x, list):
        x = x.tolist()
    if not num_class:
        num_class = np.max(x) + 1
    ohx = np.zeros((len(x), num_class))
    ohx[range(len(x)), x] = 1
    return ohx


def get_paths(args):
    paths = EasyDict()
    paths['save_dir'] = os.path.join(args.save_dir, args.ds)
    mkdir_if_missing(paths['save_dir'])

    prefix = args.data_gen + '_' + str(args.flip_rate) + '_' + str(args.seed) + '_'
    paths['multi_labels'] = os.path.join(paths['save_dir'], prefix + 'multi_labels.npy')
    paths['pre_probs'] = os.path.join(paths['save_dir'], prefix + 'pre_probs.npy')
    paths['pre_model'] = os.path.join(paths['save_dir'], prefix + 'pre_model.pth')
    paths['best_model'] = os.path.join(paths['save_dir'], prefix + 'best_model.pth')
    paths['U_model'] = os.path.join(paths['save_dir'], prefix + 'U_model.pth')

    return paths


def init_gpuseed(seed, device, benchmark=True, deterministic=True):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(device)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    torch.manual_seed(seed)


def get_scheduler(dataname, optimizer, ep):
    if dataname == 'clothing1m':
        return MultiStepLR(optimizer, milestones=[5, 10, 15], gamma=0.1)  ######
    elif dataname == 'cifar-100':
        return MultiStepLR(optimizer, milestones=[40], gamma=0.01)
    elif dataname in ['mnist', 'cifar-10']:
        return CosineAnnealingLR(optimizer, T_max=ep)
    else:
        raise ValueError('Invalid Dataset')
