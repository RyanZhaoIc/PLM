import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from numpy.testing import assert_array_almost_equal
from sklearn.model_selection import KFold

from utils.clothing1M import Clothing1M
from utils.custom_dataset import NoiseDataset, CandidateDateset

mean, std, crop_size = {}, {}, {}
mean['cifar-10'] = [x / 255 for x in [125.3, 123.0, 113.9]]
mean['mnist'] = [0.1307]
mean['cifar-100'] = [0.5071, 0.4867, 0.4408]
mean['clothing1m'] = [0.485, 0.456, 0.406]
# mean['clothing1m'] = [0.7215, 0.6846, 0.6679]
# mean['clothing1m'] = [0.6960, 0.6544, 0.6379]

std['cifar-10'] = [x / 255 for x in [63.0, 62.1, 66.7]]
std['mnist'] = [0.3081]
std['cifar-100'] = [0.2675, 0.2565, 0.2761]
std['clothing1m'] = [0.229, 0.224, 0.225]
# std['clothing1m'] = [0.3021, 0.3123, 0.3167]
# std['clothing1m'] = [0.3085, 0.3170, 0.3189]

crop_size['cifar-10'] = 32
crop_size['mnist'] = 28
crop_size['cifar-100'] = 32
crop_size['clothing1m'] = 224


def vectorized_choice(p, items):
    r = np.random.rand(p.shape[0])
    q = np.cumsum(p, axis=1)
    r = q < r[:, None]
    q[r] = 1.1
    k = q.argmin(axis=-1)
    return items[k]


def noisify_instance(train_data, train_labels, noise_rate, feature_size, num_classes, seed):
    np.random.seed(seed)
    train_labels = np.array(train_labels)
    q = np.random.normal(loc=noise_rate, scale=0.1, size=1000000)
    q = q[(q > 0) & (q < 1)]
    q = q[:len(train_labels)]
    w = np.random.normal(loc=0, scale=1, size=(feature_size, num_classes))

    sample = train_data.reshape(len(train_labels), feature_size)
    p_all = np.matmul(sample, w)
    p_all[np.arange(len(p_all)), train_labels.tolist()] = -1000000
    p_all = q[:, None] * F.softmax(torch.tensor(p_all), dim=1).numpy()
    p_all[np.arange(len(p_all)), train_labels.tolist()] = 1 - q
    p = p_all / p_all.sum(1)[:, None]
    flip_true = p.tolist()
    noisy_labels = vectorized_choice(p, items=np.arange(num_classes))
    flip_true = np.array(flip_true)
    noisy_labels = noisy_labels.flatten()
    print("Real noise rate: ", (noisy_labels != train_labels).mean())
    return noisy_labels, flip_true


def generate_noise_labels(train_labels, num_classes, data_type, flip_rate, seed, train_data=None):
    np.random.seed(seed)
    train_labels = np.array(train_labels)
    if len(train_labels) <= 0 or flip_rate == 0:
        return train_labels
    if data_type == 'symmetry':
        T_diag = np.eye(num_classes, num_classes) * (1 - flip_rate)
        T = T_diag + (1 - np.eye(num_classes, num_classes)) * (flip_rate / (num_classes - 1))
    elif data_type == 'pair':
        T_diag = np.eye(num_classes, num_classes) * flip_rate
        T = np.eye(num_classes, num_classes) * (1 - flip_rate)
        T = T + np.roll(T_diag, -1, axis=0)
    elif data_type == 'idn':
        feature_size = 1
        for i in train_data[0].shape:
            feature_size *= i
        noise_label, flip_true = noisify_instance(train_data, train_labels, flip_rate, feature_size, num_classes, seed)
        return np.array(noise_label)
    else:
        raise ValueError('Invalid data generate strategy')
    print(T)
    assert_array_almost_equal(T.sum(axis=1), np.ones(T.shape[1]))

    noise_label = np.array(train_labels.copy()) + num_classes
    for i in range(0, num_classes):
        noise_label[noise_label == (i + num_classes)] = np.random.choice(
            a=np.arange(num_classes),
            size=noise_label[noise_label == (i + num_classes)].__len__(),
            replace=True, p=T[i])
    print((noise_label == train_labels).mean())
    return noise_label


def get_transform(dataname, train=True):
    mean_, std_, crop_size_ = mean[dataname], std[dataname], crop_size[dataname]
    if train:
        if dataname in ['mnist', 'kmnist', 'fashion_mnist']:
            return transforms.Compose([
                transforms.RandomCrop(crop_size_, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean_, std_)])
            # , padding_mode = 'reflect'
        elif dataname in ['cifar-10', 'cifar-100']:
            return transforms.Compose([
                transforms.RandomCrop(crop_size_, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean_, std_)])
        elif dataname in ['stl-10']:
            return transforms.Compose([
                transforms.RandomCrop(crop_size_, padding=12),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean_, std_)])
        elif dataname in ['clothing1m']:
            return transforms.Compose([
                # transforms.Resize(256),
                # transforms.RandomCrop(crop_size_),
                transforms.Resize([256, 256]),
                transforms.RandomCrop(crop_size_),
                transforms.ToTensor(),
                transforms.Normalize(mean_, std_)
            ])
        else:
            raise ValueError('Invalid Transform')
    else:
        if dataname == 'clothing1m':
            return transforms.Compose([
                # transforms.Resize(256),
                transforms.Resize([256, 256]),
                transforms.CenterCrop(crop_size_),
                transforms.ToTensor(),
                transforms.Normalize(mean_, std_)
            ])
        else:
            return transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(mean_, std_)])


def five_cut_transform(dataname, crop_ratio=0.8):
    if dataname == 'cifar-10n':
        dataname = 'cifar-10'
    if dataname == 'cifar-100n':
        dataname = 'cifar-100'
    mean_, std_, crop_size_ = mean[dataname], std[dataname], crop_size[dataname]
    if dataname == 'clothing1m':
        return transforms.Compose([
            transforms.Resize(crop_size_),
            transforms.FiveCrop(int(crop_size_ * crop_ratio)),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([transforms.Resize(crop_size_)(crop) for crop in crops])),
            transforms.Lambda(
                lambda crops: torch.stack([transforms.Normalize(mean_, std_)(crop) for crop in crops])),
        ])
    else:
        return transforms.Compose([
            transforms.FiveCrop(int(crop_size_ * crop_ratio)),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([transforms.Resize(crop_size_)(crop) for crop in crops])),
            transforms.Lambda(
                lambda crops: torch.stack([transforms.Normalize(mean_, std_)(crop) for crop in crops])),
        ])


def get_origin_datasets(dataname, transform, data_root):
    if dataname == 'mnist':
        ordinary_train_dataset = dsets.MNIST(root=data_root, train=True, transform=transform,
                                             download=True)
        test_dataset = dsets.MNIST(root=data_root, train=False, transform=transform)
        num_classes = 10
    elif dataname == 'cifar-10':
        ordinary_train_dataset = dsets.CIFAR10(root=data_root, train=True, transform=transform, download=True)
        test_dataset = dsets.CIFAR10(root=data_root, train=False, transform=transform)
        num_classes = 10
    elif dataname == 'cifar-100':
        ordinary_train_dataset = dsets.CIFAR100(root=data_root, train=True, transform=transform, download=True)
        test_dataset = dsets.CIFAR100(root=data_root, train=False, transform=transform)
        num_classes = 100
    elif dataname == 'clothing1m':
        ordinary_train_dataset = Clothing1M(root=data_root, train=True, transform=transform)
        test_dataset = Clothing1M(root=data_root, train=False, transform=transform)
        num_classes = 14
    else:
        raise ValueError('Invalid dataset')
    return ordinary_train_dataset, test_dataset, num_classes


def indices_split(len_dataset, seed, val_ratio=None, split_group=None):
    assert val_ratio is not None or split_group is not None
    indices = np.arange(len_dataset)
    np.random.seed(seed)
    if split_group is not None:
        assert split_group > 1
        kf = KFold(n_splits=split_group, shuffle=True, random_state=seed)
        train_indices_list, val_indices_list = [], []
        for train_indices, val_indices in kf.split(indices):
            train_indices_list.append(train_indices)
            val_indices_list.append(val_indices)
        return train_indices_list, val_indices_list
    else:
        assert val_ratio > 0
        val_indices = np.random.choice(len(indices), int(val_ratio * len(indices)), replace=False)
        train_indices = np.delete(indices, val_indices)
        return train_indices.tolist(), val_indices.tolist()


def get_noise_dataset(dataset, noise_labels, transformations, indices=None):
    if indices is None:
        indices = list(range(len(dataset)))
    dset = NoiseDataset(data=dataset.data[indices],
                        noise_tar=noise_labels[indices],
                        transform=transformations)
    return dset


def get_cantar_dataset(dataset, candidate_labels, transformations, targets=None, indices=None, return_index=False):
    if targets is None:
        targets = np.array(dataset.targets)
    if indices is None:
        indices = list(range(len(dataset)))
        dset = CandidateDateset(dataset.data[indices].squeeze(), candidate_labels[indices].squeeze(),
                                targets[indices].squeeze(), transformations, return_index)
    else:
        dset = CandidateDateset(dataset.data[indices], candidate_labels[indices], targets[indices], transformations,
                                return_index)
    return dset


class BalancedSampler(object):
    def __init__(self, num_classes, labels):
        self.num_classes = num_classes
        self.labels = labels
        self.class_idx, self.len_min = self.class_index_collect()

    def class_index_collect(self):
        len_min = len(self.labels)
        class_idx = []
        for i in range(self.num_classes):
            idi = np.flatnonzero(self.labels == i)
            len_min = len(idi) if len_min > len(idi) else len_min
            class_idx.append(idi)
        assert class_idx != []
        return class_idx, len_min

    def ind_unsampling(self):
        sampled_indices = []
        for i in range(self.num_classes):
            sampled_indices.extend(np.random.choice(self.class_idx[i], self.len_min, replace=False).tolist())
        assert len(sampled_indices) == self.len_min * self.labels
        return sampled_indices
