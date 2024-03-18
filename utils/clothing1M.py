import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class Clothing1M(Dataset):
    def __init__(self, root, train, transform=None):

        self.root = root + '/clothing1m'
        self.transform = transform
        self.train = train
        self.data = []
        self.targets = []
        self.preload()

    def preload(self):
        if self.train:
            imgs_path = self.root + '/noisy_train_key_list.txt'
            labels_path = self.root + '/noisy_label_kv.txt'
            img_save_path = self.root + '/images/train_imgs.npy'
            label_save_path = self.root + '/train_labels.npy'
            indices_save_path = self.root + '/train_paths_indices.npy'
        else:
            imgs_path = self.root + '/clean_test_key_list.txt'
            labels_path = self.root + '/clean_label_kv.txt'
            img_save_path = self.root + '/images/test_imgs.npy'
            label_save_path = self.root + '/test_labels.npy'
            indices_save_path = self.root + '/test_paths_indices.npy'

        if os.path.exists(img_save_path):
            self.data = np.load(img_save_path, allow_pickle=True, encoding='bytes')
            if not os.path.exists(label_save_path):
                img_path_indices = np.load(indices_save_path, allow_pickle=True, encoding='bytes')
                img_path_indices = img_path_indices.item()
        else:
            print('Load Images')
            images_count = 0
            with open(imgs_path, 'r') as f:
                lines = f.read().splitlines()
                print('Data Numbers: ' + str(len(lines)))
                img_path_indices = {}
                for i, l in enumerate(lines):
                    img_path = self.root + '/' + l
                    img_path_indices[img_path] = i
                    image = Image.open(img_path).convert('RGB')
                    image = np.asarray(image, dtype=np.uint8)
                    self.data.append(image)
                    images_count += 1
            np.save(indices_save_path, img_path_indices)
            self.data = np.asarray(self.data)
            np.save(img_save_path, self.data)

        if os.path.exists(label_save_path):
            self.targets = np.load(label_save_path, allow_pickle=True)
        else:
            print('Load Labels')
            self.targets = np.zeros(len(self.data), dtype=np.uint8)
            with open(labels_path, 'r') as f:
                lines = f.read().splitlines()
                labels_count = 0
                for l in lines:
                    entry = l.split()
                    img_path = self.root + '/' + entry[0]
                    if img_path in img_path_indices.keys():
                        labels_count += 1
                        self.targets[img_path_indices[img_path]] = np.asarray(entry[1], dtype=np.uint8)
            assert labels_count == len(self.data)
            np.save(label_save_path, self.targets)

    def __getitem__(self, index):
        img = self.data[index]
        target = self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)
