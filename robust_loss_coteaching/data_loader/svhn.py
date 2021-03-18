import sys

import numpy as np
from PIL import Image
import torchvision
from torch.utils.data.dataset import Subset
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import torch
import torch.nn.functional as F
import random
import json
import os
import copy

def fix_seed(seed=888):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    
def get_svhn(root, cfg_trainer, train=True,
             transform_train=None, transform_val=None,
             download=True, noise_file='', teacher_idx=None):
    split = 'train' if train is True else 'test'
    base_dataset = torchvision.datasets.SVHN(root, split=split, download=download)
    if train:
        fix_seed()
        train_idxs, val_idxs = train_val_split(base_dataset.labels)
        
        train_dataset = SVHN_train(root, cfg_trainer, train_idxs, split='train', transform=transform_train)
        val_dataset = SVHN_val(root, cfg_trainer, val_idxs, split='train', transform=transform_val)
        
        if cfg_trainer['asym']:
            train_dataset.asymmetric_noise()
            val_dataset.asymmetric_noise()
        else:
            train_dataset.symmetric_noise()
            val_dataset.symmetric_noise()
            
        if teacher_idx:
            print(len(teacher_idx))
            train_dataset.truncate(teacher_idx)
            
        print(f"Train: {len(train_dataset)} Val: {len(val_dataset)}")  # Train: 45000 Val: 5000
    else:
        fix_seed()
        train_dataset = []
        val_dataset = SVHN_val(root, cfg_trainer, None, split='test', transform=transform_val)
        print(f"Test: {len(val_dataset)}")
        
    if len(val_dataset) == 0:
        return train
    else:
        return train_dataset, val_dataset
        
def train_val_split(base_dataset: torchvision.datasets.SVHN):
    fix_seed()
    num_classes = 10
    base_dataset = np.array(base_dataset)
    train_n = int(len(base_dataset) * 1.0 / num_classes)
    train_idxs = []
    val_idxs = []

    for i in range(num_classes):
        idxs = np.where(base_dataset == i)[0]
        np.random.shuffle(idxs)
        train_idxs.extend(idxs[:train_n])
        val_idxs.extend(idxs[train_n:])
    np.random.shuffle(train_idxs)
    np.random.shuffle(val_idxs)

    return train_idxs, val_idxs

class SVHN_train(torchvision.datasets.SVHN):
    def __init__(self, root, cfg_trainer, indexs, split='train',
                 transform=None, target_transform=None,
                 download=False):
        super(SVHN_train, self).__init__(root, split=split,
                                         transform=transform, target_transform=target_transform,
                                         download=download)
        
        fix_seed()
        self.num_classes = 10
        self.cfg_trainer = cfg_trainer
        self.train_data = self.data[indexs]
        self.train_labels = np.array(self.labels)[indexs]
        self.indexs = indexs
        self.prediction = np.zeros((len(self.train_data), self.num_classes, self.num_classes), dtype=np.float32)
        self.noise_indx = []
        
    def symmetric_noise(self):
        self.train_labels_gt = self.train_labels.copy()
        fix_seed()
        indices = np.random.permutation(len(self.train_data))
        for i, idx in enumerate(indices):
            if i < self.cfg_trainer['percent'] * len(self.train_data):
                self.noise_indx.append(idx)
                self.train_labels[idx] = np.random.randint(self.num_classes, dtype=np.int32)
                
    def asymmetric_noise(self):
        self.train_labels_gt = copy.deepcopy(self.train_labels)
        fix_seed()
        
        for i in range(self.num_classes):
            indices = np.where(self.train_labels == i)[0]
            np.random.shuffle(indices)
            for j, idx in enumerate(indices):
                if j < self.cfg_trainer['percent'] * len(indices):
                    self.noise_indx.append(idx)
                    # 2 -> 7
                    if i == 2:
                        self.train_labels[idx] = 7
                    # 3 -> 8
                    if i == 3:
                        self.train_labels[idx] = 8
                    # 5 <-> 6
                    if i == 5:
                        self.train_labels[idx] = 6
                    if i == 6:
                        self.train_labels[idx] = 5
                    # 7 -> 1
                    if i == 7:
                        self.train_labels[idx] = 1
                        
    def truncate(self, teacher_idx):
        self.train_data = self.train_data[teacher_idx]
        self.train_labels = self.train_labels[teacher_idx]
        self.train_labels_gt = self.train_labels_gt[teacher_idx]
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
            
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, target_gt = self.train_data[index], self.train_labels[index], self.train_labels_gt[index]
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target, index, target_gt

    def __len__(self):
        return len(self.train_data)
    
class SVHN_val(torchvision.datasets.SVHN):
    def __init__(self, root, cfg_trainer, indexs, split='train',
                 transform=None, target_transform=None,
                 download=False):
        super(SVHN_val, self).__init__(root, split=split,
                                       transform=transform, target_transform=target_transform,
                                       download=download)
        
        self.num_classes = 10
        self.cfg_trainer = cfg_trainer
        if split == 'train':
            self.train_data = self.data[indexs]
            self.train_labels = np.array(self.labels)[indexs]
        else:
            self.train_data = self.data
            self.train_labels = np.array(self.labels)
        self.train_labels_gt = self.train_labels.copy()
        
    def symmetric_noise(self):
        indices = np.random.permutation(len(self.train_data))
        for i, idx in enumerate(indices):
            if i < self.cfg_trainer['percent'] * len(self.train_data):
                self.train_labels[idx] = np.random.randint(self.num_classes, dtype=np.int32)
                
    def asymmetric_noise(self):
        for i in range(self.num_classes):
            indices = np.where(self.train_labels == i)[0]
            np.random.shuffle(indices)
            for j, idx in enumerate(indices):
                if j < self.cfg_trainer['percent'] * len(indices):
                    self.noise_indx.append(idx)
                    # 2 -> 7
                    if i == 2:
                        self.train_labels[idx] = 7
                    # 3 -> 8
                    if i == 3:
                        self.train_labels[idx] = 8
                    # 5 <-> 6
                    if i == 5:
                        self.train_labels[idx] = 6
                    if i == 6:
                        self.train_labels[idx] = 5
                    # 7 -> 1
                    if i == 7:
                        self.train_labels[idx] = 1
                        
    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, target_gt = self.train_data[index], self.train_labels[index], self.train_labels_gt[index]


        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index, target_gt