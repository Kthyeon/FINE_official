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

def get_cifar10(root, cfg_trainer, train=True,
                transform_train=None, transform_val=None,
                download=True, noise_file = '', teacher_idx=None, seed=888):
    
    base_dataset = torchvision.datasets.CIFAR10(root, train=train, download=download)
    if train:
        fix_seed(seed)
        train_idxs, val_idxs = train_val_split(base_dataset.targets, seed)

        train_dataset = CIFAR10_train(root, cfg_trainer, train_idxs, train=True, transform=transform_train, seed=seed)
        val_dataset = CIFAR10_val(root, cfg_trainer, val_idxs, train=train, transform=transform_val)
        if cfg_trainer['instance']:
            train_dataset.instance_noise()
            val_dataset.instance_noise()
        elif cfg_trainer['asym']:
            train_dataset.asymmetric_noise()
            val_dataset.asymmetric_noise()
        else:
            train_dataset.symmetric_noise()
            val_dataset.symmetric_noise()
            
        print ('##############')
        print (train_dataset.train_labels[:10])
        print (train_dataset.train_labels_gt[:10])
              
        if teacher_idx is not None:
            print(len(teacher_idx))
            train_dataset.truncate(teacher_idx)
            
        print(f"Train: {len(train_dataset)} Val: {len(val_dataset)}")  # Train: 45000 Val: 5000
    else:
        fix_seed(seed)
        train_dataset = []
        val_dataset = CIFAR10_val(root, cfg_trainer, None, train=train, transform=transform_val)
        print(f"Test: {len(val_dataset)}")
    
    if len(val_dataset) == 0:
        return train_dataset, None
    else:
        return train_dataset, val_dataset

def train_val_split(base_dataset: torchvision.datasets.CIFAR10, seed):
    fix_seed(seed)
    num_classes = 10
    base_dataset = np.array(base_dataset)
    train_n = int(len(base_dataset) * 0.9 / num_classes)
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

class CIFAR10_train(torchvision.datasets.CIFAR10):
    def __init__(self, root, cfg_trainer, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False, seed=888):
        super(CIFAR10_train, self).__init__(root, train=train,
                                            transform=transform, target_transform=target_transform,
                                            download=download)
        fix_seed(seed)
        self.num_classes = 10
        self.cfg_trainer = cfg_trainer
        self.train_data = self.data[indexs] # self.train_data[indexs]
        self.train_labels = np.array(self.targets)[indexs] # np.array(self.train_labels)[indexs]
        self.indexs = indexs
        self.prediction = np.zeros((len(self.train_data), self.num_classes, self.num_classes), dtype=np.float32)
        self.noise_indx = []
        self.seed = seed
                
    def symmetric_noise(self):
        self.train_labels_gt = self.train_labels.copy()
        fix_seed(self.seed)
        indices = np.random.permutation(len(self.train_data))
        for i, idx in enumerate(indices):
            if i < self.cfg_trainer['percent'] * len(self.train_data):
                self.noise_indx.append(idx)
                self.train_labels[idx] = np.random.randint(self.num_classes, dtype=np.int32)

    def asymmetric_noise(self):
        self.train_labels_gt = copy.deepcopy(self.train_labels)
        fix_seed(self.seed)

        for i in range(self.num_classes):
            indices = np.where(self.train_labels == i)[0]
            np.random.shuffle(indices)
            for j, idx in enumerate(indices):
                if j < self.cfg_trainer['percent'] * len(indices):
                    self.noise_indx.append(idx)
                    # truck -> automobile
                    if i == 9:
                        self.train_labels[idx] = 1
                    # bird -> airplane
                    elif i == 2:
                        self.train_labels[idx] = 0
                    # cat -> dog
                    elif i == 3:
                        self.train_labels[idx] = 5
                    # dog -> cat
                    elif i == 5:
                        self.train_labels[idx] = 3
                    # deer -> horse
                    elif i == 4:
                        self.train_labels[idx] = 7
                        
    def instance_noise(self):
        '''
        Instance-dependent noise
        https://github.com/haochenglouis/cores/blob/main/data/utils.py
        '''
        
        self.train_labels_gt = copy.deepcopy(self.train_labels)
        fix_seed(self.seed)
        
        q_ = np.random.normal(loc=self.cfg_trainer['percent'], scale=0.1, size=int(1e6))
        q = []
        for pro in q_:
            if 0 < pro < 1:
                q.append(pro)
            if len(q)==50000:
                break
                
        w = np.random.normal(loc=0, scale=1, size=(32*32*3, 10))
        for i, sample in enumerate(self.train_data):
            sample = sample.flatten()
            p_all = np.matmul(sample, w)
            p_all[self.train_labels_gt[i]] = -int(1e6)
            p_all = q[i] * F.softmax(torch.tensor(p_all), dim=0).numpy()
            p_all[self.train_labels_gt[i]] = 1 - q[i]
            self.train_labels[i] = np.random.choice(np.arange(10), p=p_all/sum(p_all))
        
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
        img = Image.fromarray(img)


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index, target_gt

    def __len__(self):
        return len(self.train_data)

class CIFAR10_val(torchvision.datasets.CIFAR10):

    def __init__(self, root, cfg_trainer, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_val, self).__init__(root, train=train,
                                          transform=transform, target_transform=target_transform,
                                          download=download)

        # self.train_data = self.data[indexs]
        # self.train_labels = np.array(self.targets)[indexs]
        self.num_classes = 10
        self.cfg_trainer = cfg_trainer
        if train:
            self.train_data = self.data[indexs]
            self.train_labels = np.array(self.targets)[indexs]
        else:
            self.train_data = self.data
            self.train_labels = np.array(self.targets)
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
                    # truck -> automobile
                    if i == 9:
                        self.train_labels[idx] = 1
                    # bird -> airplane
                    elif i == 2:
                        self.train_labels[idx] = 0
                    # cat -> dog
                    elif i == 3:
                        self.train_labels[idx] = 5
                    # dog -> cat
                    elif i == 5:
                        self.train_labels[idx] = 3
                    # deer -> horse
                    elif i == 4:
                        self.train_labels[idx] = 7
                        
                        
    def instance_noise(self):
        '''
        Instance-dependent noise
        https://github.com/haochenglouis/cores/blob/main/data/utils.py
        '''
        q_ = np.random.normal(loc=self.cfg_trainer['percent'], scale=0.1, size=int(1e6))
        q = []
        for pro in q_:
            if 0 < pro < 1:
                q.append(pro)
            if len(q)==50000:
                break
                
        w = np.random.normal(loc=0, scale=1, size=(32*32*3, 10))
        for i, sample in enumerate(self.train_data):
            sample = sample.flatten()
            p_all = np.matmul(sample, w)
            p_all[self.train_labels_gt[i]] = -int(1e6)
            p_all = q[i] * F.softmax(torch.tensor(p_all), dim=0).numpy()
            p_all[self.train_labels_gt[i]] = 1 - q[i]
            self.train_labels[i] = np.random.choice(np.arange(10), p=p_all/sum(p_all))
                        
                        
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
        img = Image.fromarray(img)


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index, target_gt
        