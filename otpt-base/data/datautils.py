import os
from typing import Tuple
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from data.hoi_dataset import BongardDataset
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from data.fewshot_datasets import *
import data.augmix_ops as augmentations

import ipdb

ID_to_DIRNAME={
    'C':'covidqu',
    'I': 'ImageNet',
    'A': 'imagenet-a',
    'K': 'ImageNet-Sketch/sketch',
    'R': 'imagenet-r',
    'V': 'imagenetv2-matched-frequency-format-val',
    'flower102': 'Flower102',
    'dtd': 'DTD',
    'pets': 'OxfordPets',
    'stanfordcar': 'StanfordCar',
    'ucf101': 'UCF101',
    'caltech101': 'Caltech101',
    'food101': 'Food101',
    'sun397': 'SUN397',
    'aircraft': 'fgvc_aircraft',
    'eurosat': 'eurosat',
    'eurosat_tv': 'eurosat_tv',
    'dermamnist': 'dermamnist',
}

distortions = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                'defocus_blur', 'glass_blur',
                'zoom_blur', 'frost',
                'brightness', 'contrast', 'elastic_transform',
                'pixelate','fog','speckle_noise','saturate', 'spatter', 'gaussian_blur']


class _MedMNISTAdapter(torch.utils.data.Dataset):
    """medmnist datasets return (PIL, np.ndarray-shape-[1]) — squash the label
    to an int and pass PIL through the given transform (which is either a plain
    torchvision Compose or an AugMixAugmenter that yields a list of tensors)."""

    def __init__(self, raw, transform):
        self.raw = raw
        self.transform = transform

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        img, label = self.raw[idx]
        # medmnist labels are numpy arrays of shape [1] for multi-class tasks.
        if hasattr(label, "item"):
            label = int(label.item()) if label.size == 1 else int(label[0])
        else:
            label = int(label)
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(label).long()


def build_dataset(set_id, transform, data_root, mode='test', n_shot=None, split="all", bongard_anno=False):
    
    if set_id == 'I':
        # ImageNet validation set
        testdir = os.path.join(os.path.join(data_root, ID_to_DIRNAME[set_id]), 'val')
        testset = datasets.ImageFolder(testdir, transform=transform)

    elif set_id == 'C':
        
        testdir = os.path.join(os.path.join(data_root, ID_to_DIRNAME[set_id]), 'Infection Segmentation Data/Val')
        testset = datasets.ImageFolder(testdir, transform=transform)

    elif set_id in ['A', 'K', 'R', 'V']:
        testdir = os.path.join(data_root, ID_to_DIRNAME[set_id])
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id in fewshot_datasets:
        if mode == 'train' and n_shot:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode, n_shot=n_shot)
        else:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode)

    elif set_id == 'eurosat_tv':
        # Backbone-swap pilot: torchvision EuroSAT. Class order is alphabetical
        # (AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture,
        #  PermanentCrop, Residential, River, SeaLake) — must match `eurosat_tv_classes`.
        root = os.path.join(data_root, ID_to_DIRNAME[set_id])
        os.makedirs(root, exist_ok=True)
        testset = datasets.EuroSAT(root=root, download=True, transform=transform)

    elif set_id == 'dermamnist':
        # Backbone-swap pilot: DermaMNIST at 224x224 via medmnist>=3.0.
        # medmnist ships PIL images; the O-TPT pipeline expects a torchvision-style
        # dataset returning (image, int_label). We adapt here.
        import medmnist  # local import so envs without medmnist can still import datautils
        root = os.path.join(data_root, ID_to_DIRNAME[set_id])
        os.makedirs(root, exist_ok=True)
        split = {'test': 'test', 'train': 'train', 'val': 'val'}.get(mode, 'test')
        raw = medmnist.DermaMNIST(split=split, size=224, download=True, root=root, as_rgb=True)
        testset = _MedMNISTAdapter(raw, transform)

    elif set_id == 'bongard':
        assert isinstance(transform, Tuple)
        base_transform, query_transform = transform
        testset = BongardDataset(data_root, split, mode, base_transform, query_transform, bongard_anno)
    else:
        raise NotImplementedError
        
    return testset


# AugMix Transforms
def get_preaugment():
    return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])

def augmix(image, preprocess, aug_list, severity=1):
    preaugment = get_preaugment()
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    if len(aug_list) == 0:
        return x_processed
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix
    return mix


class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False, 
                    severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        if augmix:
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = []
        self.severity = severity
        
    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        views = [augmix(x, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
        return [image] + views



