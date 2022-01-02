import random
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
import os

kwargs = {'num_workers': 2, 'pin_memory': True}


def load_cifar10(batch_size=128, augment=False):
    # setup data loader
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, **kwargs)
    testset = torchvision.datasets.CIFAR10(
        root='../data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, **kwargs)

    if augment == True:
        cifar10_mean = (0.4914, 0.4822, 0.4465)
        cifar10_std = (0.2471, 0.2435, 0.2616)
    else:
        cifar10_mean = (0., 0., 0.)
        cifar10_std = (1., 1., 1.)

    im_mean = torch.tensor(cifar10_mean).cuda().view(1, 3, 1, 1)
    im_std = torch.tensor(cifar10_std).cuda().view(1, 3, 1, 1)

    return train_loader, test_loader, im_mean, im_std


def load_cifar100(batch_size=128, augment=False):
    # setup data loader
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR100(
        root='../data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, **kwargs)
    testset = torchvision.datasets.CIFAR100(
        root='../data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, **kwargs)

    if augment == True:
        cifar100_mean = (0.5071, 0.4867, 0.4408)
        cifar100_std = (0.2675, 0.2565, 0.2761)
    else:
        cifar100_mean = (0., 0., 0.)
        cifar100_std = (1., 1., 1.)

    im_mean = torch.tensor(cifar100_mean).cuda().view(1, 3, 1, 1) 
    im_std = torch.tensor(cifar100_std).cuda().view(1, 3, 1, 1) 

    return train_loader, test_loader, im_mean, im_std


def load_tiny(batch_size=128, augment=False):
    data_dir = '../tinyimagenet/tiny-imagenet-200'
    train_transform = transforms.Compose([
        # transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        # transforms.Normalize(tiny_mean, tiny_std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(tiny_mean, tiny_std),
    ])
    num_workers = 2
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                         train_transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'),
                                        test_transform)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )

    if augment == True:
        # tiny_mean = (0.4802, 0.4481, 0.3975)
        # tiny_std = (0.2302, 0.2265, 0.2262)

        tiny_mean = (0.485, 0.456, 0.406)
        tiny_std = (0.229, 0.224, 0.225)
    else:
        tiny_mean = (0., 0., 0.)
        tiny_std = (1., 1., 1.)

    im_mean = torch.tensor(tiny_mean).cuda().view(1, 3, 1, 1) 
    im_std = torch.tensor(tiny_std).cuda().view(1, 3, 1, 1) 

    return train_loader, test_loader, im_mean, im_std
