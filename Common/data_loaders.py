import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tiny_imagenet import *


def getDataLoaderArgs(batchSize):
  BATCH_SIZE = batchSize
  SEED = 1

  # CUDA?
  cuda = torch.cuda.is_available()
  print("CUDA Available?", cuda)

  # For reproducibility
  torch.manual_seed(SEED)

  if cuda:
      torch.cuda.manual_seed(SEED)

  dataloader_args = dict(shuffle=True, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
  return dataloader_args


def getTrainDataLoader(type, dataloader_args, data_transforms, trainset=[]):
  trainloader = []
  if type == 'CIFAR':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=data_transforms)

  trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)
  return trainloader

def getTestDataLoader(type, dataloader_args, data_transforms, testset=[]):
  testloader = []
  if type == 'CIFAR':
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=data_transforms)
  testloader = torch.utils.data.DataLoader(testset, **dataloader_args)
  return testloader

def get_tiny_imagenet_dataset(train_split = 70,test_transforms = None,train_transforms = None):
  return TinyImageNetDataSet(train_split = train_split,test_transforms = test_transforms,train_transforms = train_transforms)