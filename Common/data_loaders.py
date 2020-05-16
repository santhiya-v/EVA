import torch
import torchvision
import torchvision.transforms as transforms


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

def getTrainDataLoader(dataloader_args, rotation=0, verticalFlip=0, horizontalFlip=0):
  transform = transforms.Compose(
    [transforms.RandomRotation((-rotation, rotation)),
     transforms.RandomVerticalFlip(p=verticalFlip),
     transforms.RandomHorizontalFlip(p=horizontalFlip),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)
  return trainloader

def getTestDataLoader(dataloader_args):
  transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
  testloader = torch.utils.data.DataLoader(testset, **dataloader_args)
  return testloader