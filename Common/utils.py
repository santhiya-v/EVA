import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid
from textwrap import wrap

def imshow(img, title=None, normalizeVal=0.5):
    img = img / 2 + normalizeVal     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)

def getDevice():
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  return device

def showImages(images, titles):
  fig3 = plt.figure(figsize = (25,15))
  for i, im in enumerate(images):
      sub = fig3.add_subplot(5, 5, i+1)
      plt.imshow(im[0].permute(1, 2, 0).cpu().numpy().squeeze(), cmap='gray_r',interpolation='none')
      sub.set_title("\n".join(wrap(titles[i])))
  plt.tight_layout()
  plt.show()

def getPredActualTitle(output, classes):
  titles = []
  for im in output:
    titles.append("Prediction : %s, Actual: %s" % (classes[im[1].data.cpu().numpy()[0]], classes[im[2].data.cpu().numpy()[0]]))
  return titles

def getMisclassifiedImages(modelClass, test_loader, device, modelPath):
  model = modelClass
  model.load_state_dict(torch.load(modelPath))
  model.cuda()
  model.eval()
  misclassifiedImages = []
  with torch.no_grad():
      for data, target in test_loader:
          data, target = data.to(device), target.to(device)
          output = model(data)
          pred = output.argmax(dim=1, keepdim=True)
          target_modified = target.view_as(pred)
          for i in range(len(pred)):
            if pred[i].item()!= target_modified[i].item():
                misclassifiedImages.append([data[i], pred[i], target_modified[i]])
  return misclassifiedImages

def plotMisclassifiedImages(misclassifiedImages, classes, noOfImages=25):
  titles = getPredActualTitle(misclassifiedImages[:noOfImages], classes)
  showImages(misclassifiedImages[:noOfImages], titles)


def saveModel(model, modelPath):
  torch.save(model.state_dict(), modelPath)


def showFewDataSetImages(loader, noOfImages=10, normalizeVal=0.5):
  image, label = iter(loader).next()
  img = make_grid(image[0:noOfImages])
  img = img / 2 + normalizeVal     # unnormalize
  npimg = img.numpy()
  fig = plt.figure(figsize=(10,10))
  plt.imshow(np.transpose(npimg, (1, 2, 0)))

def getTinyImageNetWordClasses(wordsPath, classes):
  url = wordsPath
  f = open(url, "r")
  words = [None] * 200
  for line in f:
    wordclass = line.strip('\n').split('\t')[0]

    if wordclass in classes:
      i = classes.index(line.strip('\n').split('\t')[0])
      words[i] = line.strip('\n').split('\t')[1]
  return words