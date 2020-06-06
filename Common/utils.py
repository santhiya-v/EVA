import matplotlib.pyplot as plt
import numpy as np
import torch

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def getDevice():
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  return device

def showImages(images, titles):
  fig3 = plt.figure(figsize = (15,15))
  for i, im in enumerate(images):
      sub = fig3.add_subplot(5, 5, i+1)
      plt.imshow(im[0].permute(1, 2, 0).cpu().numpy().squeeze(), cmap='gray_r',interpolation='none')
      sub.set_title(titles[i])
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