import torch.nn as nn
import torch
import torch.nn.functional as F

test_losses = []
test_acc = []

def __test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    test_acc.append(100. * correct / len(test_loader.dataset))

# def class_performance(net, testloader, device):
#   class_correct = list(0. for i in range(10))
#   class_total = list(0. for i in range(10))
#   with torch.no_grad():
#       for data in testloader:
#           images, labels = data
#           images, labels = images.to(device), labels.to(device)
#           outputs = net(images)
#           _, predicted = torch.max(outputs, 1)
#           c = (predicted == labels).squeeze()
#           for i in range(4):
#               label = labels[i]
#               class_correct[label] += c[i].item()
#               class_total[label] += 1
#   for i in range(10):
#       print('Accuracy of %5s : %2d %%' % (
#           classes[i], 100 * class_correct[i] / class_total[i]))
      
# def predict_on_test(net, testloader, device ):
#   dataiter = iter(testloader)
#   images, labels = dataiter.next()
#   images, labels = images.to(device), labels.to(device)
#   # print images
#   imshow(torchvision.utils.make_grid(images.cpu()))
#   print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(64)))
#   # print('GroundTruth: ', '%5s' % classes[labels])
#   outputs = net(images)
#   _, predicted = torch.max(outputs, 1)
#   print('Predicted: ', '%5s' % ' '.join('%5s' % classes[predicted[j]] for j in range(64)))