import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import torch

train_losses = []
train_acc = []
running_loss = 0.0

def getOptimizer(model, lr=0.001, momentum=0.9, weight_decay=0):
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
  return optimizer
  

def L1_Loss(model, loss, l1_factor=0.0005):
    # l1_crit = nn.L1Loss(size_average=False)
    reg_loss = 0
    for param in model.parameters():
          # zero_vector = torch.rand_like(param) * 0
          reg_loss += torch.sum(param.abs())

    loss += l1_factor * reg_loss
    return loss

def train(model, device, train_loader, optimizer, criterion, l1_factor=0):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  running_loss = 0.0
  for batch_idx, (data, target) in enumerate(pbar):
      # get samples
      data, target = data.to(device), target.to(device)

      # Init
      optimizer.zero_grad()
      # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
      # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

      # Predict
      y_pred = model(data)

      # Calculate loss
      loss = criterion(y_pred, target)
      if l1_factor:
        loss = L1_Loss(model, loss)
      running_loss += loss.item()
      train_losses.append(loss.item())

      # Backpropagation
      loss.backward()
      optimizer.step()

      # Update pbar-tqdm
      
      pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()
      processed += len(data)

      # print('Running loss : ', running_loss)
      pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
      train_acc.append(100*correct/processed)