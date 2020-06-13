'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from gbn import *
num_splits = 2 

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))
        # out = self.bn2(self.conv2(out))
        # out += self.shortcut(x)
        # out = F.relu(out)
        return out


class ResNetS11(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetS11, self).__init__()
        self.in_planes = 128

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = GhostBatchNorm(64, num_splits)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bn2 = GhostBatchNorm(128, num_splits)

        self.res1 = self._make_layer(block, 128, 128, num_blocks[0], stride=1)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.bn3 = GhostBatchNorm(256, num_splits)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.bn4 = GhostBatchNorm(512, num_splits)

        self.res2 = self._make_layer(block, 512, 512, num_blocks[1], stride=1)
        self.pool5 = nn.MaxPool2d(4,4)
        self.linear = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def _make_layer(self, block, in_planes, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        prepLayer = F.relu(self.bn1(self.conv1(x)))
        #layer 1
        X1 = F.relu(self.bn2(self.pool2(self.conv2(prepLayer))))
        R1 = self.res1(X1)
        layer1 = X1 + R1
        #layer 2
        layer2 = F.relu(self.bn3(self.pool3(self.conv3(layer1))))
        #layer 3
        X2 = F.relu(self.bn4(self.pool4(self.conv4(layer2))))
        R2 = self.res2(X2)
        layer3 = X2 + R2
        # #layer 4
        out = self.pool5(layer3)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.softmax(out)
        return out


def MyResNetS11():
    return ResNetS11(BasicBlock, [1, 1])

def test():
    net = MyResNetS11()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()