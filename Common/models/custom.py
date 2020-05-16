import torch.nn as nn
from gbn import *

dropout_value = 0.1
num_splits = 2 
class CustomModel(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # CONVOLUTION BLOCK 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False), #RF = 3
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),

            # Dilated Convolution
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=2, bias=False, dilation=2), #RF = 7
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False), #RF = 9 
            nn.ReLU(),
            GhostBatchNorm(128, num_splits),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), padding=0, bias=False), #RF = 9
            nn.ReLU()

        ) # output_size = 32

        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 16 RF = 10

        # CONVOLUTION BLOCK 2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False), #RF = 14
            nn.ReLU(),
            # GhostBatchNorm(64, num_splits),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),

            # Depthwise separable convolution
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False, groups=64), #RF = 18
            nn.ReLU(),
            # GhostBatchNorm(64, num_splits), 
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), padding=0, bias=False), #RF = 18
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), padding=0, bias=False), #RF = 18
            nn.ReLU()
        ) # output_size = 16
        

        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 8 RF = 20

        # CONVOLUTION BLOCK 3
        self.convblock3 = nn.Sequential( 
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=2, bias=False), #RF = 28
            nn.ReLU(),
            # GhostBatchNorm(64, num_splits),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False), #RF = 36
            nn.ReLU(),
            # GhostBatchNorm(128, num_splits),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False), #RF = 44
            nn.ReLU(),
            # GhostBatchNorm(256, num_splits),
            nn.BatchNorm2d(256),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=256, out_channels=32, kernel_size=(1, 1), padding=0, bias=False), #RF = 44
            nn.ReLU()
        ) # output_size = 10

        self.pool3 = nn.MaxPool2d(2, 2) # output_size = 5 RF = 48

        # CONVOLUTION BLOCK 4
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=2, bias=False), #RF = 64
            nn.ReLU(),
            # GhostBatchNorm(64, num_splits),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False), #RF = 80
            nn.ReLU(),
            # GhostBatchNorm(128, num_splits),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False), #RF = 96
            nn.ReLU(),
            # GhostBatchNorm(256, num_splits),
            nn.BatchNorm2d(256),
            nn.Dropout(dropout_value),

            # nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), padding=0, bias=False), 
            # nn.ReLU(),
            # nn.Dropout(dropout_value),

        ) # output_size = 7


        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=7)
        ) # output_size = 1 RF = 120
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=10, kernel_size=(1, 1), padding=0, bias=False), #RF = 
        ) # output_size = 1 RF = 120

        # self.fc1 = nn.Linear(1, 10) # output_size = 1



    def forward(self, x):
        x = self.convblock1(x)
        x = self.pool1(x)
        x = self.convblock2(x)
        x = self.pool2(x)
        x = self.convblock3(x)
        x = self.pool3(x)
        x = self.convblock4(x)
        x = self.gap(x)
        x = self.convblock5(x)
        # x = self.fc1(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)