from turtle import forward
from numpy import pad
import torch
import torchvision.models as models 
import torch.nn as nn
from torchsummary import summary


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self,x):
        return x

class MyModel(nn.Module):
    def __init__(self, num_class):
        super(MyModel,self).__init__()

        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.fc = Identity()
        self.fc = nn.Sequential(nn.Linear(512, 64), nn.ReLU(), nn.Linear(64, num_class))

    def forward(self, x):
        x = self.resnet18(x)
        x = self.fc(x)
        return x 


class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = self.CnnBlock(3, 16)
        self.conv2 = self.CnnBlock(16, 32)
        self.conv3 = self.CnnBlock(32, 16)
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride = 2, padding=0)
        self.conv4 = self.CnnBlock(16, 16)
        self.conv5 = self.CnnBlock(16, 8)

        self.fc = nn.Linear(1152, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropuot = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)  
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = self.dropuot(x)
        return self.fc2(x)

    def CnnBlock(self, input_dim, output_dim):
        conv = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3),
            nn.BatchNorm2d(output_dim),
            nn.ReLU()
        )
        return conv




