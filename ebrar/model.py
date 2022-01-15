import torchvision.models as models 
import torch.nn as nn


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




