import torch
import torch.nn as nn
import torchvision.models as model


class Resnet(nn.Module):
    def __init__(self, class_count):
        super().__init__()
        self.resnet18 = model.resnet18(pretrained=True)
        self.resnet18 = torch.nn.Sequential(*(list(self.resnet18.children())[:-1]))
        self.classifier = torch.nn.Linear(1024, class_count)

    def forward(self, image):
        resnet_pred = self.resnet18(image).squeeze()
        out = self.classifier(resnet_pred)
        return out
    