import torch
import torch.nn as nn
import torchvision.models as model


class Resnet(nn.Module):
    def __init__(self, class_count):
        super().__init__()
        self.resnet18 = model.resnet18(weights=model.ResNet18_Weights.DEFAULT)

        # Freeze parameter
        for param in self.resnet18.parameters():
            param.requires_grad = False

        self.resnet18.fc = torch.nn.Linear(self.resnet18.fc.in_features, class_count)

    def forward(self, image):
        out = self.resnet18(image).squeeze()
        return out
    