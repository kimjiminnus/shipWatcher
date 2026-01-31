import torch
import torch.nn as nn
import torchvision.models as models


# shipWatcher model architecture
# Initialise ResNet-18 with pre-trained weights as a feature extractor
# Modify full connected(fc) layer with nn.Linear for 3-class classification

def get_shipWatcher():
    model = models.resnet18(weights="DEFAULT")
    for param in model.parameters():
        param.requires_grad = False

    # Change fc layer
    input_size = model.fc.in_features
    model.fc = nn.Linear(input_size,3)
    return model
