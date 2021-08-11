import torch.nn as nn
from torchvision import models


def resnet(args):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model


def vggnet(args):
    model = models.vgg19_bn(pretrained=False, num_classes=2)
    # model = convert_relu_to_ELU(model)
    # model = remove_dropout(model)
    return model

def niggernet(args):
    model = nn.Sequential(
        nn.Conv2d(1,20,5),
        nn.ReLU(),
        nn.Conv2d(20,64,5),
        nn.ReLU()
        )
    #model = remove_dropout(model)
    return model


def googlenet(args):
    model = models.googlenet(pretrained=False, num_classes=2)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model


def alexnet(*args):
    model = models.alexnet(pretrained=False, num_classes=2)
    return model


def convert_relu_to_ELU(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.ELU())
        else:
            convert_relu_to_ELU(child)
    return model


def remove_dropout(model):
    model.classifier = nn.Sequential(
        *list(model.classifier.children())[:-6], nn.Sigmoid(), nn.Linear(4096, 2)
    )
    return model