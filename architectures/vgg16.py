import torch


def VGG16(download_dir='./'):
    torch.hub.set_dir(download_dir)
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True)
    return model