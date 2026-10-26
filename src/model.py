from torchvision.models import resnet18

def make_model(num_classes=10):
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = __import__("torch").nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = __import__("torch").nn.Identity()
    return model

