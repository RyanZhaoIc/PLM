import torch.nn as nn
import torch.nn.functional as F
from models import ResNet18, ResNet34, ResNet50_Pretrained

class LeNet(nn.Module):
    def __init__(self, output_dim):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc = nn.Linear(84, output_dim)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, (2, 2))
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, (2, 2))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc(out)
        return out


def get_model(model_name, num_classes, fix_backbone=False):
    if model_name == 'lenet':
        model = LeNet(output_dim=num_classes)
    elif model_name == 'resnet':
        model = ResNet18(num_classes=num_classes)
    elif model_name == 'resnet34':
        model = ResNet34(num_classes=num_classes)
    elif model_name == 'resnet50':
        model = ResNet50_Pretrained(num_classes=num_classes, fix_backbone=fix_backbone)
    else:
        raise ValueError('Invalid models')
    return model
