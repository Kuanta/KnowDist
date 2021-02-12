import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def create_teacher(dataset_id):
    '''
    Creates the appropriate teacher network from the dataset_id
    :param dataset_id: Id for the dataset. (MNIST:1, Cifar:2)
    :return (nn.module): Corresponding teacher model
    '''
    if dataset_id == 3:
        print("Getting Resnet50")
        model = models.resnet50(pretrained=True, progress=True)
        # for param in model.parameters():
        #     param.requires_grad = False

        model.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10))
        return model

    elif dataset_id == 2:
        print("Fashion MNIST")
        return TeacherFashionMNIST(10)
    else:
        return TeacherMNIST(10)

class TeacherLiteMNIST(nn.Module):
    def __init__(self, n_class):
        super(TeacherLiteMNIST, self).__init__()
        self.dropout_prob = 0.1
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(256, 32)
        self.fc2 = nn.Linear(32, n_class)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class TeacherMNIST(nn.Module):
    def __init__(self, n_class):
        super(TeacherMNIST, self).__init__()
        self.dropout_prob = 0.1
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_prob),  # TODO: What should be the dropout prob.?
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_prob),
            nn.AvgPool2d(kernel_size=3)
        )
        self.fc1 = nn.Linear(2304, 512)
        self.bn = nn.BatchNorm1d(512)
        self.do = nn.Dropout2d(self.dropout_prob)
        self.fc2 = nn.Linear(512, n_class)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.bn(x)
        x = self.do(x)
        x = self.fc2(x)
        return x

class TeacherFashionMNIST(nn.Module):
    '''
    Fashion MNIST has 28x28 images with 10 classes
    '''
    def __init__(self, n_class):
        super(TeacherFashionMNIST, self).__init__()
        self.dropout_prob = 0.1
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_prob),  # TODO: What should be the dropout prob.?
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_prob),
            nn.AvgPool2d(kernel_size=3)
        )
        self.fc1 = nn.Linear(2304, 512)
        self.bn = nn.BatchNorm1d(512)
        self.do = nn.Dropout2d(self.dropout_prob)
        self.fc2 = nn.Linear(512, n_class)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.bn(x)
        x = self.do(x)
        x = self.fc2(x)
        return x


class TeacherResNet50(nn.Module):
    def __init__(self, n_class):
        super(TeacherResNet50, self).__init__()
        self.model = models.resnet50(pretrained=True, progress=True)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_class))

    def forward(self, x):
        return self.model(x)

class TeacherCifar(nn.Module):
    def __init__(self, n_class):
        super(TeacherCifar, self).__init__()
        self.num_classes = n_class

        self.stack_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=2),
            nn.ReLU()
        )
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.stack_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2),
            nn.ReLU()
        )
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.stack_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2),
            nn.ReLU()
        )
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear(in_features=8192, out_features=512)
        self.bn1 = nn.BatchNorm1d(num_features=512)
        self.fc2 = nn.Linear(512,n_class)

    def forward(self, x):
        x = self.stack_1(x)
        x = self.max_pool1(x)
        x = self.stack_2(x)
        x = self.max_pool2(x)
        x = self.stack_3(x)
        x = self.max_pool3(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        return x

