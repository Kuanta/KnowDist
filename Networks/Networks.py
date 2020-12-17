import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from Networks.FuzzyLayer import FuzzyLayer

class TeacherLite(nn.Module):
    def __init__(self, n_class):
        super(TeacherLite, self).__init__()
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
        x = self.fc2(x)
        return x

class Teacher(nn.Module):
    def __init__(self, n_class):
        super(Teacher, self).__init__()
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

class Student(nn.Module):
    def __init__(self, n_inputs, n_memberships, n_outputs, learnable_memberships=True):
        super(Student, self).__init__()
        self.n_inputs = n_inputs
        self.n_memberships = n_memberships
        self.n_outputs = n_outputs
        self.pca = PCA(64)
        self.fuzzy_layer = FuzzyLayer(n_memberships=n_memberships, n_inputs=64, n_outputs=10, learnable_memberships=learnable_memberships)

    def forward(self, x):
        device = x.device
        x = self.pca.transform(x.view(x.shape[0], -1).detach().cpu().numpy())
        x = self.fuzzy_layer.forward(torch.tensor(x).to(device).float())
        return x

    def initialize(self, init_data: torch.Tensor, init_labels: torch.Tensor):
        '''
        Initializes the network by applying PCA a
        :param init_data:
        :return:
        '''
        self.fit_pca(init_data, init_labels)
        flattened = init_data.view(init_data.shape[0], -1)
        fitted = self.pca.transform(flattened.numpy())
        # 2) Initialize the weights of the fuzzy layer using c-means
        print("Activating Fuzzy")
        self.fuzzy_layer.activation_layer.initialize_gaussians(fitted, init_labels)

    def fit_pca(self, init_data:torch.Tensor, init_labels: torch.Tensor):
        self.pca = PCA(64)
        flattened = init_data.view(init_data.shape[0], -1)
        self.pca.fit(flattened.detach().cpu().numpy())

class DistillNet(nn.Module):
    def __init__(self, student, teacher):
        super(DistillNet, self).__init__()
        self.student = student
        self.teacher = teacher

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, x):
        student_logits = self.student(x)
        teacher_logits = self.teacher(x)
        return [student_logits, teacher_logits]

