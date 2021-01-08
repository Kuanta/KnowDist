import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from Networks.FuzzyLayer import FuzzyLayer
from Networks.T2FuzzyLayer import T2FuzzyLayer
import matplotlib.pyplot as plt
import enum
import numpy as np
from sklearn.preprocessing import StandardScaler

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
        x = F.relu(x)
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
    def __init__(self, n_inputs, n_memberships, n_outputs, learnable_memberships=True, fuzzy_type=1):
        super(Student, self).__init__()
        self.n_inputs = n_inputs
        self.n_memberships = n_memberships
        self.n_outputs = n_outputs

        # PCA and Data params
        self.evecs = nn.Parameter(torch.empty((784, n_inputs)))  # TODO: Don't hardcode 784
        self.evecs.requires_grad = False
        self.trainMean = nn.Parameter(torch.empty((784, 1)))
        self.trainMean.requires_grad = False
        self.trainVar = nn.Parameter(torch.empty((784)))
        self.trainMean.requires_grad = False

        if fuzzy_type == 1:
            self.fuzzy_layer = FuzzyLayer(n_memberships=n_memberships, n_inputs=n_inputs, n_outputs=10, learnable_memberships=learnable_memberships)
        else:
            self.fuzzy_layer = T2FuzzyLayer(n_memberships=n_memberships, n_inputs=n_inputs, n_outputs=n_outputs)

        # These values will be set at each pca fit
        self.data_min = 0
        self.data_max = 0

    def forward(self, x):
        x = self.fuzzy_layer.forward(self.preprocess_data(x))
        return x

    def initialize(self, init_data: torch.Tensor, init_labels, load_params=True, filename=None):
        '''
        Initializes the network.
        1) Fits the PCA
        2) Initializes the fuzzy layer
        :param init_data (torch.Tensor): Complete training data (or large as possible).
        :param load_params (boolean): If set to true and the specified filepath is not none, activation parameters will
        be load from the file
        :param filename: If load_params is false and filepath is not none, calculated activation parameters will be saved
        to the file with the given name
        :return:
        '''

        fitted = self.fit_pca(init_data, init_labels)  # 1
        self.data_max = fitted.max()
        self.data_min = fitted.min()

        if load_params and filename is not None:
            self.fuzzy_layer.activation_layer.load_parameters(filename)
        else:
            self.fuzzy_layer.activation_layer.initialize_gaussians(fitted, init_labels, filename)

    def fit_pca(self, init_data:torch.Tensor, init_labels):

        # Standardize matrix
        flattened = torch.flatten(init_data, start_dim=1)
        self.trainMean = nn.Parameter(flattened.T.mean(1, True))
        self.trainVar = nn.Parameter(flattened.T.var(1, True))
        self.trainMean.requires_grad = False


        data = flattened.T - self.trainMean
        data = data.T
        covmat = np.cov(data.T)  # Don't use all the data
        evals, evecs = torch.eig(torch.tensor(covmat).float(), eigenvectors=True)
        self.evecs = nn.Parameter(evecs[:, :self.n_inputs])  # Set eigen vectors as parameter so that it can be loaded in future
        self.evecs.requires_grad = False
        # evecs are columns vectors. So the ith eig vector is evecs[:,i]
        reduced = self.feature_extraction(data)
        return reduced

    def preprocess_data(self, data):
        if self.trainMean is None or self.evecs is None:
            raise("Initialize the Network first")

        flat = torch.flatten(data, start_dim=1)
        flat = flat.T - self.trainMean.to(data.device)
        reduced = self.feature_extraction(flat.T)
        return reduced

    def feature_extraction(self, data):
        return torch.matmul(data, self.evecs.to(data.device))

    def min_max_normalization(self, data):
        return (data-self.data_min)/(self.data_max-self.data_min)

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

