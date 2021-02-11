import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from Networks.FuzzyLayer import FuzzyLayer
from Networks.T2FuzzyLayer import T2FuzzyLayer
from Networks.Teachers import TeacherMNIST, TeacherCifar
import matplotlib.pyplot as plt
import enum
import numpy as np
from sklearn.preprocessing import StandardScaler



class StudentEncoder(nn.Module):
    def __init__(self, n_inputs, n_memberships, n_outputs, encoder, learnable_memberships=True, fuzzy_type=1, use_sigma_scale=True, use_height_scale=True):
        super(StudentEncoder, self).__init__()
        self.n_inputs = n_inputs
        self.n_memberships = n_memberships
        self.n_outputs = n_outputs
        self.encoder = encoder

        if fuzzy_type == 1:
            print("Fuzzy Type 1")
            self.fuzzy_layer = FuzzyLayer(n_memberships=n_memberships, n_inputs=n_inputs, n_outputs=n_outputs, learnable_memberships=learnable_memberships)
        else:
            print("Fuzzy Type 2")
            self.fuzzy_layer = T2FuzzyLayer(n_memberships=n_memberships, n_inputs=n_inputs, n_outputs=n_outputs, use_sigma_scale=use_sigma_scale, use_height_scale=use_height_scale)

    def initialize(self, init_data):
        encoded = self.encoder.encode(init_data)
        self.fuzzy_layer.activation_layer.initialize_gaussians(encoded, None)

    def forward(self, x):
        x = self.encoder.encode(x)
        return self.fuzzy_layer.forward(x)

class Student(nn.Module):
    def __init__(self, n_inputs, n_memberships, n_outputs, learnable_memberships=True, fuzzy_type=1, use_sigma_scale=True, use_height_scale=True):
        super(Student, self).__init__()
        self.n_inputs = n_inputs
        self.n_memberships = n_memberships
        self.n_outputs = n_outputs

        # PCA and Data params
        self.evecs = nn.Parameter(torch.empty((784, n_inputs)))  # TODO: Don't hardcode 784
        self.evecs.requires_grad = False
        self.trainMean = nn.Parameter(torch.empty((784, 1)))
        self.trainMean.requires_grad = False
        self.trainVar = nn.Parameter(torch.empty((784, 1)))
        self.trainMean.requires_grad = False
        self.trainMax = nn.Parameter()
        self.trainMax.requires_grad = False
        self.trainMin = nn.Parameter()
        self.trainMin.requires_grad = False

        if fuzzy_type == 1:
            print("Fuzzy Type 1")
            self.fuzzy_layer = FuzzyLayer(n_memberships=n_memberships, n_inputs=n_inputs, n_outputs=n_outputs, learnable_memberships=learnable_memberships)
        else:
            print("Fuzzy Type 2")
            self.fuzzy_layer = T2FuzzyLayer(n_memberships=n_memberships, n_inputs=n_inputs, n_outputs=n_outputs, use_sigma_scale=use_sigma_scale, use_height_scale=use_height_scale)

        # These values will be set at each pca fit
        self.pca = None
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

        fitted = self.fit_pca_numpy(init_data, init_labels)  # 1

        if load_params and filename is not None:
            self.fuzzy_layer.activation_layer.load_parameters(filename)
        else:
            self.fuzzy_layer.activation_layer.initialize_gaussians(fitted.detach().cpu().numpy(), init_labels, filename)

    def fit_pca(self, init_data:torch.Tensor, init_labels):

        # Standardize matrix
        flattened = torch.flatten(init_data, start_dim=1)
        data = flattened.T  # /(self.trainVar + torch.tensor(1e-20))
        data = data.T
        covmat = np.cov(data.T)  # Don't use all the data
        #Apply SVD
        left, sigma, right = torch.svd(torch.tensor(covmat).float())
        #evals, evecs = torch.eig(torch.tensor(covmat).float(), eigenvectors=True)
        self.evecs = nn.Parameter(right[:, :self.n_inputs])  # Set eigen vectors as parameter so that it can be loaded in future
        self.evecs.requires_grad = False
        # evecs are columns vectors. So the ith eig vector is evecs[:,i]
        reduced = self.feature_extraction(data)
        self.trainMean = nn.Parameter(reduced.mean())
        self.trainVar = nn.Parameter(reduced.var())
        self.trainMean.requires_grad = False
        self.trainVar.requires_grad = False
        return (reduced-self.trainMean)/self.trainVar

    def fit_pca_numpy(self, init_data, init_labels):
        flattened = torch.flatten(init_data, start_dim=1)
        self.pca = PCA(n_components=self.n_inputs)
        self.pca.fit(flattened)
        reduced = self.pca.transform(flattened)
        self.trainMax = nn.Parameter(torch.tensor(reduced.max()))
        self.trainMin = nn.Parameter(torch.tensor(reduced.min()))
        self.trainMean = nn.Parameter(torch.tensor(reduced.mean()))
        self.trainVar = nn.Parameter(torch.tensor(reduced.var()))
        self.trainMin.requires_grad = False
        self.trainMax.requires_grad = False
        self.trainMean.requires_grad = False
        self.trainVar.requires_grad = False
        reduced = torch.tensor(reduced).to(init_data.device).float()
        #reduced = (reduced-self.trainMin)/(self.trainMax-self.trainMin)
        reduced = (reduced - self.trainMean) / self.trainVar
        return reduced

    def preprocess_data(self, data):
        if self.trainMean is None or self.evecs is None:
            raise("Initialize the Network first")

        flat = torch.flatten(data, start_dim=1)
        flat = flat.T  # /(self.trainVar + torch.tensor(1e-20))
        reduced = torch.tensor(self.feature_extraction(flat.T)).to(data.device).float()
        #reduced = (reduced - self.trainMin) / (self.trainMax - self.trainMin)
        reduced = (reduced - self.trainMean)/self.trainVar
        return reduced

    def feature_extraction(self, data):
        return self.pca.transform(data.cpu())
        #return torch.matmul(data, self.evecs.to(data.device))

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

