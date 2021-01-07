import torch
import torch.nn as nn
from Networks.FuzzyLayer import *

from enum import Enum
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import skfuzzy as fuzz
import json


class T2FuzzyLayer(nn.Module):
    """
       Constructor
       n_memberships (int): Number of membership functions
       n_inputs (int): Number of inputs
       n_outputs (int): Number of outputs
       mu (array): Can give a pre initialized set of membership centers
       sigma (array): Can give a pre initialized set of membership variances
       """

    # TODO: clustering variable is pointless at this point since creating  every possible is not possible at the moment
    def __init__(self, n_memberships, n_inputs, n_outputs=1, fou_height=0.5, clustering=True, use_gpu=True,
                 learnable_memberships=True, n_random_rules=0, t_norm=TNormType.Product):

        super(T2FuzzyLayer, self).__init__()
        # Define params
        self.n_memberships = n_memberships
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.clustering = clustering
        self.n_rules_generated_unique = torch.Tensor(0)  # This value isn't necessarly same with n_rules_generated
        # There can be duplicate entries in rule_masks when generating. To be able to load the saved model,
        # unique number of rules must be saved by setting it as a non-learnable parameter
        self.t_norm = t_norm
        self.fou_height = fou_height

        device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")

        self.upper_activation_layer = GaussianLayer(n_memberships=self.n_memberships, n_inputs=self.n_inputs, device=device,
                                              n_outputs=self.n_outputs, trainable=learnable_memberships, height=fou_height)

        self.lower_activation_layer = GaussianLayer(n_memberships=self.n_memberships, n_inputs=self.n_inputs, device=device,
                                              n_outputs=self.n_outputs, trainable=learnable_memberships, height=fou_height)

        pre_rule_masks = None  # If this is None, rule_layer will create rule masks (random+same indexed rules)
        self.rule_layer = FuzzyRules(n_memberships=self.n_memberships, n_inputs=self.n_inputs,
                                     clustering=self.clustering, n_random_rules=n_random_rules,
                                     rule_masks=pre_rule_masks)
        self.n_rules = self.rule_layer.n_rules

        # Initialize rule parameters (rho) randomly
        rho = torch.rand(size=(self.n_outputs, self.n_rules, self.n_inputs + 1))  # TODO: Search for alternative methods
        self.rho = torch.nn.Parameter(rho)
        self.rho.requires_grad = True

    def initialize_activation_layers(self, TrainData, TrainLabels=None, params_filepath=None):
        self.upper_activation_layer.initialize_gaussians(TrainData, TrainLabels, params_filepath)
        # Sigma and mean values of activation layers should be the same
        self.lower_activation_layer.sigma = self.upper_activation_layer.sigma
        self.lower_activation_layer.mu = self.lower_activation_layer

    def forward(self, x):
        batch_size = x.shape[0]
        if x.shape != (batch_size, self.n_inputs):
            raise Exception("Expected input shape of ", (1, self.n_inputs), " got ", x[0].shape)

        upper_membs = self.upper_activation_layer.forward(x)
        lower_membs = self.lower_activation_layer.forward(x)

        upper_firings = self.rule_layer(upper_membs, self.t_norm)
        lower_firings = self.rule_layer(lower_membs, self.t_norm)

        upper_conseqs = torch.matmul(x, self.rho[:, :, :-1].transpose(2, 1)).transpose(1, 0) + self.rho[:, :, self.n_inputs]
        upper_conseqs = upper_conseqs * (upper_firings / upper_firings.sum(1, True)).unsqueeze(1)
        upper_conseqs = upper_conseqs.sum(dim=2)

        lower_conseqs = torch.matmul(x, self.rho[:, :, :-1].transpose(2, 1)).transpose(1, 0) + self.rho[:, :, self.n_inputs]
        lower_conseqs = lower_conseqs * (lower_firings / lower_firings.sum(1, True)).unsqueeze(1)
        lower_conseqs = lower_conseqs.sum(dim=2)

        conseqs = 0.5*lower_conseqs + 0.5*upper_conseqs
        return conseqs

    def draw(self, input_index):
        """
        Draws the membership functions for the input with the given example
        :param input_index: Index of the input (row in membership parameters
        """
        self.activation_layer.draw(input_index)

    def save_model(self, name):
        torch.save(self.state_dict(), name)