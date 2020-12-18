import torch
import torch.nn as nn
from enum import Enum
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import skfuzzy as fuzz



# TODO: Add support for other membership functions

class TNormType(Enum):
    """
    Enumeration class for t-norm operator that will be used when calculating rule firings
    """
    Min = 0
    Product = 1


class ActivationLayers(Enum):
    Gaussian = 0

class SinusoidalLayer(nn.Module):
    def __init__(self, n_memberships, n_inputs, device='cuda:0'):
        super(SinusoidalLayer, self).__init__()
        self.n_memberships = n_memberships
        self.n_inputs = n_inputs

        # Init sinusodials
        self.frequencies = nn.Parameter(torch.rand((self.n_memberships, self.n_inputs))).to(device)
        self.biases = nn.Parameter(torch.rand((self.n_memberships, n_inputs))).to(device)

    def forward(self, x):
        return torch.sin(x.unsqueeze(1) * self.frequencies + self.biases) * 0.5 + 0.5  # Range of sin is [-1,1]


class GaussianLayer(nn.Module):
    """
    Defines a layer for gaussian membership functions
    """

    def __init__(self, n_memberships, n_inputs, device, n_outputs=1, trainable=False):
        super(GaussianLayer, self).__init__()
        self.n_memberships = n_memberships
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.trainable = trainable
        self.device = device
        self.initialize_gaussians()

    def forward(self, x):
        output = [self.calculate_memberships(batch, self.mu, self.sigma) for batch in x]
        return torch.stack(output)

    def calculate_memberships(self, batch, mu, sigma):
        batch = batch.float() - mu.float()
        batch = torch.div(batch, sigma)
        batch = torch.mul(batch, batch)
        batch = batch / 2
        batch = torch.exp(-1 * batch)
        return batch

    def initialize_gaussians(self, TrainData=None, TrainLabels=None):
        if TrainData is not None and TrainLabels is not None:
            self.update_membs_with_fcm(TrainData)
        else:
            self.sigma = torch.rand(size=(self.n_memberships, self.n_inputs), dtype=torch.double).to(self.device)
            self.mu = torch.rand(size=(self.n_memberships, self.n_inputs), dtype=torch.double).to(self.device)
            self.sigma = nn.Parameter(self.sigma.float())
            self.mu = nn.Parameter(self.mu.float())

            if self.trainable:
                self.sigma.requires_grad = True
                self.mu.requires_grad = True
            else:
                self.sigma.requires_grad = False
                self.mu.requires_grad = False

    def update_membs_with_fcm(self, TrainData):
        centers, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            TrainData.transpose(), self.n_memberships, 1.8, error=0.001, maxiter=500, init=None, seed=42)
        data = TrainData.transpose(1, 0)
        centers2 = []
        for i in range(data.shape[0]):
            center, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                np.expand_dims(data[i], 0), self.n_memberships, 1.8, error=0.001, maxiter=500,
                init=None, seed=42)
            centers2.append(center)

        # Calculate standard deviations
        diffs = np.expand_dims(TrainData, 1) - np.expand_dims(centers, 0)
        squared = -1*np.square(diffs)
        membs = np.expand_dims(2 * np.log(u.transpose()), 2)
        logs = np.sqrt(squared / membs)
        N = logs.shape[0]
        logs = np.sum(squared / membs, 0, keepdims=True)
        logs = logs / N

        torch.rand(())
        self.mu = nn.Parameter(torch.tensor(centers).float().to(self.device))
        #self.sigma = nn.Parameter(torch.tensor(logs.squeeze(0)).float().to(self.device))
        sigma = torch.rand(size=(self.n_memberships, self.n_inputs)).float().to(self.device)*0.5
        self.sigma = nn.Parameter(sigma)
        if self.trainable:
            self.sigma.requires_grad = True
            self.mu.requires_grad = True
        else:
            self.sigma.requires_grad = False
            self.mu.requires_grad = False


    def update_membs_with_kmeans(self, TrainData=None, TrainLabels=None):
        if TrainData is not None and TrainLabels is not None:
            # KMeans Based Init #  TODO: Below code is from Derek Anderson, remove it after usage
            R = self.n_memberships  # our number of rules
            A = self.n_inputs  # our number of antecedents

            # run the k-means clustering algorithm
            kmeans = KMeans(n_clusters=R, n_init=1, init='k-means++', tol=1e-6, max_iter=500, random_state=0).fit(
                TrainData)
            # steal the cluster centers for our ant's
            mu = torch.rand(R, A)
            for i in range(R):
                for j in range(A):
                    mu[i, j] = float(kmeans.cluster_centers_[i, j])
            self.mu = torch.nn.Parameter(mu)

            # now, estimate the variances
            sig = torch.ones((R, A))
            # for r in range(R):
            #     inds = np.where(kmeans.labels_ == r)
            #     classdata = torch.squeeze(TrainData[inds, :])
            #
            #     for d in range(A):
            #         # FIXME: What to do when a row in classdata has single element (How to calculate std?)
            #         # Sometimes class data returns a 1 dimensional array, convert it to a shape of (1, n_inputs)
            #         if len(classdata.shape) == 1:
            #             sig[r, d] = 1
            #         else:
            #             sig[r, d] = torch.std(torch.squeeze(classdata[:, d]))
            # # print('K-means sigma guess')
            self.sigma = torch.nn.Parameter(sig)

    def draw(self, input_index):
        """
        Draws the membership functions for the input with the given example
        :param input_index: Index of the input (row in membership parameters
        """
        mu_values = self.mu.cpu().data.numpy()
        mu_values = np.transpose(mu_values, (1, 0))
        sigma_values = self.sigma.cpu().data.numpy()
        sigma_values = np.transpose(sigma_values, (1, 0))
        x = np.linspace(-5, 5, 1000)
        for i in range(mu_values[input_index].size):
            mu_value = mu_values[input_index][i]
            sigma_value = sigma_values[input_index][i]
            y = np.exp(-(x - mu_value) * (x - mu_value) / (2 * sigma_value * sigma_value))
            plt.plot(x, y)
        plt.show()


class FuzzyRules(nn.Module):
    """
    Defines a layer for calculating rule firings and rule outputs
    """

    def __init__(self, n_memberships, n_inputs, clustering, n_random_rules=0, rule_masks=None):
        super(FuzzyRules, self).__init__()
        self.rule_masks = None
        self.n_random_rules = n_random_rules
        self.n_rules = 0
        # TODO: What if we insert random rules and ensure uniqueness?
        """
        Example code for unique list of lists:
        testdata = [[0, 1, 2], [0, 1, 2], [2, 2, 0], [1, 0, 2]]
        unique_data = [list(x) for x in set(tuple(x) for x in testdata)]
        print(unique_data)
        """
        # Create Rule Masks
        if rule_masks is not None:
            # Don't create if its already given
            self.rule_masks = rule_masks
            self.n_rules = rule_masks.shape[0]
        else:
            if clustering:
                self.rule_masks = [torch.tensor([i] * n_inputs) for i in range(n_memberships)]
                while len(self.rule_masks) < n_memberships + n_random_rules:
                    random_rule = np.random.randint(0, n_memberships - 1, n_inputs).tolist()
                    self.rule_masks.append(random_rule)
                    self.rule_masks = [list(x) for x in set(tuple(x) for x in self.rule_masks)]

                self.n_rules = n_memberships + n_random_rules
                self.rule_masks = torch.stack(self.rule_masks)
            else:
                self.n_rules = n_memberships ** n_inputs
                _rule_masks = itertools.product(*[range(n) for n in [n_memberships] * n_inputs])
                self.rule_masks = list(_rule_masks)

        # Make rule masks a non learnable parameter so when loading model, these indices can also be loaded
        if type(self.rule_masks) == torch.Tensor:
            # FIXME: When rules aren't generated and clustering is false, rule_masks variable is a list and it can't be set as parameter
            self.rule_masks = nn.Parameter(self.rule_masks.float())
            self.rule_masks.requires_grad = False

    def forward(self, membership_matrices, t_norm):

        rule_indices = self.rule_masks.expand(membership_matrices.shape[0], -1, -1)
        antecedents = torch.gather(membership_matrices, 1, rule_indices.long().to(membership_matrices.device))

        if t_norm == TNormType.Min:
            min_values, _ = torch.min(antecedents,
                                      dim=2)  # FIXME: Using product as T-norm causes underflow, have to use min
            return min_values
        elif t_norm == TNormType.Product:
            return antecedents.prod(dim=2)


class FuzzyLayer(nn.Module):
    """
    Constructor
    n_memberships (int): Number of membership functions
    n_inputs (int): Number of inputs
    n_outputs (int): Number of outputs
    mu (array): Can give a pre initialized set of membership centers
    sigma (array): Can give a pre initialized set of membership variances
    """

    # TODO: clustering variable is pointless at this point since creating  every possible is not possible at the moment
    def __init__(self, n_memberships, n_inputs, n_outputs=1, clustering=True, use_gpu=True,
                 learnable_memberships=True, n_random_rules=0, TrainData=None, TrainLabels=None, rule_generation=False,
                 n_rules_generated=-1, t_norm=TNormType.Product):
        super(FuzzyLayer, self).__init__()
        # Define params
        self.n_memberships = n_memberships
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.clustering = clustering
        self.n_rules_generated_unique = torch.Tensor(0)  # This value isn't necessarly same with n_rules_generated
        # There can be duplicate entries in rule_masks when generating. To be able to load the saved model,
        # unique number of rules must be saved by setting it as a non-learnable parameter
        self.t_norm = t_norm

        device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")

        self.activation_layer = GaussianLayer(n_memberships=self.n_memberships, n_inputs=self.n_inputs, device=device,
                                             n_outputs=self.n_outputs, trainable=learnable_memberships)

        pre_rule_masks = None  # If this is None, rule_layer will create rule masks (random+same indexed rules)
        if rule_generation:
            pre_rule_masks = self.generate_rule_masks(train_data=TrainData, rule_count=n_rules_generated).to(device)
        self.rule_layer = FuzzyRules(n_memberships=self.n_memberships, n_inputs=self.n_inputs,
                                     clustering=self.clustering, n_random_rules=n_random_rules,
                                     rule_masks=pre_rule_masks)
        # Set rule masks as parameter
        self.n_rules = self.rule_layer.n_rules

        # Initialize rule parameters (rho) randomly
        rho = torch.rand(size=(self.n_outputs, self.n_rules, self.n_inputs + 1))  # TODO: Search for alternative methods
        self.rho = torch.nn.Parameter(rho)
        self.rho.requires_grad = True

    def generate_rule_masks(self, train_data, rule_count):
        # FIXME: There could be duplicates in rule_masks
        """
        If a valid training data is given, generates rule indices. For a sample, it finds the maximum membership values
        for each input. Indices of the maximum membership for each input creates a rule. Than, firing rates of
        these rules calculated and the ones that gives the highest firing rates are returned
        @param train_data: Training data to generate rules from. Must have a shape of (n_batch, n_inputs)
        @param rule_count:
        @return:
        """
        if train_data is None:
            # This is probably for initialization. A fuzzy layer with proper rule_masks shape must be created to be able to
            # load state dictionary from a previous training
            return torch.ones(size=(rule_count, self.n_inputs))

        else:
            if rule_count > train_data.shape[0]:
                rule_count = train_data.shape[0]
                print(
                    "\033[93mNumber of rules to generate is bigger than total samples, setting the number of rules to maximum possible value\033[0m")
            # shape of memb_values is: (n_batch, n_membership, n_input)
            memb_values = self.gaussian_layer.forward(train_data, self.mu, self.sigma)

            # To find the max memb value of each input, a transpose is needed
            max_values, indices = torch.max(memb_values.transpose(2, 1), dim=2)

            # Find degrees of rules

            degrees = torch.min(max_values, dim=1)[0].view(-1,
                                                           1)  # Warning Used min as t-norm operator, might change in future
            # Sort by max_values
            concated = torch.cat((degrees.float(), indices.float()), dim=1)
            _, sorted_indices = torch.sort(concated[:, 0], descending=True)

            if rule_count > 0:
                rule_masks = indices[sorted_indices][0:rule_count, :]
            else:
                rule_masks = indices[sorted_indices]

            return rule_masks

    def forward(self, x):
        # Apply Min-Max Normalization
        batch_size = x.shape[0]
        if x.shape != (batch_size, self.n_inputs):
            raise Exception("Expected input shape of ", (1, self.n_inputs), " got ", x[0].shape)

        # membership_values = self.gaussian_layer.forward(x, self.mu, self.sigma)
        membership_values = self.activation_layer.forward(x)
        firings = self.rule_layer(membership_values, self.t_norm)  # Firings for each rule
        # firings = members, reduction="batchmean"hip_values.prod(dim=2)  # This is a quick way for clustered rules
        summed_firings = torch.sum(firings, dim=1).reshape(batch_size, 1) + 0.0000001
        firings = torch.div(firings, summed_firings)  # + self.firing_biases
        output = []

        # for batch_index in range(batch_size):
        #     # An example in the batch is a 1d tensor. In order to be able to multiply it with rule_params in an
        #     # element-wise fashion, needs to be expanded to a 3d tensor
        #     batch = x[batch_index].expand(self.rule_params.shape[0], self.rule_params.shape[1], -1)
        #     batch_output = torch.sum(batch*self.rule_params + self.rule_biases, dim=2).double()
        #     batch_firings = firings[batch_index].expand(self.rule_params.shape[0], -1)
        #     fired_batch_output = batch_output.double()*batch_firings.double()
        #
        #     # Defuzzify
        #     defuzzified = torch.sum(fired_batch_output, dim=1) / self.n_rules
        #     output.append(defuzzified)

        for batch_index in range(batch_size):
            # An example in the batch is a 1d tensor. In order to be able to multiply it with rule_params in an
            # element-wise fashion, needs to be expanded to a 3d tensor
            sample = x[batch_index]

            rule_outputs = torch.matmul(sample, self.rho[:, :, :-1].transpose(2, 1)) + self.rho[:, :,
                                                                                       self.n_inputs]  # rho has a shape of (n_outputs, n_rules, n_inputs)
            batch_firings = firings[batch_index].expand(self.n_outputs, -1)  # Expand it for number of outputs
            fired_batch_output = rule_outputs * batch_firings

            # Defuzzify
            defuzzified = torch.sum(fired_batch_output, dim=1)
            output.append(defuzzified)

        return torch.stack(output)

    def draw(self, input_index):
        """
        Draws the membership functions for the input with the given example
        :param input_index: Index of the input (row in membership parameters
        """
        self.activation_layer.draw(input_index)

    def save_model(self, name):
        torch.save(self.state_dict(), name)