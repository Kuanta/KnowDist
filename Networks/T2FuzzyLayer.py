from Networks.FuzzyLayer import *

class T2FuzzyLayer(nn.Module):

    # TODO: clustering variable is pointless at this point since creating  every possible is not possible at the moment
    def __init__(self, n_memberships, n_inputs, n_outputs=1, clustering=True, use_gpu=True,
                 learnable_memberships=True, n_random_rules=0, t_norm=TNormType.Product, use_height_scale=True, use_sigma_scale=True):
        '''


        :param n_memberships: Number of membership functions
        :param n_inputs: Number of inputs
        :param n_outputs: Number of outputs
        :param clustering: If set to true, rule number will be same as the membership value. Don't set to true for large inputs
        :param use_gpu: True if use gpu. Not essential since a model can be transferred to gpu with torch functions
        :param learnable_memberships: Set to true for learnable memberships (sigma and mu)
        :param n_random_rules: Set to true to generate random rules
        :param t_norm: Type of the t_norm operation. (Min or product)
        :param use_height_scale: If set to true, lower membership function will be scaled
        :param use_sigma_scale: If set to true, lower membership function's sigma will be scaled
        '''
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
        self.use_sigma_scale = use_sigma_scale
        self.use_height_scale = use_height_scale

        # Learnable Parameters
        self.fou_height = nn.Parameter(torch.rand((self.n_memberships, self.n_inputs), dtype=torch.float))
        self.fou_height.requires_grad = True
        self.alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float))
        self.alpha.requires_grad = True
        self.sigma_scale = nn.Parameter(torch.rand((self.n_memberships, self.n_inputs), dtype=torch.float))
        self.sigma_scale.requires_grad = True

        device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")

        self.activation_layer = GaussianLayer(n_memberships=self.n_memberships, n_inputs=self.n_inputs, device=device,
                                              n_outputs=self.n_outputs, trainable=learnable_memberships)


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
        self.activation_layer.initialize_gaussians(TrainData, TrainLabels, params_filepath)

    def forward(self, x):
        # Check if the input has the correct size
        batch_size = x.shape[0]
        if x.shape != (batch_size, self.n_inputs):
            raise Exception("Expected input shape of ", (1, self.n_inputs), " got ", x[0].shape)

        upper_membs = self.activation_layer.forward(x)
        if self.use_sigma_scale:
            sigma_scale = torch.sigmoid(self.sigma_scale)*0.5+0.5  # TODO: Sigmoid may 0 some sigmoids which causes null losses. Can try by multiplying with 0.5 and summin 0.5
        else:
            sigma_scale = None

        # if self.use_height_scale:
        #     lower_membs = self.activation_layer.forward(x, sigma_scale=sigma_scale)*torch.sigmoid(self.fou_height)
        # else:
        #     lower_membs = self.activation_layer.forward(x, sigma_scale=sigma_scale)

        lower_membs = upper_membs*torch.sigmoid(self.fou_height)
        upper_firings = self.rule_layer(upper_membs, self.t_norm)
        lower_firings = self.rule_layer(lower_membs, self.t_norm)

        conseqs = torch.matmul(x, self.rho[:, :, :-1].transpose(2, 1)).transpose(1, 0) + self.rho[:, :, self.n_inputs]
        upper_out = (conseqs * (upper_firings / upper_firings.sum(1, True)+torch.tensor(1e-20)).unsqueeze(1)).sum(dim=2)
        lower_out = (conseqs * (lower_firings / lower_firings.sum(1, True)+torch.tensor(1e-20)).unsqueeze(1)).sum(dim=2)
        out = (1-self.alpha)*upper_out + self.alpha*lower_out  #TODO: Set this
        return out

    def draw(self, input_index):
        """
        Draws the membership functions for the input with the given example
        :param input_index: Index of the input (row in membership parameters
        """
        self.activation_layer.draw(input_index)

    def save_model(self, name):
        torch.save(self.state_dict(), name)