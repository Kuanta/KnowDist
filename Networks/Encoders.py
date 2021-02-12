import torch
import torch.nn as nn
import torch.nn.functional as F
from DeepTorch.Datasets.Cifar import CifarLoader
import DeepTorch.Trainer as trn

class CifarEncoder(nn.Module):
    def __init__(self, n_dims):
        super(CifarEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(6,16, kernel_size=5),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16,6,kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6,3,kernel_size=5),
            nn.ReLU(True)
        )
        self.fc_1 = nn.Linear(in_features=16*24*24,out_features=512)
        self.fc_2 = nn.Linear(in_features=512, out_features=n_dims)
        self.fc_3 = nn.Linear(in_features=n_dims, out_features=512)
        self.fc_4 = nn.Linear(in_features=512, out_features=16*24*24)
    def encode(self, x):
        x = self.encoder(x)
        x = self.fc_1(x.view(x.shape[0], -1))
        x = self.fc_2(x)
        return x

    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        x = self.fc_1(x.view(x.shape[0], -1))
        x = F.relu(x)
        x = self.fc_2(x)

        # Decoder
        x = self.fc_3(x)
        x = F.relu(x)
        x = self.fc_4(x)
        x = self.decoder(x.view(-1,16,24,24))
        return x

def train_encoder():
    import os
    cLoader = CifarLoader(CIFAR_DIR=os.path.join("../", "cifar-10-batches-py", ""),validation_partition=1.0)
    train_set = cLoader.get_training_dataset()
    train_set.labels = train_set.data
    train_set.labels = train_set.labels.transpose(0, 3, 1, 2)
    train_opts = trn.TrainingOptions()
    train_opts.optimizer_type = trn.OptimizerType.Adam
    train_opts.learning_rate = 0.001
    train_opts.learning_rate_update_by_step = False  # Update schedular at epochs
    train_opts.learning_rate_drop_factor = 0.5
    train_opts.learning_rate_drop_type = trn.SchedulerType.StepLr
    train_opts.learning_rate_drop_step_count = 2
    train_opts.batch_size = 64
    train_opts.weight_decay = 1e-5
    train_opts.n_epochs = 6
    train_opts.use_gpu = True
    train_opts.save_model = True
    train_opts.saved_model_name = "models/encoders/cifar_encoder"
    train_opts.regularization_method = None

    net = CifarEncoder(100)
    loss_fnc = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), weight_decay=1e-5)
    num_epochs = 8
    batch_size = 100
    num_iters = int(train_set.get_element_count()/batch_size)
    for e in range(num_epochs):
        train_set.shuffle_data()
        for iter in range(num_iters):
            batch_data, batch_labels = train_set.get_batch(batch_size)
            out = net.forward(batch_data)
            loss = loss_fnc(out, batch_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iter % 200 == 0:
                print('epoch [{}/{}], iter [{}/{}], loss:{:.4f}'.format(e + 1, num_epochs, iter+1, num_iters, loss.item()))
    torch.save(net.state_dict(), "cifar_encoder")

if __name__ == "__main__":
    train_encoder()
