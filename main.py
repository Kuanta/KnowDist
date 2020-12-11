'''
Implementation of knowledge distillation paper:
"Distilling a Deep Neural Network into a Takagi-Sugeno-Kang Fuzzy Inference
System"
'''

import torch
import DeepTorch.Trainer as trn
from DeepTorch.Datasets.Cifar import CifarDataset, CifarLoader
from DeepTorch.Datasets.MNIST import MNISTLoader, MNISTDataset
from Networks.Networks import Student, Teacher, DistillNet
from distillation import DistillationLoss
import numpy as np

# Flags
TRAIN = True
TRAIN_TEACHER = False
TEST = False
DATASET = "Cifar"

# Constants
TEACHER_MODEL_PATH = "./models/teacher"

# Func defs
def validate_distillation(data, labels):
    outputs = data[0].cpu().data.numpy()
    outputs = [np.argmax(pred) for pred in outputs]
    labels = labels.cpu().data.numpy()
    corrects = [outputs[i] == labels[i] for i in range(len(outputs))]
    return sum(corrects) / len(corrects) * 100

# Load dataset
if DATASET == "Cifar":
    cLoader = CifarLoader(validation_partition=0.7)
    train_set = cLoader.get_training_dataset()
    val_set = cLoader.get_validation_dataset()
    test_set = cLoader.get_test_dataset()
else:
    mLoader = MNISTLoader("./data")
    train_set = mLoader.get_training_dataset()
    val_set = None
    test_set = mLoader.get_test_dataset()

# Define networks
student = Student(n_memberships=7, n_inputs=64, n_outputs=10)
teacher = Teacher(10)

# Load teacher
#teacher.load_state_dict(torch.load(TEACHER_MODEL_PATH))
if TRAIN_TEACHER:
    train_opts = trn.TrainingOptions()
    train_opts.optimizer_type = trn.OptimizerType.Adam
    train_opts.learning_rate = 0.001
    train_opts.batch_size = 64
    train_opts.n_epochs = 50
    train_opts.use_gpu = True
    train_opts.save_model = True
    train_opts.saved_model_name = "models/teacher"

    trainer = trn.Trainer(teacher, train_opts)
    trainer.train(torch.nn.CrossEntropyLoss(), train_set, val_set, is_classification=True)
else:
    teacher.load_state_dict(torch.load("models/teacher"))
if TRAIN:
    # Define training options
    train_opts = trn.TrainingOptions()
    train_opts = trn.TrainingOptions()
    train_opts.optimizer_type = trn.OptimizerType.Adam
    train_opts.learning_rate = 0.001
    train_opts.batch_size = 64
    train_opts.n_epochs = 50
    train_opts.use_gpu = True
    train_opts.custom_validation_func = validate_distillation
    train_opts.save_model = False
    # Define loss
    dist_loss = DistillationLoss(1, 2.5, 0.25)  # TODO: Search for the correct values from the paper

    # Initialzie student
    init_data, init_labels = train_set.get_batch(-1, 0)
    student.initialize(init_data, init_labels)
    student.to("cuda:0")
    # Define distillation network
    dist_net = DistillNet(student, teacher)
    trainer = trn.Trainer(dist_net, train_opts)
    results = trainer.train(dist_loss, train_set, val_set, is_classification=True)
    torch.save(student.state_dict(), "models/student")


