'''
Implementation of knowledge distillation paper:
"Distilling a Deep Neural Network into a Takagi-Sugeno-Kang Fuzzy Inference
System"
'''

import torch
import DeepTorch.Trainer as trn
from DeepTorch.Datasets.Cifar import CifarDataset, CifarLoader
from DeepTorch.Datasets.MNIST import MNISTLoader, MNISTDataset
from Networks.Networks import Student, Teacher, DistillNet, TeacherLite
from distillation import DistillationLoss
import numpy as np
import os

# Flags
TRAIN = False
TRAIN_TEACHER = True
TEST = False
COMPARE = True
DATASET = "Mnist"

EXP_NO = 4

if not os.path.exists("./models/{}".format(EXP_NO)):
    os.mkdir("./models/{}".format(EXP_NO))
ROOT = "./models/{}".format(EXP_NO)

TEACHER_MODEL_PATH = "./models/teacher_lite"
STUDENT_MODEL_PATH = ROOT + "/student_modified_kl"

# Constants
STUDENT_TEMP = 1
TEACHER_TEMP = 2.5
ALPHA = 0.25

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
    val_set = mLoader.get_validation_dataset()
    test_set = mLoader.get_test_dataset()

# Define networks
student = Student(n_memberships=15, n_inputs=64, n_outputs=10)
teacher = TeacherLite(10)

# Load teacher
#teacher.load_state_dict(torch.load(TEACHER_MODEL_PATH))
if TRAIN_TEACHER:
    train_opts = trn.TrainingOptions()
    train_opts.optimizer_type = trn.OptimizerType.Adam
    train_opts.learning_rate = 0.001
    train_opts.batch_size = 64
    train_opts.n_epochs = 2
    train_opts.use_gpu = True
    train_opts.save_model = True
    train_opts.saved_model_name = TEACHER_MODEL_PATH

    trainer = trn.Trainer(teacher, train_opts)
    trainer.train(torch.nn.CrossEntropyLoss(), train_set, val_set, is_classification=True)
else:
    teacher.load_state_dict(torch.load(TEACHER_MODEL_PATH))
    print("Loading Teacher")
if TRAIN:
    # Define training options
    train_opts = trn.TrainingOptions()
    train_opts.optimizer_type = trn.OptimizerType.Adam
    train_opts.learning_rate = 0.01
    train_opts.learning_rate_drop_type = trn.SchedulerType.StepLr
    train_opts.learning_rate_update_by_step = False  # Update at every epoch
    train_opts.learning_rate_drop_factor = 0.5  # Halve the learning rate
    train_opts.learning_rate_drop_step_count = 5  # Drop learning rate at every 25 epochs
    train_opts.batch_size = 64
    train_opts.n_epochs = 20
    train_opts.use_gpu = True
    train_opts.custom_validation_func = validate_distillation
    train_opts.save_model = False
    train_opts.verbose_freq = 100
    train_opts.weight_decay = 0.004
    # Define loss
    dist_loss = DistillationLoss(STUDENT_TEMP, TEACHER_TEMP, ALPHA)  # TODO: Search for the correct values from the paper

    # Initialzie student
    init_data, init_labels = train_set.get_batch(-1, 0, "cpu")
    student.initialize(init_data, init_labels)
    student.to("cuda:0")
    # Define distillation network
    dist_net = DistillNet(student, teacher)
    trainer = trn.Trainer(dist_net, train_opts)
    results = trainer.train(dist_loss, train_set, val_set, is_classification=True)
    torch.save(student.state_dict(), STUDENT_MODEL_PATH)
    trn.save_train_info(results, STUDENT_MODEL_PATH+"_train_info")

if TEST:
    student.load_state_dict(torch.load(STUDENT_MODEL_PATH))
    teacher.to("cuda:0")
    test_batch, test_labels = test_set.get_batch(-1, 0, "cuda:0")
    teacher_preds = teacher.forward(test_batch.float())
    teacher_acc = validate_distillation([teacher_preds], test_labels)
    print("Teacher Acc:{}".format(teacher_acc))

    student.to("cuda:0")
    init_data, init_labels = train_set.get_batch(-1, 0, "cuda:0")
    student.fit_pca(init_data, init_labels)
    student_pred = student.forward(test_batch.float())
    student_acc = validate_distillation([student_pred], test_labels)
    print("Student Acc:{}".format(student_acc))

if COMPARE:
    # Compare kl and no kl
    student_no_kl = Student(n_memberships=15, n_inputs=64, n_outputs=10).to("cuda:0")
    student_kl = Student(n_memberships=15, n_inputs=64, n_outputs=10).to("cuda:0")
    student_modified_kl = Student(n_memberships=15, n_inputs=64, n_outputs=10).to("cuda:0")

    init_data, init_labels = train_set.get_batch(-1, 0, "cuda:0")
    test_batch, test_labels = test_set.get_batch(-1, 0, "cuda:0")

    student_no_kl.fit_pca(init_data, init_labels)
    student_no_kl.load_state_dict(torch.load(ROOT+"/student_modified_kl"))
    no_kl_res = student_no_kl.forward(test_batch.float())
    no_kl_acc = validate_distillation([no_kl_res], test_labels)

    # student_kl.load_state_dict(torch.load(ROOT+"/student_kl"))
    # student_kl.fit_pca(init_data, init_labels)
    # kl_res = student_kl.forward(test_batch.float())
    # kl_acc = validate_distillation([kl_res], test_labels)

    student_modified_kl.load_state_dict(torch.load(ROOT + "/student_modified_kl"))
    student_modified_kl.fit_pca(init_data, init_labels)
    modified_kl_res = student_modified_kl = student_modified_kl.forward(test_batch.float())
    modified_kl_acc = validate_distillation([modified_kl_res], test_labels)

    print("No KL:{} - KL:{} - Modified KL:{}".format(no_kl_acc, 0, modified_kl_acc))
