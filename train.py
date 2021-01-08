import numpy as np
import torch
import torch.nn.functional as F
import DeepTorch.Trainer as trn
from Networks.Networks import TeacherLite, Teacher, Student, DistillNet
from distillation import DistillationLoss
import config as cfg
import matplotlib.pyplot as plt

def validate_distillation(data, labels):
    outputs = torch.nn.functional.softmax(data[0], dim=1).cpu().data.numpy()
    outputs = [np.argmax(pred) for pred in outputs]
    labels = labels.cpu().data.numpy()
    corrects = [outputs[i] == labels[i] for i in range(len(outputs))]
    return sum(corrects) / len(corrects) * 100

def train_teacher(train_set, val_set):
    train_opts = trn.TrainingOptions()
    train_opts.optimizer_type = trn.OptimizerType.Adam
    train_opts.learning_rate = 0.01
    train_opts.learning_rate_update_by_step = False  # Update schedular at epochs
    train_opts.learning_rate_drop_factor = 0.5
    train_opts.learning_rate_drop_type = trn.SchedulerType.StepLr
    train_opts.learning_rate_drop_step_count = 2
    train_opts.batch_size = 64
    train_opts.weight_decay = 1e-5
    train_opts.n_epochs = 8
    train_opts.use_gpu = True
    train_opts.save_model = True
    train_opts.saved_model_name = cfg.TEACHER_MODEL_PATH

    teacher = Teacher(10)
    trainer = trn.Trainer(teacher, train_opts)
    trainer.train(torch.nn.CrossEntropyLoss(), train_set, val_set, is_classification=True)
    return teacher

def train_student(train_set, val_set, teacher, params):
    EXP_NO = params.exp_no
    EXP_ID = params.exp_id
    STUDENT_TEMP = params.student_temp
    TEACHER_TEMP = params.teacher_temp
    ALPHA = params.alpha
    N_RULES = params.n_rules

    ROOT = "./models/{}".format(EXP_ID)
    if not os.path.exists(ROOT):
        os.mkdir(ROOT)
    ROOT = "./models/{}/{}".format(EXP_ID, EXP_NO)
    if not os.path.exists(ROOT):
        os.mkdir(ROOT)
    STUDENT_MODEL_PATH = ROOT + "/student"

    train_opts = trn.TrainingOptions()
    train_opts.optimizer_type = trn.OptimizerType.Adam
    train_opts.learning_rate = 0.01
    train_opts.learning_rate_drop_type = trn.SchedulerType.StepLr
    train_opts.learning_rate_update_by_step = False # Update at every epoch
    train_opts.learning_rate_drop_factor = 0.5  # Halve the learning rate
    train_opts.learning_rate_drop_step_count = params.learn_drop_epochs
    train_opts.batch_size = 64
    train_opts.n_epochs = params.n_epochs
    train_opts.use_gpu = True
    train_opts.custom_validation_func = validate_distillation
    train_opts.save_model = False
    train_opts.verbose_freq = 100
    train_opts.weight_decay = 1e-4
    train_opts.shuffle_data =True
    # Define loss
    dist_loss = DistillationLoss(STUDENT_TEMP, TEACHER_TEMP, ALPHA)

    student = Student(n_memberships=N_RULES, n_inputs=params.n_inputs, n_outputs=10, learnable_memberships=params.learn_ants, fuzzy_type=args.fuzzy_type)
    # Initialzie student
    print("Initializing Student")
    train_set.shuffle_data()
    init_data, init_labels = train_set.get_batch(60000, 0, "cpu")
    student.initialize(init_data, init_labels, load_params=False, filename="clusters7")
    #student.load_state_dict(torch.load(STUDENT_MODEL_PATH))
    print("Done Initializing Student")
    # student.fuzzy_layer.draw(1)
    # plt.plot(student.feature_extraction(init_data)[:,1:2], np.zeros(init_data.shape[0]), 'o')
    # plt.show()
    student.to("cuda:0")
    # Define distillation network
    dist_net = DistillNet(student, teacher)
    trainer = trn.Trainer(dist_net, train_opts)
    results = trainer.train(dist_loss, train_set, val_set, is_classification=True)
    torch.save(student.state_dict(), STUDENT_MODEL_PATH)
    trn.save_train_info(results, STUDENT_MODEL_PATH + "_train_info")

    return student

def run_experiments(train_set, val_set, teacher, param):
    print("Running experiment ID:{} No:{}".format(param.exp_id, param.exp_no))
    train_student(train_set, val_set, teacher, param)

if __name__ == "__main__":
    import os
    from DeepTorch.Datasets.Cifar import CifarLoader
    from DeepTorch.Datasets.MNIST import MNISTLoader
    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", default=5, type=int)
    parser.add_argument("--exp_no", default=1, type=int)
    parser.add_argument("--student_temp", default=1, type=float)
    parser.add_argument("--teacher_temp", default=2.5, type=float)
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha variable in the loss. 1 means full KL")
    parser.add_argument("--n_rules", type=int, default=15, help="Number of memberships")
    parser.add_argument("--learn_ants", type=bool, default=True, help="If set to true, membership funcitons won't be learned")
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--learn_drop_epochs", type=int, default=5, help="Number of epochs to train before updating learning rate")
    parser.add_argument("--n_inputs", type=int, default=10, help="Number of inputs of fuzzy layer")
    parser.add_argument("--fuzzy_type", type=int, default=1, help="Type of the fuzzy system (1 or 2)")
    args = parser.parse_args()
    # Load dataset
    if cfg.DATASET == "Cifar":
        cLoader = CifarLoader(validation_partition=0.7)
        train_set = cLoader.get_training_dataset()
        val_set = cLoader.get_validation_dataset()
        test_set = cLoader.get_test_dataset()
    else:
        mLoader = MNISTLoader("./data")
        train_set = mLoader.get_training_dataset()
        val_set = mLoader.get_validation_dataset()
        test_set = mLoader.get_test_dataset()

    if cfg.TRAIN_TEACHER:
        teacher = train_teacher(train_set, val_set)
    else:
        teacher = Teacher(10)
        teacher.load_state_dict(torch.load(cfg.TEACHER_MODEL_PATH))

    run_experiments(train_set, val_set, teacher, args)