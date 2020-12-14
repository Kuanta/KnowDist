import numpy as np
import torch
import DeepTorch.Trainer as trn
from Networks.Networks import TeacherLite, Student, DistillNet
from distillation import DistillationLoss
import config as cfg

def validate_distillation(data, labels):
    outputs = data[0].cpu().data.numpy()
    outputs = [np.argmax(pred) for pred in outputs]
    labels = labels.cpu().data.numpy()
    corrects = [outputs[i] == labels[i] for i in range(len(outputs))]
    return sum(corrects) / len(corrects) * 100

def train_teacher(train_set, val_set):
    train_opts = trn.TrainingOptions()
    train_opts.optimizer_type = trn.OptimizerType.Adam
    train_opts.learning_rate = 0.001
    train_opts.batch_size = 64
    train_opts.n_epochs = 2
    train_opts.use_gpu = True
    train_opts.save_model = True
    train_opts.saved_model_name = cfg.TEACHER_MODEL_PATH

    teacher = TeacherLite(10)
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
    dist_loss = DistillationLoss(STUDENT_TEMP, TEACHER_TEMP, ALPHA)

    student = Student(n_memberships=N_RULES, n_inputs=64, n_outputs=10)
    init_data, init_labels = train_set.get_batch(-1, 0, "cpu")
    student.initialize(init_data, init_labels)
    student.to("cuda:0")

    # Initialzie student
    print("Initializing Student")
    init_data, init_labels = train_set.get_batch(-1, 0, "cpu")
    student.initialize(init_data, init_labels)
    print("Done Initializing Student")
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
    parser.add_argument("--exp_id", default=1, type=int)
    parser.add_argument("--exp_no", default=1, type=int)
    parser.add_argument("--student_temp", default=1, type=float)
    parser.add_argument("--teacher_temp", default=1, type=float)
    parser.add_argument("--alpha", type=float, default=0, help="ALpha variable in the loss. 1 means full KL")
    parser.add_argument("--n_rules", type=int, default=7, help="Number of memberships")

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
        teacher = TeacherLite(10)
        teacher.load_state_dict(torch.load(cfg.TEACHER_MODEL_PATH))

    if cfg.TRAIN:
        run_experiments(train_set, val_set, teacher, args)