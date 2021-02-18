import numpy as np
import torch
import torch.nn.functional as F
import DeepTorch.Trainer as trn

from Networks.Networks import Student, DistillNet, StudentEncoder
from Networks.Teachers import TeacherCifar, TeacherMNIST, TeacherQuickDraw, TeacherResNet50, create_teacher
from Networks.Encoders import CifarEncoder
from distillation import DistillationLoss
import config as cfg
import matplotlib.pyplot as plt
import json

def validate_distillation(data, labels):
    outputs = torch.nn.functional.softmax(data[0], dim=1).cpu().data.numpy()
    outputs = [np.argmax(pred) for pred in outputs]
    labels = labels.cpu().data.numpy()
    corrects = [outputs[i] == labels[i] for i in range(len(outputs))]
    return sum(corrects) / len(corrects) * 100

def train_teacher(train_set, val_set, teacher, train_opts, teacher_model_path):
    # train_opts = trn.TrainingOptions()
    # train_opts.optimizer_type = trn.OptimizerType.Adam
    # train_opts.learning_rate = 0.01
    # train_opts.learning_rate_update_by_step = False  # Update schedular at epochs
    # train_opts.learning_rate_drop_factor = 0.5
    # train_opts.learning_rate_drop_type = trn.SchedulerType.StepLr
    # train_opts.learning_rate_drop_step_count = 2
    # train_opts.batch_size = 64
    # train_opts.weight_decay = 1e-5
    # train_opts.n_epochs = 8
    # train_opts.use_gpu = True
    # train_opts.save_model = True
    # train_opts.saved_model_name = teacher_model_path

    teacher.to("cuda:0")
    trainer = trn.Trainer(teacher, train_opts)
    trainer.train(torch.nn.CrossEntropyLoss(), train_set, val_set, is_classification=True)
    return teacher

def train_student_encoded(train_set, val_set, teacher, params):
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

    # Save Params
    with open(ROOT + "/params", "w") as f:
        json.dump(vars(args), f)
    STUDENT_MODEL_PATH = ROOT + "/student"

    train_opts = trn.TrainingOptions()
    train_opts.optimizer_type = trn.OptimizerType.Adam
    train_opts.learning_rate = 0.01
    train_opts.learning_rate_drop_type = trn.SchedulerType.StepLr
    train_opts.learning_rate_update_by_step = False  # Update at every epoch
    train_opts.learning_rate_drop_factor = 0.5  # Halve the learning rate
    train_opts.learning_rate_drop_step_count = params.learn_drop_epochs
    train_opts.batch_size = 128
    train_opts.n_epochs = params.n_epochs
    train_opts.use_gpu = True
    train_opts.custom_validation_func = validate_distillation
    train_opts.save_model = False
    train_opts.verbose_freq = 100
    train_opts.weight_decay = 1e-8
    train_opts.shuffle_data = True
    train_opts.regularization_method = None
    # Define loss
    dist_loss = DistillationLoss(STUDENT_TEMP, TEACHER_TEMP, ALPHA)

    encoder = CifarEncoder(n_dims=params.n_inputs)
    encoder.load_state_dict(torch.load("Networks/cifar_encoder"))
    student = StudentEncoder(n_memberships=N_RULES, n_inputs=params.n_inputs, n_outputs=10,
                      learnable_memberships=params.learn_ants,encoder=encoder,
                      fuzzy_type=params.fuzzy_type, use_sigma_scale=params.use_sigma_scale,
                      use_height_scale=params.use_height_scale)
    # Initialize student
    print("Initializing Student")
    train_set.shuffle_data()
    init_data, init_labels = train_set.get_batch(60000, 0, "cpu")
    student.initialize(init_data)
    # student.load_state_dict(torch.load(STUDENT_MODEL_PATH))
    print("Done Initializing Student")
    # student.fuzzy_layer.draw(5)
    # plt.plot(student.feature_extraction(init_data)[:,1:2], np.zeros(init_data.shape[0]), 'o')
    # plt.show()
    device = "cuda:"+args.gpu_no
    student.to(device)
    # Define distillation network
    dist_net = DistillNet(student, teacher)
    trainer = trn.Trainer(dist_net, train_opts)
    results = trainer.train(dist_loss, train_set, val_set, is_classification=True)
    torch.save(student.state_dict(), STUDENT_MODEL_PATH)
    trn.save_train_info(results, STUDENT_MODEL_PATH + "_train_info")

    return student

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

    # Save Params
    with open(ROOT+"/params", "w") as f:
        json.dump(vars(args), f)
    STUDENT_MODEL_PATH = ROOT + "/student"

    train_opts = trn.TrainingOptions()
    train_opts.optimizer_type = trn.OptimizerType.Adam
    train_opts.learning_rate = 0.01
    train_opts.learning_rate_drop_type = trn.SchedulerType.StepLr
    train_opts.learning_rate_update_by_step = False # Update at every epoch
    train_opts.learning_rate_drop_factor = 0.5  # Halve the learning rate
    train_opts.learning_rate_drop_step_count = params.learn_drop_epochs
    train_opts.batch_size = 128
    train_opts.n_epochs = params.n_epochs
    train_opts.use_gpu = True
    train_opts.custom_validation_func = validate_distillation
    train_opts.save_model = False
    train_opts.verbose_freq = 100
    train_opts.weight_decay = 1e-8
    train_opts.shuffle_data =True
    train_opts.regularization_method = None
    # Define loss
    dist_loss = DistillationLoss(STUDENT_TEMP, TEACHER_TEMP, ALPHA)

    student = Student(n_memberships=N_RULES, n_inputs=params.n_inputs, n_outputs=10, learnable_memberships=params.learn_ants,
                      fuzzy_type=params.fuzzy_type, use_sigma_scale=params.use_sigma_scale, use_height_scale=params.use_height_scale)
    # Initialize student
    print("Initializing Student")
    train_set.shuffle_data()
    init_data, init_labels = train_set.get_batch(60000, 0, "cpu")
    if params.fuzzy_type == 1:
        sigma_mag = 3
    else:
        sigma_mag = 2
    student.initialize(init_data, init_labels, load_params=False, filename="clusters7", sigma_mag=sigma_mag)
    #student.load_state_dict(torch.load(STUDENT_MODEL_PATH))
    print("Done Initializing Student")
    #student.fuzzy_layer.draw(5)
    #plt.plot(student.feature_extraction(init_data)[:,1:2], np.zeros(init_data.shape[0]), 'o')
    #plt.show()
    device = "cuda:{}".format(args.gpu_no)
    student.to(device)
    for param in student.parameters():
        print(param.device)
        break
    # Define distillation network
    dist_net = DistillNet(student, teacher)
    trainer = trn.Trainer(dist_net, train_opts)
    results = trainer.train(dist_loss, train_set, val_set, is_classification=True)
    torch.save(student.state_dict(), STUDENT_MODEL_PATH)
    trn.save_train_info(results, STUDENT_MODEL_PATH + "_train_info")

    return student

def run_experiments(train_set, val_set, teacher, param):
    print("Running experiment ID:{} No:{}".format(param.exp_id, param.exp_no))

    if param.dataset == 4:
        train_student(train_set, val_set, teacher, param)
    elif param.dataset == 3:
        # Use autoencoder as dimensionality reduction for cifar
        train_student_encoded(train_set, val_set, teacher, param)
    elif param.dataset == 2:
        # Use PCA as dimensionality reduction for fashion mnist
        train_student(train_set, val_set, teacher, param)
    elif param.dataset == 1:
        # Use PCA as dimensionality reduction for mnist
        train_student(train_set, val_set, teacher, param)
    else:
        print("Invalid dataset id")

if __name__ == "__main__":
    import os
    from DeepTorch.Datasets.QuickDraw import QuickDrawLoader
    from DeepTorch.Datasets.Cifar import CifarLoader
    from DeepTorch.Datasets.MNIST import MNISTLoader
    from DeepTorch.Datasets.FashionMNIST import FashionMNISTLoader
    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", default=9999, type=int)
    parser.add_argument("--exp_no", default=1, type=int)
    parser.add_argument("--student_temp", default=1, type=float)
    parser.add_argument("--teacher_temp", default=2.5, type=float)
    parser.add_argument("--alpha", type=float, default=0.0, help="Alpha variable in the loss. 1 means full KL")
    parser.add_argument("--n_rules", type=int, default=7, help="Number of memberships")
    parser.add_argument("--learn_ants", type=bool, default=True, help="If set to true, membership funcitons won't be learned")
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--learn_drop_epochs", type=int, default=5, help="Number of epochs to train before updating learning rate")
    parser.add_argument("--n_inputs", type=int, default=30, help="Number of inputs of fuzzy layer")
    parser.add_argument("--fuzzy_type", type=int, default=2, help="Type of the fuzzy system (1 or 2)")
    parser.add_argument("--dataset", type=int, default=4, help="MNIST:1, FashionMNIST:2, Cifar:3")
    parser.add_argument("--use_sigma_scale", default=1, type=int)
    parser.add_argument("--use_height_scale", default=0, type=int)
    parser.add_argument("--gpu_no", default=0, type=int)

    args = parser.parse_args()
    args.exp_id = "{}_{}_{}_{}_{}_{}_{}_{}".format(args.dataset, args.fuzzy_type, args.n_inputs, args.n_rules, args.alpha, args.teacher_temp, args.use_sigma_scale, args.use_height_scale)
    # Load dataset
    if args.dataset == 4:  # Quick Draw
        TEACHER_PATH = "./models/teacher/QuickDraw/teacher_quick_draw"
        metadata = [['airplanes.npy',0], ['apple.npy', 1], ['bread.npy', 2], ['dog.npy', 3], ['guitar.npy', 4], ['lion.npy',5],
                    ['star.npy',6], ['zebra.npy',7], ['anvil.npy',8], ['car.npy',9]]
        qdLoader = QuickDrawLoader(metadata=metadata, data_root='./data/QuickDraw', max_data=10000)
        train_set = qdLoader.get_training_dataset()
        val_set = qdLoader.get_validation_dataset()
        test_set = qdLoader.get_test_dataset()

        # Teacher Options for QuickDraw
        teacher_train_opts = trn.TrainingOptions()
        teacher_train_opts.optimizer_type = trn.OptimizerType.Adam
        teacher_train_opts.learning_rate = 0.01
        teacher_train_opts.learning_rate_update_by_step = False  # Update schedular at epochs
        teacher_train_opts.learning_rate_drop_factor = 0.5
        teacher_train_opts.learning_rate_drop_type = trn.SchedulerType.StepLr
        teacher_train_opts.learning_rate_drop_step_count = 2
        teacher_train_opts.batch_size = 64
        teacher_train_opts.weight_decay = 1e-5
        teacher_train_opts.n_epochs = 8
        teacher_train_opts.use_gpu = True
        teacher_train_opts.save_model = True
        teacher_train_opts.saved_model_name = TEACHER_PATH

    elif args.dataset == 3:
        TEACHER_PATH = "./models/teacher/Cifar/teacher_cifar_resnet"
        cLoader = CifarLoader(validation_partition=0.9)
        train_set = cLoader.get_training_dataset()
        val_set = cLoader.get_validation_dataset()
        test_set = cLoader.get_test_dataset()

        # Teacher Train Opts for Cifar
        teacher_train_opts = trn.TrainingOptions()
        teacher_train_opts.optimizer_type = trn.OptimizerType.Adam
        teacher_train_opts.learning_rate = 0.001
        teacher_train_opts.learning_rate_update_by_step = False  # Update schedular at epochs
        teacher_train_opts.learning_rate_drop_factor = 0.5
        teacher_train_opts.learning_rate_drop_type = trn.SchedulerType.StepLr
        teacher_train_opts.learning_rate_drop_step_count = 2
        teacher_train_opts.batch_size = 64
        teacher_train_opts.weight_decay = 1e-5
        teacher_train_opts.n_epochs = 8
        teacher_train_opts.use_gpu = True
        teacher_train_opts.save_model = True
        teacher_train_opts.saved_model_name = TEACHER_PATH

    elif args.dataset == 2:
        TEACHER_PATH = "./models/teacher/FashionMNIST/teacher_fashion_mnist"
        fLoader = FashionMNISTLoader("./data")
        train_set = fLoader.get_training_dataset()
        val_set = fLoader.get_validation_dataset()
        test_set = fLoader.get_test_dataset()


        # Teacher Train Opts for FashionMNIST
        teacher_train_opts = trn.TrainingOptions()
        teacher_train_opts.optimizer_type = trn.OptimizerType.Adam
        teacher_train_opts.learning_rate = 0.01
        teacher_train_opts.learning_rate_update_by_step = False  # Update schedular at epochs
        teacher_train_opts.learning_rate_drop_factor = 0.5
        teacher_train_opts.learning_rate_drop_type = trn.SchedulerType.StepLr
        teacher_train_opts.learning_rate_drop_step_count = 2
        teacher_train_opts.batch_size = 64
        teacher_train_opts.weight_decay = 1e-5
        teacher_train_opts.n_epochs = 8
        teacher_train_opts.use_gpu = True
        teacher_train_opts.save_model = True
        teacher_train_opts.saved_model_name = TEACHER_PATH

    else:
        TEACHER_PATH = "./models/teacher/MNIST/teacher_mnist"
        mLoader = MNISTLoader("./data")
        train_set = mLoader.get_training_dataset()
        val_set = mLoader.get_validation_dataset()
        test_set = mLoader.get_test_dataset()

        # Teacher Train Opts for MNISt
        teacher_train_opts = trn.TrainingOptions()
        teacher_train_opts.optimizer_type = trn.OptimizerType.Adam
        teacher_train_opts.learning_rate = 0.01
        teacher_train_opts.learning_rate_update_by_step = False  # Update schedular at epochs
        teacher_train_opts.learning_rate_drop_factor = 0.5
        teacher_train_opts.learning_rate_drop_type = trn.SchedulerType.StepLr
        teacher_train_opts.learning_rate_drop_step_count = 2
        teacher_train_opts.batch_size = 64
        teacher_train_opts.weight_decay = 1e-5
        teacher_train_opts.n_epochs = 8
        teacher_train_opts.use_gpu = True
        teacher_train_opts.save_model = True
        teacher_train_opts.saved_model_name = TEACHER_PATH

    teacher = create_teacher(args.dataset)
    # Try to load the teacher
    if os.path.exists(TEACHER_PATH):
        teacher.load_state_dict(torch.load(TEACHER_PATH))
    else:
        teacher = train_teacher(train_set, val_set, teacher, teacher_train_opts, TEACHER_PATH)

    run_experiments(train_set, val_set, teacher, args)