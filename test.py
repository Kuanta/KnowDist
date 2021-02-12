import numpy as np
import config as cfg
import torch
from DeepTorch.Trainer import plot_results
from Networks.Networks import Student
from DeepTorch.Datasets.Cifar import CifarLoader
from DeepTorch.Datasets.MNIST import MNISTLoader
from DeepTorch.Datasets.FashionMNIST import FashionMNISTLoader

def test_accuracy(model, data, labels):
    data = model.forward(data)
    outputs = data.cpu().data.numpy()
    outputs = [np.argmax(pred) for pred in outputs]
    labels = labels.cpu().data.numpy()
    corrects = [outputs[i] == labels[i] for i in range(len(outputs))]
    return sum(corrects) / len(corrects) * 100

def test_experiment(student, test_set, exp_id, exp_no):
    # Test Teacher

    root = "./models/{}/{}".format(exp_id, exp_no)
    model_path = root + "/student"
    student.load_state_dict(torch.load(model_path))
    test_data, test_labels = test_set.get_batch(-1, 0, "cpu")

    # teacher = Teacher(10)
    # teacher.load_state_dict(torch.load(cfg.TEACHER_MODEL_PATH))
    # teacher_acc = test_accuracy(teacher, test_data.float(), test_labels.float())

    return test_accuracy(student, test_data.float(), test_labels.float())

if __name__ == "__main__":
    import argparse
    import DeepTorch.Trainer as trn

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", type=int, default=27)
    parser.add_argument("--exp_no", type=int, default=1)
    parser.add_argument("--fuzzy_type", default=1, type=int)
    parser.add_argument("--use_sigma_scale", type=int, default=1)
    parser.add_argument("--use_height_scale", type=int, default=1)
    parser.add_argument("--n_inputs", type=int, default=30)
    parser.add_argument("--n_rules", type=int, default=7)
    parser.add_argument("--dataset", type=int, default=2)

    args = parser.parse_args()

    if args.dataset == 3:  # Cifar
        cLoader = CifarLoader(validation_partition=0.7)
        train_set = cLoader.get_training_dataset()
        test_set = cLoader.get_test_dataset()
    elif args.dataset == 2:
        print("Fashion")
        fLoader = FashionMNISTLoader("./data")
        train_set = fLoader.get_training_dataset()
        test_set = fLoader.get_test_dataset()
    else:  #MNIST
        mLoader = MNISTLoader("./data")
        train_set = mLoader.get_training_dataset()
        test_set = mLoader.get_test_dataset()

    student = Student(n_inputs=args.n_inputs, n_memberships=args.n_rules, n_outputs=10, fuzzy_type=args.fuzzy_type,use_height_scale=args.use_height_scale, use_sigma_scale=args.use_sigma_scale)


    init_data, init_labels = train_set.get_batch(60000, 0, "cpu")
    student.fit_pca(init_data, init_labels)
    student.to("cpu")
    acc1 = test_experiment(student, test_set, args.exp_id, args.exp_no)
    print("Exp ID:{} No:{} Acc:{}".format(args.exp_id, args.exp_no, acc1))

    # info = trn.get_info(info_path, print_out=False)
    # acc1 = info[4][-1]
    file1 = open("results.txt", "a")
    file1.write("Exp ID:{} - Exp No:{} - Acc 1:{}\n".format(args.exp_id, args.exp_no, acc1))
    file1.close()

    # Plot results
    # root = "./models/{}/{}".format(args.exp_id, args.exp_no)
    # info_path = root + "/student_train_info"
    # plot_results(info_path, "Training Results")
