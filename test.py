import numpy as np
import config as cfg
import torch
from Networks.Networks import Student
from DeepTorch.Datasets.Cifar import CifarLoader
from DeepTorch.Datasets.MNIST import MNISTLoader

def test_accuracy(model, data, labels):
    data = model.forward(data)
    outputs = data.cpu().data.numpy()
    outputs = [np.argmax(pred) for pred in outputs]
    labels = labels.cpu().data.numpy()
    corrects = [outputs[i] == labels[i] for i in range(len(outputs))]
    return sum(corrects) / len(corrects) * 100

def test_experiment(student, test_set, exp_id, exp_no):

    root = "./models/{}/{}".format(exp_id, exp_no)
    model_path = root + "/student"
    student.load_state_dict(torch.load(model_path))
    test_data, test_labels = test_set.get_batch(-1, 0, "cpu")
    return test_accuracy(student, test_data.float(), test_labels.float())

if __name__ == "__main__":
    import argparse
    import DeepTorch.Trainer as trn

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", default=3, type=int)
    parser.add_argument("--exp_no", default=1, type=int)

    args = parser.parse_args()

    # student = Student(n_inputs=64, n_memberships=15, n_outputs=10)
    # mLoader = MNISTLoader("./data")
    # train_set = mLoader.get_training_dataset()
    # test_set = mLoader.get_test_dataset()
    #
    # init_data, init_labels = train_set.get_batch(-1, 0, "cpu")
    # student.fit_pca(init_data, init_labels)
    # student.to("cpu")
    # acc1 = test_experiment(student, test_set, args.exp_id, args.exp_no)
    root = "./models/{}/{}".format(args.exp_id, args.exp_no)
    info_path = root + "/student_train_info"
    info = trn.get_info(info_path, print_out=False)
    acc1 = info[3][-1]
    file1 = open("results.txt", "a")
    file1.write("Exp ID:{} - Exp No:{} - Acc 1:{}\n".format(args.exp_id, args.exp_no, acc1))
    file1.close()
    print("Exp ID:{} - Exp No:{} - Acc 1:{}".format(args.exp_id, args.exp_no, acc1))