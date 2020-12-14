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
    student = Student(n_inputs=64, n_memberships=15, n_outputs=10)
    mLoader = MNISTLoader("./data")
    train_set = mLoader.get_training_dataset()
    test_set = mLoader.get_test_dataset()

    init_data, init_labels = train_set.get_batch(-1, 0, "cpu")
    student.fit_pca(init_data, init_labels)
    student.to("cpu")
    acc1 = test_experiment(student, test_set, 1, 1)
    print("Acc 1:{}".format(acc1))