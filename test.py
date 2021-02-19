import numpy as np
import config as cfg
import torch
import os
import json
from DeepTorch.Trainer import plot_results
from Networks.Networks import Student
from DeepTorch.Datasets.Cifar import CifarLoader
from DeepTorch.Datasets.MNIST import MNISTLoader
from DeepTorch.Datasets.QuickDraw import QuickDrawLoader
from DeepTorch.Datasets.FashionMNIST import FashionMNISTLoader

def test_accuracy(model, data, labels):
    data = model.forward(data)
    outputs = torch.nn.functional.softmax(data, dim=1).cpu().data.numpy()
    outputs = [np.argmax(pred) for pred in outputs]
    labels = labels.cpu().data.numpy()
    corrects = [outputs[i] == labels[i] for i in range(len(outputs))]
    return sum(corrects) / len(corrects) * 100

    # data = model.forward(data)
    # outputs = data.cpu().data.numpy()
    # outputs = [np.argmax(pred) for pred in outputs]
    # labels = labels.cpu().data.numpy()
    # corrects = [outputs[i] == labels[i] for i in range(len(outputs))]
    # return sum(corrects) / len(corrects) * 100

def test_experiment(student, test_set, exp_id, exp_no):
    # Check result json
    data = {}
    if os.path.exists("./results.json"):
        f = open("./results.json")
        data = json.load(f)
        f.close()

    root = "./models/{}/{}".format(exp_id, exp_no)
    model_path = root + "/student"
    if not os.path.exists(model_path):
        print("Model not trained yet")
        return 0
    student.load_state_dict(torch.load(model_path, map_location="cpu"))
    test_set.shuffle_data()
    test_data, test_labels = test_set.get_batch(-1, 0, "cpu")

    # teacher = Teacher(10)
    # teacher.load_state_dict(torch.load(cfg.TEACHER_MODEL_PATH))
    # teacher_acc = test_accuracy(teacher, test_data.float(), test_labels.float())

    curr_acc = test_accuracy(student, test_data.float(), test_labels.float())
    if not args.exp_id in data.keys():
        data[exp_id] = {
            "accs" : [],
            "max_acc" : 0,
            "avg_acc" : 0,
            "losses":[],
            "min_loss":0,
           " avg_loss":0
        }

    if len(data[exp_id]["accs"]) < args.exp_no:
        # Fill the empty spaces
        for i in range(args.exp_no - len(data[exp_id]["accs"])):
            data[exp_id]["accs"].append(0)
    print(args.exp_no)
    data[exp_id]["accs"][args.exp_no - 1] = curr_acc

    # Get max
    data[exp_id]["max_acc"] = max(data[exp_id]["accs"])
    data[exp_id]["avg_acc"] = sum(data[exp_id]["accs"])/len(data[exp_id]["accs"])
    with open('results.json', 'w') as fp:
        json.dump(data, fp)

    return curr_acc

if __name__ == "__main__":
    import argparse
    import DeepTorch.Trainer as trn

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", type=int, default=1)
    parser.add_argument("--exp_no", type=int, default=1)
    parser.add_argument("--fuzzy_type", default=2, type=int)
    parser.add_argument("--use_sigma_scale", type=int, default=0)
    parser.add_argument("--use_height_scale", type=int, default=1)
    parser.add_argument("--teacher_temp", type=float, default=2.5)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--n_inputs", type=int, default=15)
    parser.add_argument("--n_rules", type=int, default=7)
    parser.add_argument("--dataset", type=int, default=4)

    args = parser.parse_args()
    args.exp_id = "{}_{}_{}_{}_{}_{}_{}_{}".format(args.dataset, args.fuzzy_type, args.n_inputs, args.n_rules,
                                                   args.alpha, args.teacher_temp, args.use_sigma_scale,
                                                   args.use_height_scale)

    if args.dataset == 4:  # Quick Draw
        TEACHER_PATH = "./models/teacher/QuickDraw/teacher_quick_draw"
        metadata = [['airplanes.npy',0], ['apple.npy', 1], ['bread.npy', 2], ['dog.npy', 3], ['guitar.npy', 4], ['lion.npy',5],
                    ['star.npy',6], ['zebra.npy',7], ['anvil.npy',8], ['car.npy',9]]
        qdLoader = QuickDrawLoader(metadata=metadata, data_root='./data/QuickDraw', max_data=10000)
        train_set = qdLoader.get_training_dataset()
        val_set = qdLoader.get_validation_dataset()
        test_set = qdLoader.get_test_dataset()

    elif args.dataset == 3:  # Cifar
        cLoader = CifarLoader(validation_partition=0.7)
        train_set = cLoader.get_training_dataset()
        test_set = cLoader.get_test_dataset()
    elif args.dataset == 2:
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

    root = "./models/{}/{}".format(args.exp_id, args.exp_no)
    model_path = root + "/student"
    if os.path.exists(model_path):
        acc1 = test_experiment(student, val_set, args.exp_id, args.exp_no)
        print("Exp ID:{} No:{} Acc:{}".format(args.exp_id, args.exp_no, acc1))

    # info = trn.get_info(info_path, print_out=False)
    # acc1 = info[4][-1]
    # file1 = open("results.txt", "a")
    # file1.write("Exp ID:{} - Exp No:{} - Acc 1:{}\n".format(args.exp_id, args.exp_no, acc1))
    # file1.close()

    # Plot results
    # root = "./models/{}/{}".format(args.exp_id, args.exp_no)
    # info_path = root + "/student_train_info"
    # plot_results(info_path, "Training Results")
