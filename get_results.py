import sys
sys.path.append("..")
from utils.dl_utils import set_random_seed
import numpy as np
from tabulate import tabulate


def main():
    set_random_seed(seed=42)
    # subject_list = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]
    subject_list = [i for i in range(1, 16)]

    learn_type = "LDL-cheat"
    model = "MLP"
    dataset = "SEED5"
    sessions_id = [2, 3]

    f1s = []
    accs = []
    for subject in subject_list:
        subject_ = "Subject-{}".format(subject)
        for sess_id in sessions_id:
            session = "Session-{}".format(sess_id)
            if learn_type == "SLL":
                path = "./experiments/{}/{}/{}/{}/{}/".format(dataset, learn_type, model, subject_, session)
            else:
                path = "./experiments/{}/{}/LEVI/{}/{}/{}/".format(dataset, learn_type, model, subject_, session)
            filename = path + "metrics.txt"
            f = open(filename, 'r')
            metrics = f.read()
            f.close()
            aux = metrics.split("\t\t")
            acc = float(aux[0].split("=")[1])
            f1 = float(aux[1].split("=")[1])
            f1s.append(f1)
            accs.append(acc)


    acc_mean = np.around(np.mean(accs), 2)
    acc_std = np.around(np.std(accs), 2)
    acc_total = "{}+-{}".format(acc_mean, acc_std)
    f1_mean = np.around(np.mean(f1s), 2)
    f1_std = np.around(np.std(f1s), 2)
    f1_total = "{}+-{}".format(f1_mean, f1_std)

    hearders = ["ACC", "F1"]
    data = [[acc_total, f1_total]]
    table = tabulate(data, headers=hearders, tablefmt="fancy_grid")
    message = ("\nCross-session results for {} learning approach and"
               " model {} are:\n").format(learn_type, model)
    print(message)
    print(table)


if __name__ == '__main__':
    main()