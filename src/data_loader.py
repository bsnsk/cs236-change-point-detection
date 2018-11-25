import arff
import numpy as np
from os.path import join

def load_eeg(dir):
    path_under_dir = "EEG/EEG Eye State.arff.txt"
    data = np.array(arff.load(open(join(dir, path_under_dir), "r"))['data'])
    X = data[:, :-1].astype(np.float)
    Y = data[:, -1].astype(np.int)
    return X, Y

def sliding_window(X, window_size, step_size=1):
    return np.hstack(X[i:1 + i - window_size or None:step_size] for i in range(0, window_size))

def load_syn(dir, window_size):
    path_under_dir = "syn"
    num_total_files = 50
    num_train_files = 45

    labels = []
    with open(join(dir, "{}/label.txt".format(path_under_dir)), "r") as fp:
        for line in fp.readlines():
            labels.append([int(num) for num in line.split("[")[1].split("]")[0].split()])

    X_train, y_train, X_dev, y_dev = [], [], [], []
    for i in range(0, num_total_files):
        X = np.expand_dims(np.loadtxt(join(dir, "{}/{}.txt".format(path_under_dir, i))), 1)  # (T, D_x)
        X = sliding_window(X, window_size=2*window_size)  # (T-2D_w+1, D_w*D_x)
        y = np.zeros((X.shape[0], 1), dtype=np.int)
        label_index = [
            index - window_size
            for index in labels[i]
            if (index-window_size) >= 0 and (index-window_size) < y.shape[0]
        ]
        y[label_index] = 1
        if i < num_train_files:
            X_train.append(X)
            y_train.append(y)
        else:
            X_dev.append(X)
            y_dev.append(y)
    X_train = np.vstack(X_train)
    y_train = np.vstack(y_train)
    X_dev = np.vstack(X_dev)
    y_dev = np.vstack(y_dev)
    return X_train, y_train, X_dev, y_dev