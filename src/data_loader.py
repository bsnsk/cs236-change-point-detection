import arff
import numpy as np
import torch
from os.path import join
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable

data_names = ['eeg', 'syn']

def load_eeg(dir):
    path_under_dir = "EEG/EEG Eye State.arff.txt"
    data = np.array(arff.load(open(join(dir, path_under_dir), "r"))['data'])
    X = data[:, :-1].astype(np.float)
    Y = data[:, -1].astype(np.int)
    return X, Y

def sliding_window(X, window_size, step_size=1):
    return np.hstack(X[i:1 + i - window_size or None:step_size] for i in range(0, window_size))

def load_syn(dir, window_size, normalize=True):
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

    if normalize:
        scaler = MinMaxScaler(feature_range=(0,1))
        X_train = scaler.fit(X_train).transform(X_train)
        X_dev = scaler.fit(X_dev).transform(X_dev)

    return X_train, y_train, X_dev, y_dev

def load_and_build_tensors(data_name, args, device):
    assert(data_name in data_names)
    if data_name == "eeg":
        X_raw, _ = load_eeg(args.data_dir)
        print("Raw data shape: {}".format(X_raw.shape))

        train_size = int(X_raw.shape[0] * 0.8)
        X_train_raw, X_dev_raw = X_raw[:train_size, :], X_raw[train_size:, :]
        print("Raw X train shape: {}, X dev shape: {}".format(X_train_raw.shape, X_dev_raw.shape))

        X_train_sliding = sliding_window(X_train_raw, args.window_size)
        X_dev_sliding = sliding_window(X_dev_raw, args.window_size)
        print("Sliding X train shape: {}, X dev shape: {}".format(X_train_sliding.shape, X_dev_sliding.shape))
        X_train = Variable(torch.Tensor(X_train_sliding), requires_grad=False).to(device)
        X_dev = Variable(torch.Tensor(X_dev_sliding), requires_grad=False).to(device)
        input_dim = X_train.shape[1]
        y_train, y_dev = None, None
    elif data_name == "syn":
        X_train, y_train, X_dev, y_dev = load_syn(args.data_dir, args.window_size)
        input_dim = X_train.shape[1] // 2
        X_train = Variable(torch.Tensor(X_train), requires_grad=False).to(device)
        y_train = Variable(torch.Tensor(y_train), requires_grad=False).to(device)
        X_dev = Variable(torch.Tensor(X_dev), requires_grad=False).to(device)
        y_dev = Variable(torch.Tensor(y_dev), requires_grad=False).to(device)
        print("X_train shape: {}, y_train shape: {}".format(X_train.size(), y_train.size()))
    return (input_dim, {
        "X_train": X_train,
        "y_train": y_train,
        "X_dev": X_dev,
        "y_dev": y_dev,
    })
