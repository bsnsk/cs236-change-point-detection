import arff
import numpy as np
import torch
from os.path import join
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable

data_names = ['eeg', 'syn', 'iops', 'har']


# WARNING: This only loads the X and y defined by the original data set,
# which is very different from what we have with sliding window.
def _load_har_raw(path):
    har_path = join(path, "har/")

    def loadX(filename):
        with open(filename, "r") as f:
            lines = f.readlines()
        data = [
            [float(ele) for ele in line.strip().split(" ") if len(ele) > 0]
            for line in lines
        ]
        return np.array(data)

    def loadY(filename):
        with open(filename, "r") as f:
            lines = f.readlines()
        data = [float(line.strip()) for line in lines]
        return np.array(data)

    X_train = loadX(join(har_path, "train/X_train.txt"))
    y_train = loadX(join(har_path, "train/y_train.txt"))
    X_dev = loadX(join(har_path, "test/X_test.txt"))
    y_dev = loadX(join(har_path, "test/y_test.txt"))
    return X_train, y_train, X_dev, y_dev


def _y_activity2change_point(ys):
    ys = ys.reshape([-1, 1])
    ys_processed = np.zeros(ys.shape)
    for i in range(1, ys.shape[0]):
        if ys[i, 0] != ys[i-1, 0]:
            ys_processed[i, 0] = 1
    return ys_processed


def load_har_raw(path):
    X1, y1, X2, y2 = _load_har_raw(path)
    Xs = np.concatenate((X1, X2))
    ys = np.concatenate((y1, y2)).reshape([-1, 1])
    return Xs, _y_activity2change_point(ys)


def load_har(path, window_size, normalize=True):
    X_train_raw, y_train_raw, X_dev_raw, y_dev_raw = _load_har_raw(path)
    X_train = sliding_window(X_train_raw, 2 * window_size)
    X_dev = sliding_window(X_dev_raw, 2 * window_size)

    y_train = y_train_raw[window_size - 1:-window_size]
    y_dev = y_dev_raw[window_size:-window_size + 1]

    if normalize:
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train = scaler.fit(X_train).transform(X_train)
        X_dev = scaler.fit(X_dev).transform(X_dev)

    print("HAR Data: {} 1's in y_train, {} 1's in y_dev".format(
        sum(y_train), sum(y_dev)))
    print("# X_train: {}".format(X_train.shape))

    return X_train, y_train, X_dev, y_dev


def load_iops_raw(dir):
    path_under_dir = "iops/server_res_eth1out_curve_6.csv"
    data = np.loadtxt(join(dir, path_under_dir), delimiter=",", skiprows=1)
    X, y = np.expand_dims(data[:, 1], 1), np.expand_dims(data[:, 2], 1)
    return X, y


def load_iops(dir, window_size, normalize=True):
    path_under_dir = "iops/server_res_eth1out_curve_6.csv"
    data = np.loadtxt(join(dir, path_under_dir), delimiter=",", skiprows=1)
    X, y = np.expand_dims(data[:, 1], 1), np.expand_dims(data[:, 2], 1)

    train_ratio = 0.75
    train_size = int(X.shape[0] * train_ratio)
    X_train, X_dev = X[:train_size, :], X[train_size:, :]
    X_train = sliding_window(X_train, 2 * window_size)
    X_dev = sliding_window(X_dev, 2 * window_size)

    y_train = y[window_size - 1 : train_size - window_size].reshape([-1, 1])
    y_dev = y[train_size + window_size : -window_size + 1].reshape([-1, 1])

    print("IOPS Data: {} 1's in y_train, {} 1's in y_dev".format(y_train.sum(), y_dev.sum()))
    if normalize:
        scaler = MinMaxScaler(feature_range=(0,1))
        X_train = scaler.fit(X_train).transform(X_train)
        X_dev = scaler.fit(X_dev).transform(X_dev)

    return X_train, y_train, X_dev, y_dev


def load_eeg_raw(dir):
    path_under_dir = "EEG/EEG Eye State.arff.txt"
    data = np.array(arff.load(open(join(dir, path_under_dir), "r"))['data'])
    X_raw = data[:, :-1].astype(np.float)
    y_raw = data[:, -1].astype(np.int)
    return X_raw, y_raw


def load_eeg(dir, window_size, normalize=True):
    X_raw, y_raw = load_eeg_raw(dir)

    y_processed = np.zeros(y_raw.shape, dtype=np.int)
    for i in range(1, y_raw.shape[0]):
        if y_raw[i] != y_raw[i - 1]:
            y_processed[i] = 1
    assert(y_processed.shape == y_raw.shape)

    train_ratio = 0.75

    train_size = int(X_raw.shape[0] * train_ratio)
    X_train_raw, X_dev_raw = X_raw[:train_size, :], X_raw[train_size:, :]
    X_train = sliding_window(X_train_raw, 2 * window_size)
    X_dev = sliding_window(X_dev_raw, 2 * window_size)

    y_train = y_processed[window_size - 1:train_size-window_size].reshape([-1, 1])
    y_dev = y_processed[train_size+window_size:-window_size + 1].reshape([-1, 1])

    print("EEG Data: {} 1's in y_train, {} 1's in y_dev".format(
        sum(y_train), sum(y_dev)))

    if normalize:
        scaler = MinMaxScaler(feature_range=(0,1))
        X_train = scaler.fit(X_train).transform(X_train)
        X_dev = scaler.fit(X_dev).transform(X_dev)
    return X_train, y_train, X_dev, y_dev

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
        X = sliding_window(X, window_size=2 * window_size)  # (T-2D_w+1, D_w*D_x)
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

def load_one_syn_raw(dir, file):
    path_under_dir = "syn"
    labels = []
    with open(join(dir, "{}/label.txt".format(path_under_dir)), "r") as fp:
        for line in fp.readlines():
            labels.append([int(num) for num in line.split("[")[1].split("]")[0].split()])
    X = np.expand_dims(np.loadtxt(join(dir, "{}/{}.txt".format(path_under_dir, file))), 1)
    y = labels[file]
    y = np.array([1 if i in y else 0 for i in range(X.shape[0])])
    return X, y

def load_one_syn(dir, window_size, file, normalize=True):
    path_under_dir = "syn"

    labels = []
    with open(join(dir, "{}/label.txt".format(path_under_dir)), "r") as fp:
        for line in fp.readlines():
            labels.append([int(num) for num in line.split("[")[1].split("]")[0].split()])

    X = np.expand_dims(np.loadtxt(join(dir, "{}/{}.txt".format(path_under_dir, file))), 1)
    X = sliding_window(X, window_size=2 * window_size)
    y = np.zeros((X.shape[0], 1), dtype=np.int)

    label_index = [
        index - window_size
        for index in labels[file]
        if (index - window_size) >= 0 and (index - window_size) < y.shape[0]
    ]
    y[label_index] = 1

    if normalize:
        scaler = MinMaxScaler(feature_range=(0,1))
        X = scaler.fit(X).transform(X)

    return X, y

def load_and_build_tensors(data_name, args, device):
    assert(data_name in data_names)
    if data_name == "eeg":
        X_train, y_train, X_dev, y_dev = load_eeg(args.data_dir, args.window_size)
    elif data_name == "syn":
        X_train, y_train, X_dev, y_dev = load_syn(args.data_dir, args.window_size)
    elif data_name == "iops":
        X_train, y_train, X_dev, y_dev = load_iops(args.data_dir, args.window_size)
    elif data_name == "har":
        X_train, y_train, X_dev, y_dev = load_har(args.data_dir, args.window_size)
    else:
        assert(False)

    input_dim = X_train.shape[1] // 2
    X_train = Variable(torch.Tensor(X_train), requires_grad=False).to(device)
    y_train = Variable(torch.Tensor(y_train), requires_grad=False).to(device)
    X_dev = Variable(torch.Tensor(X_dev), requires_grad=False).to(device)
    y_dev = Variable(torch.Tensor(y_dev), requires_grad=False).to(device)

    tensors = {
        "X_train": X_train,
        "y_train": y_train,
        "X_dev": X_dev,
        "y_dev": y_dev,
    }
    for name in tensors:
        print("Shape of {} is {}".format(
            name, tensors[name].shape if tensors[name] is not None else None))
    return (input_dim, tensors)
