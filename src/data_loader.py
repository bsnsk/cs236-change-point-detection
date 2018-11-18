import arff
import numpy as np

def load_eeg(path):
    data = np.array(arff.load(open(path, 'r'))['data'])
    X = data[:, :-1].astype(np.float)
    Y = data[:, -1].astype(np.int)
    return X, Y

def sliding_window(X, window_size, step_size=1):
    return np.hstack(X[i:1 + i - window_size or None:step_size] for i in range(0, window_size))