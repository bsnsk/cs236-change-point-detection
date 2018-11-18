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