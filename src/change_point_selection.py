import numpy as np
import torch
from torch.autograd import Variable
import json
from data_loader import load_eeg_raw, sliding_window
from data_loader import load_one_syn_raw
from autoencoder import AutoEncoder
from evaluate import draw_roc_threshold, compute_auc
import matplotlib  # a workaround for virtualenv on macOS
matplotlib.use('TkAgg')  # a workaround for virtualenv on macOS
import matplotlib.pyplot as plt

# directory info
# choose different experiment runs by changing exp_path
base_path = "./"
exp_name = "2018-11-18 22:11:30.930851"  # EEG
# exp_name = "baseline-2018-11-29 23:35:18.669600"  # syn
exp_path = "{}experiments/{}/".format(base_path, exp_name)
args_path = "{}args.txt".format(exp_path)
checkpoint_path = "{}checkpoints/best.pt".format(exp_path)

data_name = "eeg"  # TODO: data set
data_idx = 46

# Load arguments for the model
with open(args_path, "r") as f:
    args_dict = f.read()
args = json.loads(args_dict)

device = 'cpu'  # predict on CPU

# load data and obtain latent representations z
if data_name == "eeg":
    X_raw, y_raw = load_eeg_raw("{}data/raw/".format(base_path))
elif data_name == "syn":
    X_raw, y_raw = load_one_syn_raw("{}data/raw/".format(base_path), data_idx)
else:
    assert(False)
print("y_raw: {}".format(y_raw.shape))
X_sliding = sliding_window(X_raw, args["window_size"])
X_variable = Variable(torch.Tensor(X_sliding), requires_grad=False).to(device)
auto_encoder = AutoEncoder(input_dim=X_sliding.shape[1],
                           hidden_sizes=args["hidden_sizes"],
                           latent_dim=args["latent_dim"],
                           ).to(device)
auto_encoder.load_state_dict(torch.load(checkpoint_path, map_location=device))
z = auto_encoder.encode(X_variable).detach().numpy()
print(z.shape)


def find_peaks(z):
    dists = np.sqrt(np.sum(np.diff(z, axis=0) ** 2, axis=1))
    print(dists.shape)

    def mean(xs):
        return sum(xs) * 1. / len(xs)

    # inspect width, i.e. for t we inspect [t-d, t+d]
    d = 50
    indices = [
        i
        for i in range(d, dists.shape[0] - d)
        if dists[i] > max(dists[i-d:i]) and dists[i] > max(dists[i+1:i+d+1])
        and mean(dists[i-d:i]) < mean(dists[i-d//2:i+d//2])
        and mean(dists[i+1:i+d]) < mean(dists[i-d//2:i+d//2])
    ]

    # remove candidates that are very close
    result = [indices[0]]
    for idx in indices:
        if idx - result[-1] >= d * 2:
            result.append(idx)
    indices = result
    print("# indices ({}): {}".format(len(indices), indices))
    return indices, dists


def visualize(X_raw, y_raw, indices):
    featureLB = 3000
    featureUB = 5000
    plt.clf()

    # feature 0
    xs = [x
          for x in range(X_raw.shape[0])
          if X_raw[x, 0] < featureUB and X_raw[x, 0] > featureLB]
    ys = [X_raw[x, 0] for x in xs]
    plt.plot(xs, ys, '-b', label="feature 0")

    # feature 1
    xs = [x
          for x in range(X_raw.shape[0])
          if X_raw[x, 1] < featureUB and X_raw[x, 1] > featureLB]
    ys = [X_raw[x, 1] for x in xs]
    plt.plot(xs, ys, '-g', label="feature 1")

    # detected indices
    alarmLB = 3500
    alarmUB = 4500
    plt.plot([], [], '-c', label="detected")
    for idx in indices:
        plt.plot([idx - 1, idx - 1], [alarmLB, alarmUB], '-c', linewidth=0.7)

    # label
    labelLB = 3900
    labelUB = 4800
    plt.plot([], [], '-r', label="label")
    truths = [x for x in range(1, y_raw.shape[0]) if y_raw[x] != y_raw[x - 1]]
    for x in truths:
        plt.plot([x, x], [labelLB, labelUB], '-r', linewidth=0.7)
    print("# labels ({}): {}".format(len(truths), truths))
    print("# overlap {}".format(len([1 for t in indices if t in truths])))

    plt.legend()
    plt.xlabel("time")
    plt.ylabel("amplitude")
    plt.savefig("img/visualization.eps")
    plt.savefig("img/visualization.png")


indices, dists = find_peaks(z)
y_preds = np.zeros([y_raw.shape[0], 1])
for i in indices:
    y_preds[i + args["window_size"] - 1] = 1.0
print("# indices: {}".format(indices))
np.save("./log/{}-preds-{}{}.npy".format(
    data_name, 'Baseline', '-{}'.format(data_idx) if data_name == "syn" else ""
), y_preds)
# visualize(X_raw, y_raw, indices)

print("# dists: {}".format([dists[i] for i in indices]))

truths = [x for x in range(1, y_raw.shape[0]) if y_raw[x] != y_raw[x - 1]]
# draw_roc_threshold(indices, truths, dists)
auc, auprc = compute_auc(indices, truths)
print("AUC = {}, AUPRC = {}".format(auc, auprc))
