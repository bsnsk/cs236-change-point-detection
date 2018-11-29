import torch
import json
import numpy as np
from torch.autograd import Variable
from data_loader import load_eeg, load_eeg_raw
from data_loader import load_one_syn, load_one_syn_raw
from flow_nice import NICEModel
from autoencoder import AutoEncoder
import matplotlib  # a workaround for virtualenv on macOS
matplotlib.use('TkAgg')  # a workaround for virtualenv on macOS
import matplotlib.pyplot as plt

base_path = "./experiments/"
exp_name = "AE-2018-11-29 09:05:22.816729"  # select experiment
data_name = "syn"                             # select data set
exp_path = "{}{}/".format(base_path, exp_name)
args_path = "{}args.txt".format(exp_path)
checkpoint_path = "{}checkpoints/best.pt".format(exp_path)

tau = 150  # TODO: tolerance

device = 'cpu'  # predict on CPU

# Load arguments for the model
with open(args_path, "r") as f:
    args_dict = f.read()
args = json.loads(args_dict)
print(args)


def makePlot(X_raw, Xs, ys, idx=None):

    if exp_name.startswith("FLOW"):
        model = NICEModel(input_dim=Xs.shape[1] // 2,
                          hidden_sizes=args['hidden_sizes'],
                          device=device).to(device)
    elif exp_name.startswith("AE"):
        model = AutoEncoder(input_dim=Xs.shape[1] // 2,
                            hidden_sizes=args['hidden_sizes'],
                            latent_dim=args['latent_dim']
                            ).to(device)
    else:
        assert(False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    y_truths = ys
    Xs = Variable(torch.Tensor(Xs), requires_grad=False).to(device)
    print("Xs: {}".format(Xs.shape))

    y_preds = model.predict(Xs).detach().numpy()

    plt.clf()
    min_val = 1e30
    max_val = 0
    # colors = [
    #         'b', 'y', 'g', 'grey', 'orange',
    #         'm', 'k', 'hotpink', 'deepskyblue', 'lime',
    #         'olive', 'sienna', 'tan', 'rosybrown', 'darkred',
    # ]
    if len(X_raw.shape) > 1:
        plt.plot([], [], 'k-', linewidth='0.3', label='feature')
        for k in range(min(14, X_raw.shape[1])):
            offset = 400 * k
            plt.plot(range(X_raw.shape[0]), X_raw[:, k] + offset, 'k-',
                     linewidth='0.3')
            mi, ma = min(X_raw[:, k] + offset), max(X_raw[:, k] + offset)
            min_val = min(min_val, mi)
            max_val = max(max_val, ma)
    else:
        plt.plot(range(X_raw.shape[0]), X_raw, 'b-', label="feature")
        min_val, max_val = min(X_raw), max(X_raw)

    prefix = args['window_size'] - 1
    truthLB = (2 * min_val + max_val) / 3
    truthUB = (- min_val + 9 * max_val) / 8
    plt.plot([], [], 'r--', label="truth")
    for i in range(y_truths.shape[0]):
        if np.sum(y_truths[i]) > 0:
            plt.plot([prefix + i, prefix + i], [truthLB, truthUB],
                     'r--', linewidth='0.5')

    threshold = sorted(y_preds.reshape([-1]))[-100]  # TODO: threshold
    print("# threshold: %.6lf" % (threshold))
    y_preds = y_preds.reshape([-1])
    for i in range(y_truths.shape[0]):
        if y_truths[i, 0] > 0:
            exist = np.sum(y_preds[max(0, i - tau):i + tau + 1])
            if exist > 0:
                y_preds[i - tau:i + tau + 1] *= 0
                y_preds[i] = 1
                # print("# cleaning {} to {}: {}".format(
                #     i - tau, i + tau, np.sum(y_preds[i - tau:i + tau + 1])))
    for i in range(y_preds.shape[0]):
        if y_preds[i] >= threshold and np.sum(y_truths[i - tau:i + tau + 1]) == 0:
            pre = [
                k for k in range(max(0, i - tau), i) if y_preds[k] >= threshold
            ]
            if len(pre) > 1:
                y_preds[i] = 0

    predLB = max(0, (9 * min_val - max_val) / 8)
    predUB = (min_val + 2 * max_val) / 3
    print("y_preds: {}".format(y_preds.shape))
    plt.plot([], [], 'c--', label="prediction")
    tot = 0
    for i in range(y_preds.shape[0]):
        if y_preds[i] >= threshold:
            tot += 1
            plt.plot([prefix + i, prefix + i], [predLB, predUB],
                     'c--', linewidth='0.5')
    print("# tot = {}".format(tot))

    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
    plt.title("{}-{}".format(data_name.upper(), exp_name.split("-")[0]))
    filename = "img/visualization-{}-{}{}.eps".format(
        data_name, exp_name.split("-")[0],
        "" if idx is None else "-{}".format(idx))
    plt.savefig(filename, bbox_inches='tight')


#  main
if data_name == "eeg":
    X_train, y_train, X_dev, y_dev = load_eeg(
        "./data/raw/", args['window_size'])
    ys = np.concatenate((y_train, y_dev))
    Xs = np.concatenate((X_train, X_dev))
    X_raw, _ = load_eeg_raw("./data/raw/")
    makePlot(X_raw, Xs, ys)
elif data_name == "syn":
    for i in range(50):
        X_raw, _ = load_one_syn_raw("./data/raw/", i)
        Xs, ys = load_one_syn("./data/raw/", args['window_size'], i)
        makePlot(X_raw, Xs, ys)
else:
    assert(False)

# plt.clf()
# ys = sorted(y_preds.reshape([-1]))
# plt.plot(range(len(ys)), ys, '.')
# print("plot done")
# plt.savefig("./img/tmp.png")
