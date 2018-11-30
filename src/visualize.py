import torch
import json
import numpy as np
from sys import argv
from torch.autograd import Variable
from data_loader import load_eeg, load_eeg_raw
from data_loader import load_one_syn, load_one_syn_raw
from data_loader import load_iops, load_iops_raw
from flow_nice import NICEModel
from autoencoder import AutoEncoder
from VariationalAutoEncoder import VariationalAutoEncoder
import matplotlib  # a workaround for virtualenv on macOS
matplotlib.use('TkAgg')  # a workaround for virtualenv on macOS
import matplotlib.pyplot as plt

base_path = "./experiments/"
# exp_name = "FLOW-2018-11-29 10:19:03.279279"  # select experiment Flow syn
# exp_name = "AE-2018-11-28 00:25:52.158186"  # select experiment AE syn
exp_name = "VAE-2018-11-29 05:44:01.765073"  # select experiment VAE syn
# exp_name = "FLOW-2018-11-29 06:28:23.099959"  # select experiment Flow eeg
# exp_name = "AE-2018-11-29 09:05:22.816729"  # select experiment AE eeg
data_name = "syn"                             # select data set
exp_path = "{}{}/".format(base_path, exp_name)
args_path = "{}args.txt".format(exp_path)
checkpoint_path = "{}checkpoints/best.pt".format(exp_path)

tau = 80 if data_name == "syn" else 150  # TODO: tolerance
threshold = 0.8  # TODO: threshold

device = 'cpu'  # predict on CPU

# Load arguments for the model
with open(args_path, "r") as f:
    args_dict = f.read()
args = json.loads(args_dict)
print(args)


def makePlot(X_raw, Xs, ys, idx=None):
    print("# Xs: {}".format(Xs.shape))

    if exp_name.startswith("FLOW"):
        model = NICEModel(input_dim=Xs.shape[1] // 2,
                          hidden_sizes=args['hidden_sizes'],
                          device=device).to(device)
    elif exp_name.startswith("AE"):
        model = AutoEncoder(input_dim=Xs.shape[1] // 2,
                            hidden_sizes=args['hidden_sizes'],
                            latent_dim=args['latent_dim']
                            ).to(device)
    elif exp_name.startswith("VAE"):
        model = VariationalAutoEncoder(input_dim=Xs.shape[1] // 2,
                                       hidden_sizes=args['hidden_sizes'],
                                       latent_dim=args['latent_dim'],
                                       device=device
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

    predLB = (9 * min_val - max_val) / 8
    predUB = (min_val + 2 * max_val) / 3
    print("y_preds: {}".format(y_preds.shape))
    plt.plot([], [], 'c--', label="prediction")
    tot = 0
    for i in range(y_preds.shape[0]):
        if y_preds[i] >= threshold:
            y_preds[i] = 1
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

    # save preds, truths, and Xraw
    idxStr = "-{}".format(idx) if idx is not None else ""
    np.save("./log/{}-Xraw{}.npy".format(data_name, idxStr), X_raw)
    np.save("./log/{}-truths{}.npy".format(data_name, idxStr), y_truths)
    np.save("./log/{}-preds-{}{}.npy".format(
            data_name, exp_name.split("-")[0], idxStr), y_preds)


def makePlotsUnified(X_raw, y_truths, preds, idxStr):
    plt.clf()
    if data_name == "eeg":
        plt.figure(figsize=(4, 8))
    min_val = 1e30
    max_val = 0
    colors = [
            'orange', 'y', 'g', 'grey', 'blue',
            'm', 'k', 'hotpink', 'deepskyblue', 'lime',
            'olive', 'sienna', 'tan', 'rosybrown', 'darkred',
    ]
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
    print("# features plotted")

    num_models = len(preds)
    model_dist = 600 if data_name == "eeg" else 10
    offset = 3500 if data_name == "eeg" else min_val
    axisLB = offset - model_dist * num_models
    axisUB = max_val + (max_val - min_val) * 0.01
    k = 0
    for name in preds:
        ys = preds[name]
        pattern = '^' if name != 'label' else 'o'
        print("# plotting {} : {}".format(name, ys.shape))
        for i in range(ys.shape[0]):
            if ys[i] == 1:
                plt.plot([i, i], [axisLB, axisUB], 'r--', linewidth='0.3')
                plt.plot([i], [offset - k * model_dist],
                         pattern, color=colors[k],
                         markersize=3)
        plt.plot([], [], pattern, color=colors[k], label=name)
        k += 1
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=5)
    plt.xlabel("time")
    plt.yticks([])
    if data_name == "eeg":
        plt.ylim((1500, 10000))
    elif data_name == "syn":
        plt.ylim((-50, 70))
    plt.title("{} data set visualization".format(data_name.upper()))
    plt.savefig("./img/visualization-{}-all{}.eps".format(data_name, idxStr),
                bbox_inches='tight')


#  main
if len(argv) > 1 and argv[1] == "load":
    # load y_preds from files
    if data_name in ["eeg", "syn"]:
        idxStr = "-45" if data_name == "syn" else ""
        X_raw = np.load("./log/{}-Xraw{}.npy".format(data_name, idxStr))
        y_truths = np.load("./log/{}-truths{}.npy".format(data_name, idxStr))
        flow_preds = np.load("./log/{}-preds-FLOW{}.npy".format(
            data_name, idxStr))
        ae_preds = np.load("./log/{}-preds-AE{}.npy".format(data_name, idxStr))
        vae_preds = np.load("./log/{}-preds-VAE{}.npy".format(
            data_name, idxStr))
        makePlotsUnified(X_raw, y_truths, {
            "label": y_truths,
            "FLOW": flow_preds,
            "AE": ae_preds,
            "VAE": vae_preds,
        }, idxStr)
else:
    # load model and compute y_preds
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
            makePlot(X_raw, Xs, ys, i)
    elif data_name == "iops":
        X_train, y_train, X_dev, y_dev = load_iops(
            "./data/raw/", args['window_size'])
        ys = np.concatenate((y_train, y_dev))
        Xs = np.concatenate((X_train, X_dev))
        X_raw, _ = load_iops_raw("./data/raw/")
        print([i for i in range(ys.shape[0]) if ys[i, 0] > 0])
        makePlot(X_raw, Xs, ys)
    else:
        assert(False)

# plt.clf()
# ys = sorted(y_preds.reshape([-1]))
# plt.plot(range(len(ys)), ys, '.')
# print("plot done")
# plt.savefig("./img/tmp.png")
