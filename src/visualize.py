import torch
import json
import numpy as np
from torch.autograd import Variable
from data_loader import load_eeg, load_syn
from data_loader import load_eeg_raw
from flow_nice import NICEModel
import matplotlib  # a workaround for virtualenv on macOS
matplotlib.use('TkAgg')  # a workaround for virtualenv on macOS
import matplotlib.pyplot as plt

base_path = "./experiments/"
exp_name = "FLOW-2018-11-29 06:28:23.099959"  # select experiment
data_name = "eeg"                             # select data set
exp_path = "{}{}/".format(base_path, exp_name)
args_path = "{}args.txt".format(exp_path)
checkpoint_path = "{}checkpoints/best.pt".format(exp_path)

tau = 150  # TODO: tolerance

device = 'cpu'  # predict on CPU

data_loaders = {
    "eeg": load_eeg,
    "syn": load_syn,
}

data_loaders_raw = {
    "eeg": load_eeg_raw,
}

# Load arguments for the model
with open(args_path, "r") as f:
    args_dict = f.read()
args = json.loads(args_dict)
print(args)

X_train, y_train, X_dev, y_dev = data_loaders[data_name](
    "./data/raw/", args['window_size'])

if exp_name.startswith("FLOW"):
    model = NICEModel(input_dim=X_train.shape[1] // 2,
                      hidden_sizes=args['hidden_sizes'],
                      device=device).to(device)
else:
    assert(False)

model.load_state_dict(torch.load(checkpoint_path, map_location=device))
divider = args['window_size'] - 1 + y_train.shape[0]
y_truths = np.concatenate((y_train, y_dev))
Xs = np.concatenate((X_train, X_dev))
Xs = Variable(torch.Tensor(Xs), requires_grad=False).to(device)
print("X_train: {}".format(X_train.shape))
print("Xs: {}".format(Xs.shape))

y_preds = model.predict(Xs).detach().numpy()

plt.clf()
X_raw, _ = data_loaders_raw[data_name]("./data/raw/")
min_val = 1e30
max_val = 0
colors = [
        'b', 'y', 'g', 'grey', 'orange',
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
y_preds_backup = y_preds
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
        pre = [k for k in range(max(0, i - tau), i) if y_preds[k] >= threshold]
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
plt.savefig(
    "img/visualization-{}-{}.eps".format(data_name, exp_name.split("-")[0]),
    bbox_inches='tight',
)


plt.clf()
ys = sorted(y_preds.reshape([-1]))
plt.plot(range(len(ys)), ys, '.')
print("plot done")
plt.savefig("./img/tmp.png")