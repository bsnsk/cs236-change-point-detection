import os, json, argparse
from datetime import datetime

import torch
from torch.autograd import Variable

from data_loader import *
from autoencoder import AutoEncoder
from train import train

REPO_DIR = os.path.relpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Data
parser.add_argument("--data", choices=["eeg"], default="eeg", help="Which dataset to use")
parser.add_argument("--data_dir", type=str, help="Data directory")

# Experiments
parser.add_argument("--iter_max", type=int, default=20000, help="Max training iterations")
parser.add_argument("--iter_eval", type=int, default=50, help="Evaluate dev set performance every n iterations")

# Training
parser.add_argument("--batch_size", type=int, default=500)
parser.add_argument("--learning_rate", type=float, default=1e-1)
parser.add_argument("--reg", type=float, default=0)

# Model
parser.add_argument("--window_size", type=int)
parser.add_argument("--hidden_sizes", type=int, nargs="+")
parser.add_argument("--latent_dim", type=int)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_raw, _ = load_eeg(args.data_dir)
print("Raw data shape: {}".format(X_raw.shape))

# TODO Split train/dev/test in prerocess, and think about whether/how to shuffle?
train_size = int(X_raw.shape[0] * 0.8)
X_train_raw, X_dev_raw = X_raw[:train_size, :], X_raw[train_size:, :]
print("Raw X train shape: {}, X dev shape: {}".format(X_train_raw.shape, X_dev_raw.shape))

X_train_sliding = sliding_window(X_train_raw, args.window_size)
X_dev_sliding = sliding_window(X_dev_raw, args.window_size)
print("Sliding X train shape: {}, X dev shape: {}".format(X_train_sliding.shape, X_dev_sliding.shape))
X_train = Variable(torch.Tensor(X_train_sliding), requires_grad=False).to(device)
X_dev = Variable(torch.Tensor(X_dev_sliding), requires_grad=False).to(device)

auto_encoder = AutoEncoder(input_dim=X_train.shape[1],
                           hidden_sizes=args.hidden_sizes,
                           latent_dim=args.latent_dim
                           ).to(device)
print(auto_encoder)

exp_folder = os.path.join(REPO_DIR, "experiments/"+str(datetime.now()))
os.mkdir(exp_folder)
with open(os.path.join(exp_folder, "args.txt"), "w") as fp:
    fp.write(json.dumps(vars(args)))

train(model=auto_encoder, exp_folder=exp_folder, X_train=X_train, y_train=None, X_dev=X_dev, y_dev=None,
      iter_max=args.iter_max, iter_eval=args.iter_eval,
      batch_size=args.batch_size, lr=args.learning_rate, reg=args.reg)