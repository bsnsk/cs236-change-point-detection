# NICE
import os
import json
import argparse
from datetime import datetime

import torch

from data_loader import load_and_build_tensors, data_names
from flow_nice import NICEModel
from train_flow import train

REPO_DIR = os.path.relpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Data
parser.add_argument("--data", choices=data_names, default="syn", help="Which dataset to use")
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

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_dim, tensors = load_and_build_tensors(args.data, args, device)

flow_model = NICEModel(input_dim=input_dim,
                       hidden_sizes=args.hidden_sizes,
                       device=device).to(device)
print(flow_model)

exp_folder = os.path.join(REPO_DIR, "experiments/FLOW-"+str(datetime.now()))
os.mkdir(exp_folder)
with open(os.path.join(exp_folder, "args.txt"), "w") as fp:
    fp.write(json.dumps(vars(args)))

train(
    model=flow_model, exp_folder=exp_folder,
    X_train=tensors['X_train'], y_train=tensors['y_train'],
    X_dev=tensors['X_dev'], y_dev=tensors['y_dev'],
    iter_max=args.iter_max, iter_eval=args.iter_eval,
    batch_size=args.batch_size, lr=args.learning_rate, reg=args.reg,
)
