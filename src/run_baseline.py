import os, json, argparse
from datetime import datetime

import torch
from torch.autograd import Variable

from data_loader import load_and_build_tensors, data_names
from autoencoder import AutoEncoder
from train import *

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
parser.add_argument("--latent_dim", type=int)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_dim, tensors = load_and_build_tensors(args.data, args, device)

auto_encoder = AutoEncoder(input_dim=input_dim,
                           hidden_sizes=args.hidden_sizes,
                           latent_dim=args.latent_dim
                           ).to(device)
print(auto_encoder)

exp_folder = os.path.join(REPO_DIR, "experiments/baseline-"+str(datetime.now()))
os.mkdir(exp_folder)
with open(os.path.join(exp_folder, "args.txt"), "w") as fp:
    fp.write(json.dumps(vars(args)))

model=auto_encoder
X_train=tensors['X_train']
y_train=tensors['y_train']
X_dev=tensors['X_dev']
y_dev=tensors['y_dev']
iter_max=args.iter_max
iter_eval=args.iter_eval
batch_size=args.batch_size
lr=args.learning_rate
reg=args.reg

save_dir = os.path.join(exp_folder, "checkpoints")
os.mkdir(save_dir)
summary_writer = tf.summary.FileWriter(exp_folder)

mse = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg)

iter = 0
best_dev_loss = float("+inf")
train_size = X_train.shape[0]
indices = np.arange(train_size)

X_dev = X_dev.view(-1, X_dev.shape[1] // 2)

with tqdm(total=iter_max) as pbar:
    while iter <= iter_max:
        np.random.shuffle(indices)
        for batch_start in np.arange(0, train_size, batch_size):
            model.train()
            optimizer.zero_grad()

            batch_indices = indices[batch_start: batch_start + batch_size]
            X_train_batch = X_train[batch_indices]
            y_train_batch = y_train[batch_indices]

            X_train_batch = X_train_batch.view(-1, X_train_batch.shape[1] // 2)
            z_train_batch = model.encode(X_train_batch)
            X_hat_batch = model.decode(z_train_batch)

            train_rec_loss = mse(X_hat_batch, X_train_batch)
            train_rec_loss.backward()
            optimizer.step()
            write_summaries(summary_writer, iter, "train", {"total_loss": train_rec_loss})

            if iter % iter_eval == 0:
                model.eval()
                with torch.no_grad():
                    z_dev = model.encode(X_dev)
                    X_hat_dev = model.decode(z_dev)

                    dev_rec_loss = mse(X_hat_dev, X_dev)

                write_summaries(summary_writer, iter, "dev", {"total_loss": dev_rec_loss})

                if dev_rec_loss < best_dev_loss:
                    best_dev_loss = dev_rec_loss
                    file_path = os.path.join(save_dir, "best.pt")
                    state = model.state_dict()
                    torch.save(state, file_path)
                    print("Best model saved to {}".format(file_path))
                    summary = "Best dev loss: {}".format(dev_rec_loss)
                    with open(os.path.join(exp_folder, "best_stats.txt"), 'w') as fp:
                        fp.write(summary)
                    print(summary)

            pbar.set_postfix(
                train_loss="{:.2e}".format(train_rec_loss),
                dev_loss="{:.2e}".format(dev_rec_loss),
            )
            iter += 1
            pbar.update(1)
