import os, json, argparse, math
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm
import tensorflow as tf

from data_loader import load_and_build_tensors
import hvae
from train import write_summaries

REPO_DIR = os.path.relpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Data
parser.add_argument("--data", choices=["eeg", "syn"], default="syn", help="Which dataset to use")
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
parser.add_argument("--latent_dim", type=int, nargs="+")

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_dim, tensors = load_and_build_tensors(args.data, args, device)
X_train, y_train = tensors['X_train'], tensors['y_train']
X_dev, y_dev = tensors['X_dev'], tensors['y_dev']

# TODO args in
dim = 50
state_sizes = [dim, dim]
generative = hvae.GenerativeModel(state_sizes, [hvae.build_mlp(dim, [dim]), hvae.build_mlp(dim, [input_dim])])
recog = hvae.ApproxPosterior(input_dim, [dim, dim], state_sizes, 1)
model = hvae.HVAE(generative, recog).to(device)
print('Model structure:')
print(list(model.children()))

exp_folder = os.path.join(REPO_DIR, "experiments/HVAE-"+args.data+"-"+str(datetime.now()))
os.mkdir(exp_folder)
with open(os.path.join(exp_folder, "args.txt"), "w") as fp:
    fp.write(json.dumps(vars(args)))

save_dir = os.path.join(exp_folder, "checkpoints")
os.mkdir(save_dir)
summary_writer = tf.summary.FileWriter(exp_folder)

autoencoder=model
epochs = args.iter_max
batch_size=args.batch_size
lr=args.learning_rate


def train_batch(autoencoder, X, y, optimizer):
    optimizer.zero_grad()
    dist, output, samples = autoencoder(X)
    loss, rec, kl, pred, auc, auprc = autoencoder.loss(X, y, dist, output, samples)
    loss.backward()
    optimizer.step()
    return loss.item(), rec, kl, pred, auc, auprc


def evaluate_elbo(autoencoder, X, y, device):
    autoencoder.eval()
    autoencoder = autoencoder.to(device=device)
    X = X.to(device=device)
    X = X.view(-1, X.shape[1] // 2)
    y = y.to(device=device)
    dist, output, samples = autoencoder(X)
    loss, rec, kl, pred, auc, auprc = autoencoder.loss(X, y, dist, output, samples)
    return loss, rec, kl, pred, auc, auprc

n = X_train.size(0)

# this will probably leave out the last couple training points...
autoencoder = autoencoder.to(device=device)
sgd = torch.optim.Adam(autoencoder.parameters(), lr)

iter = 0
best_dev_auc = 0
train_size = X_train.shape[0]
indices = np.arange(train_size)


with tqdm(total=args.iter_max) as pbar:
    while iter <= args.iter_max:
        np.random.shuffle(indices)
        for start in np.arange(0, train_size, args.batch_size):
            autoencoder.train()
            data = X_train[start:start+batch_size].to(device=device)
            y_batch = y_train[start:start+batch_size].to(device=device)
            data = data.view(-1, data.shape[1] // 2)
            loss, rec, kl, pred, auc, auprc = train_batch(autoencoder, data, y_batch, sgd)

            write_summaries(summary_writer, iter, "train", {
                "total_loss": loss,
                "nelbo": rec+kl, "rec_loss": rec, "kld": kl,
                "ce_loss": pred,
                "auc": auc, "auprc": auprc
            })

            if iter % args.iter_eval == 0:
                autoencoder.eval()
                with torch.no_grad():
                    dev_loss, dev_rec, dev_kl, dev_pred, dev_auc, dev_auprc = evaluate_elbo(autoencoder, X_dev, y_dev, device)
                    write_summaries(summary_writer, iter, "dev", {
                        "total_loss": dev_loss,
                        "nelbo": dev_rec+dev_kl, "rec_loss": dev_rec, "kld": dev_kl,
                        "ce_loss": dev_pred,
                        "auc": dev_auc, "auprc": dev_auprc
                    })
                if dev_auc > best_dev_auc:
                    best_dev_auc = dev_auc
                    file_path = os.path.join(save_dir, "best.pt")
                    state = autoencoder.state_dict()
                    torch.save(state, file_path)
                    print("Best model saved to {}".format(file_path))
                    summary = "Best dev auc: {}".format(dev_auc)
                    with open(os.path.join(exp_folder, "best_stats.txt"), 'w') as fp:
                        fp.write(summary)
                    print(summary)
            pbar.set_postfix(
                train_loss="{:.2e}".format(loss),
                train_auc="%.4f" % auc,
                dev_loss="{:.2e}".format(dev_loss),
                dev_auc="%.4f" % dev_auc
            )
            iter += 1
            pbar.update(1)


