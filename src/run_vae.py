import os, json, argparse
from datetime import datetime

import torch
from torch.autograd import Variable
from tqdm import tqdm
from torch import nn, optim
from sklearn.metrics import roc_auc_score, average_precision_score

import tensorflow as tf

from data_loader import *
from VariationalAutoEncoder import VariationalAutoEncoder
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
parser.add_argument("--latent_dim", type=int)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TODO move this to data_loader
if args.data == "eeg":
    X_raw, _ = load_eeg(args.data_dir)
    print("Raw data shape: {}".format(X_raw.shape))

    train_size = int(X_raw.shape[0] * 0.8)
    X_train_raw, X_dev_raw = X_raw[:train_size, :], X_raw[train_size:, :]
    print("Raw X train shape: {}, X dev shape: {}".format(X_train_raw.shape, X_dev_raw.shape))

    X_train_sliding = sliding_window(X_train_raw, args.window_size)
    X_dev_sliding = sliding_window(X_dev_raw, args.window_size)
    print("Sliding X train shape: {}, X dev shape: {}".format(X_train_sliding.shape, X_dev_sliding.shape))
    X_train = Variable(torch.Tensor(X_train_sliding), requires_grad=False).to(device)
    X_dev = Variable(torch.Tensor(X_dev_sliding), requires_grad=False).to(device)
    input_dim = X_train.shape[1]
    y_train, y_dev = None, None
elif args.data == "syn":
    X_train, y_train, X_dev, y_dev = load_syn(args.data_dir, args.window_size)
    input_dim = X_train.shape[1] // 2
    X_train = Variable(torch.Tensor(X_train), requires_grad=False).to(device)
    y_train = Variable(torch.Tensor(y_train), requires_grad=False).to(device)
    X_dev = Variable(torch.Tensor(X_dev), requires_grad=False).to(device)
    y_dev = Variable(torch.Tensor(y_dev), requires_grad=False).to(device)
    print("X_train shape: {}, y_train shape: {}".format(X_train.size(), y_train.size()))

vae = VariationalAutoEncoder(input_dim=input_dim,
                           hidden_sizes=args.hidden_sizes,
                           latent_dim=args.latent_dim,
                           device=device
                           ).to(device)
print(vae)

exp_folder = os.path.join(REPO_DIR, "experiments/VAE-"+str(datetime.now()))
os.mkdir(exp_folder)
with open(os.path.join(exp_folder, "args.txt"), "w") as fp:
    fp.write(json.dumps(vars(args)))

save_dir = os.path.join(exp_folder, "checkpoints")
os.mkdir(save_dir)
summary_writer = tf.summary.FileWriter(exp_folder)

mse = nn.MSELoss()
ce_logits = nn.BCEWithLogitsLoss()
ce = nn.BCELoss()

optimizer = optim.Adam(vae.parameters(), lr=args.learning_rate, weight_decay=args.reg)

iter = 0
best_dev_auc = 0
train_size = X_train.shape[0]
indices = np.arange(train_size)

X_dev = X_dev.view(-1, X_dev.shape[1] // 2)

with tqdm(total=args.iter_max) as pbar:
    while iter <= args.iter_max:
        np.random.shuffle(indices)
        for batch_start in np.arange(0, train_size, args.batch_size):
            vae.train()
            optimizer.zero_grad()

            batch_indices = indices[batch_start: batch_start + args.batch_size]
            X_train_batch = X_train[batch_indices]
            y_train_batch = y_train[batch_indices]

            X_train_batch = X_train_batch.view(-1, X_train_batch.shape[1] // 2)
            qm_train, qv_train = vae.encode(X_train_batch)
            z_train_batch = vae.sample_gaussian(qm_train, qv_train)

            # NELBO = (-)REC + KL
            X_hat_batch = vae.decode(z_train_batch)
            # train_rec_loss = -torch.mean(-ce_logits(input=X_hat_logits_batch, target=X_train_batch).sum(-1), 0)
            train_rec_loss = mse(X_hat_batch, X_train_batch)
            train_kld = torch.mean(vae.kl_normal(qm_train, qv_train, vae.z_prior_m, vae.z_prior_v))
            train_nelbo = train_rec_loss + train_kld

            # Pred loss
            z_before = z_train_batch[[i for i in range(z_train_batch.size()[0]) if i % 2 == 0], :]
            z_after = z_train_batch[[i for i in range(z_train_batch.size()[0]) if i % 2 == 1], :]
            # probs = torch.sigmoid(torch.norm(z_before - z_after, 2, 1, keepdim=True))
            probs = vae.latentDifferent(z_before, z_after)
            train_pred_loss = ce(probs, y_train_batch)

            # Total loss & optimize
            train_loss = train_rec_loss + train_kld + train_pred_loss
            train_loss.backward()
            optimizer.step()

            try:
                y_np, prob_np = y_train_batch.cpu().detach().numpy(), probs.cpu().detach().numpy()
                train_auc = roc_auc_score(y_np, prob_np)
                train_auprc = average_precision_score(y_np, prob_np)
            except:
                train_auc, train_auprc = 0, 0
            write_summaries(summary_writer, iter, "train", {
                "total_loss": train_loss,
                "nelbo": train_nelbo,"rec_loss": train_rec_loss, "kld": train_kld,
                "ce_loss": train_pred_loss,
                "auc": train_auc, "auprc": train_auprc
            })

            if iter % args.iter_eval == 0:
                vae.eval()
                with torch.no_grad():
                    qm_dev, qv_dev = vae.encode(X_dev)
                    z_dev = vae.sample_gaussian(qm_dev, qv_dev)

                    # NELBO = (-)REC + KL
                    X_hat_dev = vae.decode(z_dev)
                    # dev_rec_loss = -torch.mean(-ce_logits(input=X_hat_logits_dev, target=X_dev).sum(-1), 0)
                    dev_rec_loss = mse(X_hat_dev, X_dev)
                    dev_kld = torch.mean(vae.kl_normal(qm_dev, qv_dev, vae.z_prior_m, vae.z_prior_v))
                    dev_nelbo = dev_rec_loss + dev_kld

                    # Pred loss
                    z_dev_before = z_dev[[i for i in range(z_dev.size()[0]) if i % 2 == 0], :]
                    z_dev_after = z_dev[[i for i in range(z_dev.size()[0]) if i % 2 == 1], :]
                    # dev_probs = torch.sigmoid(torch.norm(z_dev_before - z_dev_after, 2, 1, keepdim=True))
                    dev_probs = vae.latentDifferent(z_dev_before, z_dev_after)
                    dev_pred_loss = ce(dev_probs, y_dev)

                    # Total loss
                    dev_loss = dev_rec_loss + dev_kld + dev_pred_loss

                    try:
                        y_dev_np, prob_dev_np = y_dev.cpu().detach().numpy(), dev_probs.cpu().detach().numpy()
                        dev_auc = roc_auc_score(y_dev_np, prob_dev_np)
                        dev_auprc = average_precision_score(y_dev_np, prob_dev_np)
                    except:
                        dev_auc, dev_auprc = 0, 0

                write_summaries(summary_writer, iter, "dev", {
                    "total_loss": dev_loss,
                    "nelbo": dev_nelbo, "rec_loss": dev_rec_loss, "kld": dev_kld,
                    "ce_loss": dev_pred_loss,
                    "auc": dev_auc, "auprc": dev_auprc
                })
                if dev_auc > best_dev_auc:
                    best_dev_auc = dev_auc
                    file_path = os.path.join(save_dir, "best.pt")
                    state = vae.state_dict()
                    torch.save(state, file_path)
                    print("Best model saved to {}".format(file_path))
                    summary = "Best dev auc: {}".format(dev_auc)
                    with open(os.path.join(exp_folder, "best_stats.txt"), 'w') as fp:
                        fp.write(summary)
                    print(summary)

            pbar.set_postfix(
                train_loss="{:.2e}".format(train_loss),
                train_auc="%.4f" % train_auc,
                dev_loss="{:.2e}".format(dev_loss),
                dev_auc="%.4f" % dev_auc
            )
            iter += 1
            pbar.update(1)
