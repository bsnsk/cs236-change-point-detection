import os
import numpy as np
import torch
from tqdm import tqdm
from torch import nn, optim
from sklearn.metrics import roc_auc_score, average_precision_score
from train import write_summaries

import tensorflow as tf


def train(model, exp_folder, X_train, y_train, X_dev, y_dev,
          iter_max, iter_eval,
          batch_size, lr, reg):
    save_dir = os.path.join(exp_folder, "checkpoints")
    os.mkdir(save_dir)
    summary_writer = tf.summary.FileWriter(exp_folder)

    mse = nn.MSELoss()
    ce = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg)

    iter = 0
    best_dev_auc = 0
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
                z_train_batch = model.forward(X_train_batch)
                z_before = z_train_batch[range(0, z_train_batch.size()[0], 2), :]
                z_after = z_train_batch[range(1, z_train_batch.size()[0], 2), :]

                # model loss for latent representation
                train_latent_loss = -torch.mean(
                    model.logLikelihood(z_before)
                    + model.logLikelihood(z_after)
                ) / z_before.shape[-1]  # normalization

                # change point prediction loss
                # y_hat = torch.sigmoid(torch.norm(z_before-z_after, 2, 1, keepdim=True))
                y_hat = model.latentDifferent(z_before, z_after)
                train_pred_loss = ce(y_hat, y_train_batch)

                # combine together with normalization
                train_loss = train_latent_loss  + train_pred_loss

                train_loss.backward()
                optimizer.step()
                try:
                    y_np, prob_np = y_train_batch.cpu().detach().numpy(), y_hat.cpu().detach().numpy()
                    train_auc = roc_auc_score(y_np, prob_np)
                    train_auprc = average_precision_score(y_np, prob_np)
                except:  # TODO the fuck?
                    train_auc, train_auprc = 0, 0
                write_summaries(summary_writer, iter, "train", {
                    "total_loss": train_loss, "latent_loss": train_latent_loss, "ce_loss": train_pred_loss,
                    "auc": train_auc, "auprc": train_auprc
                })

                if iter % iter_eval == 0:
                    model.eval()
                    with torch.no_grad():
                        z_dev = model.forward(X_dev)
                        z_dev_before = z_dev[[i for i in range(z_dev.size()[0]) if i % 2 == 0], :]
                        z_dev_after = z_dev[[i for i in range(z_dev.size()[0]) if i % 2 == 1], :]

                        dev_latent_loss = -torch.mean(
                            model.logLikelihood(z_dev_before)
                            + model.logLikelihood(z_dev_after)
                        ) / z_dev.shape[-1]   # normalization

                        # dev_probs = torch.sigmoid(torch.norm(z_dev_before-z_dev_after, 2, 1, keepdim=True))
                        dev_probs = model.latentDifferent(z_dev_before, z_dev_after)
                        dev_pred_loss = ce(dev_probs, y_dev)

                        dev_loss = dev_latent_loss + dev_pred_loss
                        try:
                            y_dev_np, prob_dev_np = y_dev.cpu().detach().numpy(), dev_probs.cpu().detach().numpy()
                            dev_auc = roc_auc_score(y_dev_np, prob_dev_np)
                            dev_auprc = average_precision_score(y_dev_np, prob_dev_np)
                        except:
                            dev_auc, dev_auprc = 0, 0

                    write_summaries(summary_writer, iter, "dev", {
                        "total_loss": dev_loss, "latent_loss": dev_latent_loss, "ce_loss": dev_pred_loss,
                        "auc": dev_auc, "auprc": dev_auprc
                    })
                    if dev_auc > best_dev_auc:
                        best_dev_auc = dev_auc
                        file_path = os.path.join(save_dir, "best.pt")
                        state = model.state_dict()
                        torch.save(state, file_path)
                        print("Best model saved to {}".format(file_path))
                        # TODO better way to save / visualize summary?
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
