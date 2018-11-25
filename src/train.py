import os
import numpy as np
import torch
from tqdm import tqdm
from torch import nn, optim
from sklearn.metrics import roc_auc_score, average_precision_score

# TODO clean this shit up...
def train_on_new_loss(model, exp_folder, X_train, y_train, X_dev, y_dev,
                      iter_max, iter_eval,
                      batch_size, lr, reg,
                      ):
    save_dir = os.path.join(exp_folder, "checkpoints")
    os.mkdir(save_dir)

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
                z_train_batch = model.encode(X_train_batch)
                X_hat_batch = model.decode(z_train_batch)

                train_rec_loss = mse(X_hat_batch, X_train_batch)

                z_before = z_train_batch[[i for i in range(z_train_batch.size()[0]) if i % 2 == 0], :]
                z_after = z_train_batch[[i for i in range(z_train_batch.size()[0]) if i % 2 == 1], :]
                probs = torch.sigmoid(torch.norm(z_before-z_after, 2, 1, keepdim=True))
                train_pred_loss = ce(probs, y_train_batch)

                train_loss = train_rec_loss + train_pred_loss
                train_loss.backward()
                optimizer.step()
                try:
                    train_auc = roc_auc_score(y_train_batch.detach().numpy(), probs.detach().numpy())
                except:
                    train_auc = 0

                if iter % iter_eval == 0:
                    model.eval()
                    with torch.no_grad():
                        z_dev = model.encode(X_dev)
                        X_hat_dev = model.decode(z_dev)

                        dev_rec_loss = mse(X_hat_dev, X_dev)
                        z_dev_before = z_dev[[i for i in range(z_dev.size()[0]) if i % 2 == 0], :]
                        z_dev_after = z_dev[[i for i in range(z_dev.size()[0]) if i % 2 == 1], :]
                        dev_probs = torch.sigmoid(torch.norm(z_dev_before-z_dev_after, 2, 1, keepdim=True))
                        dev_pred_loss = ce(dev_probs, y_dev)
                        dev_loss = dev_rec_loss + dev_pred_loss
                        try:
                            dev_auc = roc_auc_score(y_dev.detach().numpy(), dev_probs.detach().numpy())
                        except:
                            dev_auc = 0

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

def train(model, exp_folder, X_train, y_train, X_dev, y_dev,
          iter_max, iter_eval,
          batch_size, lr, reg,
          ):
    save_dir = os.path.join(exp_folder, "checkpoints")
    os.mkdir(save_dir)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg)

    iter = 0
    best_dev_loss = float("+inf")
    train_size = X_train.shape[0]
    indices = np.arange(train_size)

    with tqdm(total=iter_max) as pbar:
        while iter <= iter_max:
            np.random.shuffle(indices)
            for batch_start in np.arange(0, train_size, batch_size):
                model.train()
                optimizer.zero_grad()

                batch_indices = indices[batch_start: batch_start + batch_size]
                X_train_batch = X_train[batch_indices]
                X_hat_batch = model(X_train_batch)

                train_loss = criterion(X_hat_batch, X_train_batch)
                train_loss.backward()
                optimizer.step()

                if iter % iter_eval == 0:
                    model.eval()
                    with torch.no_grad():
                        X_dev_hat = model(X_dev)
                        dev_loss = criterion(X_dev_hat, X_dev)
                    if dev_loss < best_dev_loss:
                        best_dev_loss = dev_loss
                        file_path = os.path.join(save_dir, "best.pt")
                        state = model.state_dict()
                        torch.save(state, file_path)
                        print("Best model saved to {}".format(file_path))
                        # TODO better way to save / visualize summary?
                        summary = "Best dev loss: {}, train loss: {}\n".format(dev_loss, train_loss)
                        with open(os.path.join(exp_folder, "best_stats.txt"), 'w') as fp:
                            fp.write(summary)
                        print(summary)

                pbar.set_postfix(
                    train_loss="{:.2e}".format(train_loss),
                    dev_loss="{:.2e}".format(dev_loss)
                )
                iter += 1
                pbar.update(1)



