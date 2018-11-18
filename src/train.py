import os
import numpy as np
import torch
from tqdm import tqdm
from torch import nn, optim

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



