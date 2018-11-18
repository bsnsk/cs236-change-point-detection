import numpy as np
from tqdm import tqdm
from torch import nn, optim

def train(model, X_train, # y_train, X_dev, y_dev,
          batch_size=500,
          lr=1e-1,
          iter_max=20000,
          reg=0,
          ):
    train_size = X_train.shape[0]
    indices = np.arange(train_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
    iter = 0
    with tqdm(total=iter_max) as pbar:
        while iter <= iter_max:
            np.random.shuffle(indices)
            for batch_start in np.arange(0, train_size, batch_size):
                optimizer.zero_grad()

                batch_indices = indices[batch_start: batch_start + batch_size]
                X_train_batch = X_train[batch_indices]
                X_hat_batch = model(X_train_batch)

                train_loss = criterion(X_hat_batch, X_train_batch)
                train_loss.backward()
                optimizer.step()

                iter += 1
                pbar.set_postfix(train_loss='{:.2e}'.format(train_loss))
                pbar.update(1)



