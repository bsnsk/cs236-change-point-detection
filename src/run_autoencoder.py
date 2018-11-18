import torch
from data_loader import *
from autoencoder import AutoEncoder
from train import train
from torch.autograd import Variable

window_size = 25
X, _ = load_eeg('/Users/yongshangwu/Desktop/Projects/cs236/cs236-change-point-detection/data/EEG/EEG Eye State.arff.txt')
preprocessed_X = Variable(torch.Tensor(sliding_window(X, 25)), requires_grad=False)
print(preprocessed_X.size())
input_dim = preprocessed_X.shape[1]
auto_encoder = AutoEncoder(input_dim, [300, 300], 200)
print(auto_encoder)
train(auto_encoder, preprocessed_X)