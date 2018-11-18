from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_sizes, latent_dim):
        super(AutoEncoder, self).__init__()
        encoder_layers = [nn.Linear(input_dim, hidden_sizes[0]), nn.Sigmoid()]
        decoder_layers = [nn.Linear(hidden_sizes[0], input_dim)]
        for i in range(len(hidden_sizes)-1):
            d_in, d_out = hidden_sizes[i], hidden_sizes[i+1]
            encoder_layers.append(nn.Linear(d_in, d_out))
            encoder_layers.append(nn.Sigmoid())
            decoder_layers.insert(0, nn.Sigmoid())
            decoder_layers.insert(0, nn.Linear(d_out, d_in))
        encoder_layers.append(nn.Linear(hidden_sizes[-1], latent_dim))
        decoder_layers.insert(0, nn.Sigmoid())
        decoder_layers.insert(0, nn.Linear(latent_dim, hidden_sizes[-1]))

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def encode(self, x):
        return self.encoder(x)
