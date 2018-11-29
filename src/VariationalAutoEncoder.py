import torch
from torch import nn
from torch.nn import functional as F

class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_sizes, latent_dim, device):
        super(VariationalAutoEncoder, self).__init__()
        encoder_layers = [nn.Linear(input_dim, hidden_sizes[0]), nn.Sigmoid()]
        decoder_layers = [nn.Linear(hidden_sizes[0], input_dim)]
        for i in range(len(hidden_sizes)-1):
            d_in, d_out = hidden_sizes[i], hidden_sizes[i+1]
            encoder_layers.append(nn.Linear(d_in, d_out))
            encoder_layers.append(nn.Sigmoid())
            decoder_layers.insert(0, nn.Sigmoid())
            decoder_layers.insert(0, nn.Linear(d_out, d_in))
        encoder_layers.append(nn.Linear(hidden_sizes[-1], 2 * latent_dim))
        decoder_layers.insert(0, nn.Sigmoid())
        decoder_layers.insert(0, nn.Linear(latent_dim, hidden_sizes[-1]))

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)

        self.device = device

    def encode(self, x):
        h = self.encoder(x)
        m, v = self.gaussian_parameters(h, dim=1)
        return m, v

    def decode(self, z):
        return self.decoder(z)

    def gaussian_parameters(self, h, dim=-1):
        m, h = torch.split(h, h.size(dim) // 2, dim=dim)
        v = F.softplus(h) + 1e-8
        return m, v

    def sample_gaussian(self, m, v):
        epsilon = torch.randn(m.shape).to(self.device)
        z = m + epsilon * torch.sqrt(v)
        return z

    def kl_normal(self, qm, qv, pm, pv):
        element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
        kl = element_wise.sum(-1)
        return kl

    def nelbo(self, x):
        qm, qv = self.encode(x)
        z = self.sample_gaussian(qm, qv)

        logits = self.decoder(z)
        bce = nn.BCEWithLogitsLoss(reduction="none")

        rec = -torch.mean(-bce(input=logits, target=x).sum(-1), 0)
        kl = torch.mean(self.kl_normal(qm, qv, self.z_prior_m, self.z_prior_v))
        nelbo = rec + kl
        return nelbo, kl, rec

    def latentDifferent(self, z1, z2, qv):
        gaussian = torch.distributions.normal.Normal(
            z1,
            torch.sqrt(qv))
        pSame = 1 - torch.abs(1 - 2 * gaussian.cdf(z2))
        return 1 - torch.prod(pSame, dim=1, keepdim=True)