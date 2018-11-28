import torch
from torch import nn


class AdditiveCouplingLayer(nn.Module):
    def __init__(self, num_dims, partition, mlp):
        super(AdditiveCouplingLayer, self).__init__()
        assert(partition in ['even', 'odd'])
        self.partition = partition
        self._I1 = lambda xs: xs[:, 0::2]
        self._I2 = lambda xs: xs[:, 1::2]
        if partition == 'odd':
            self._I1, self._I2 = self._I2, self._I1
        self.add_module("mlp", mlp)

    def _combine(self, part1, part2, flag):
        def combineEvenAndOdd(part1, part2):  # part1-even, part2-odd
            cols = []
            for k in range(part2.shape[1]):
                cols.append(part1[:, k])
                cols.append(part2[:, k])
            if part1.shape[1] > part2.shape[1]:
                cols.append(part2[:, -1])
            return cols
        if flag == "even":  # part1 is even, and part2 is odd
            cols = combineEvenAndOdd(part1, part2)
        else:  # part2 is even, and part 1 is odd
            cols = combineEvenAndOdd(part2, part1)
        return torch.stack(cols, dim=1)

    def forward(self, x):
        return self._combine(
            self._I1(x),
            self._I2(x) + self.mlp(self._I1(x)),
            self.partition,
        )

    def inverse(self, z):
        return self._combine(
            self._I1(z),
            self._I2(z) - self.mpl(self._I1(z)),
            self.partition,
        )


class NICEModel(nn.Module):
    def _MLPLayers(self, num_dims, hidden_sizes):
        layers = [nn.Linear(num_dims, hidden_sizes[0]), nn.ReLU()]
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], num_dims))
        return nn.Sequential(*layers)

    def __init__(self, input_dim, hidden_sizes):
        super(NICEModel, self).__init__()
        assert(input_dim % 2 == 0)  # an assumption to make life easier
        self.dim = input_dim
        self.hidden_sizes = hidden_sizes

        self.couplingLayers = [
            AdditiveCouplingLayer(
                input_dim, 'odd', self._MLPLayers(self.dim // 2, hidden_sizes)
            ),
            AdditiveCouplingLayer(
                input_dim, 'even', self._MLPLayers(self.dim // 2, hidden_sizes)
            ),
            AdditiveCouplingLayer(
                input_dim, 'odd', self._MLPLayers(self.dim // 2, hidden_sizes)
            ),
            AdditiveCouplingLayer(
                input_dim, 'even', self._MLPLayers(self.dim // 2, hidden_sizes)
            ),
        ]
        self.scalingParameter = nn.Parameter(torch.ones(input_dim))

    def forward(self, x):
        z = x
        for layer in self.couplingLayers:
            z = layer(z)
        z = z * torch.exp(self.scalingParameter)
        return z

    def inverse(self, z):
        x = z
        with torch.no_grad():
            x = z * torch.reciprocal(torch.exp(self.scalingParameter))
            for i in range(len(self.couplingLayers) - 1, -1, -1):
                x = self.couplingLayers[i].inverse(x)
        return x

    # standard Gaussian as prior
    def logLikelihood(self, z):
        gaussian = torch.distributions.normal.Normal(torch.zeros(z.shape),
                                                     torch.ones(z.shape))
        log_prob = torch.sum(gaussian.log_prob(z), dim=-1)
        return log_prob
