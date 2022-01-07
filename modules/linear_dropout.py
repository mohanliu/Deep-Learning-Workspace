import torch.nn as nn
from torch import Tensor


class LinearDropoutModel(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        dropout=0.5,
    ):
        super(LinearDropoutModel, self).__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        return x

    def _reset_parameters(self):
        self.apply(_init_weights)


def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


if __name__ == "__main__":
    m = LinearDropoutModel(5, 1)
    for k, v in m.named_parameters():
        print("{}: {}".format(k, v))
