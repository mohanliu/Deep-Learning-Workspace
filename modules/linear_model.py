import torch
import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self, ndim, init_mode="unit"):
        super(LinearModel, self).__init__()

        # helper parameters
        self._ndim = ndim
        self._init_mode = init_mode

        # Note: here we use `nn.Parameter` to make sure the parameters
        #   can be calculate gradients
        # Otherwise, the following settings can only allow forward pass
        # '''
        #   self._weight = torch.ones(ndim, 1)
        #   self._bias = torch.zeros(1)
        # '''
        if init_mode == "unit":
            # Initialize weight to be `1` and bias to be `0`, this is
            #   just a sum of elements.
            self._weight = nn.Parameter(torch.ones(ndim, 1))
            self._bias = nn.Parameter(torch.zeros(1))
        elif init_mode == "random":
            self._weight = nn.Parameter(torch.randn(ndim, 1))
            self._bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return x @ self._weight + self._bias


if __name__ == "__main__":
    NDIM = 8  # dimension of input and weights
    BS = 4  # batch size

    # initialize model
    lm = LinearModel(ndim=NDIM, init_mode="unit")
    print(list(lm.named_parameters()))

    # get input
    x = torch.randn(BS, NDIM)

    # get output
    y = lm(x)

    # check output
    input_sum_ = torch.sum(x, dim=1).unsqueeze(-1)
    diff_ = torch.sum(input_sum_ - y).detach().numpy()

    assert diff_ < 1e-6
