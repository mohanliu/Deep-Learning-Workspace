import torch.nn as nn


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=inp,
                        out_channels=hidden_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        dilation=1,
                        groups=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                )
            )
        layers.extend(
            [
                # dw
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                        dilation=1,
                        groups=hidden_dim,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                ),
                # pw-linear
                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=oup,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


if __name__ == "__main__":
    from rich.console import Console

    console = Console()

    m = InvertedResidual(inp=2, oup=4, stride=2, expand_ratio=6)

    console.rule("Model summary")
    console.print(m)

    console.rule("Weights")
    for k, v in m.named_parameters():
        console.print("{}: {}".format(k, str(v.shape)))
