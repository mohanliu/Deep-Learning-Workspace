import torch
import torch.nn as nn


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()

        # setting hidden expanded dimension (bottleneck from the other perspective)
        hidden_dim = int(round(inp * expand_ratio))

        # trigger residual criteria:
        # 1. stride = 1
        # 2. in_channels == out_channels
        # Note: at each stage of bottleneck, the first block will be
        #   a `stride=s` (s in [1, 2]) & expanded width (out_channels > in_channels).
        #   Then the trailing blocks will be a `stride=1` and `in_channels=out_channels`
        #   layer repeatedly, where the `in_channels` will be set to the output of first
        #   block in this stage. Therefore, the trailing blocks will have residual connections.
        self.use_res_connect = stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pointwise
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=inp,
                        out_channels=hidden_dim,
                        kernel_size=1,
                        stride=1,
                        padding=0,
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
                # depthwise (set `groups == out_channels`)
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
                # pointwise & linear (no ReLU activation)
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=hidden_dim,
                        out_channels=oup,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False,
                    ),
                    nn.BatchNorm2d(oup),
                ),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


if __name__ == "__main__":
    import sys
    from rich.console import Console

    console = Console()

    stride = int(sys.argv[1]) if len(sys.argv) >= 2 else 2
    assert stride in [1, 2]

    if stride == 2:
        INPUT_CHANNELS = 16
        OUTPUT_CHANNELS = 24
    elif stride == 1:
        INPUT_CHANNELS = 24
        OUTPUT_CHANNELS = 24

    EXPANSION_FACTOR = 6
    RESOLUTION = 112

    m = InvertedResidual(
        inp=INPUT_CHANNELS,
        oup=OUTPUT_CHANNELS,
        stride=stride,
        expand_ratio=EXPANSION_FACTOR,
    )

    console.rule("Model summary")
    console.print(m)
    console.print("Use residual connection: {}".format(m.use_res_connect))

    console.rule("Weights")
    for k, v in m.named_parameters():
        console.print("{}: {}".format(k, str(v.shape)))

    console.rule("Forward pass")
    input_tensor = torch.randn(5, INPUT_CHANNELS, RESOLUTION, RESOLUTION)
    output_tensor = m(input_tensor)

    console.print("Input tensor shape: {}".format(input_tensor.shape))
    console.print("Output tensor shape: {}".format(output_tensor.shape))
