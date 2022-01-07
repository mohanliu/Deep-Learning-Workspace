import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    """Tiny modification from official implemntation from torchvision:

    https://github.com/pytorch/vision/blob/01dfa8ea81972bb74b52dc01e6a1b43b26b62020/torchvision/models/resnet.py#L86-L141
    """

    expansion: int = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        groups=1,
        base_width=64,
        dilation=1,
    ) -> None:
        super(Bottleneck, self).__init__()

        width = (
            int(planes * (base_width / 64.0)) * groups
        )  # by default, `width == planes`

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        # Note that all Convolution layers does not have bias!!

        # Firstly, a Conv1x1 to map input channels to hidden channels
        self.conv1 = nn.Conv2d(
            in_channels=inplanes,
            out_channels=width,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(width)

        # Secondly, a Conv3x3 with stride to downsize spatial size
        self.conv2 = nn.Conv2d(
            in_channels=width,
            out_channels=width,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(width)

        # Then, another Conv1x1 to map to expanded channels
        self.conv3 = nn.Conv2d(
            in_channels=width,
            out_channels=planes * self.expansion,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        # Same in-place ReLU layer used after all previous stages
        self.relu = nn.ReLU(inplace=True)

        # Downsample the input to match dimension for residual connection
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=inplanes,
                    out_channels=planes * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * self.expansion),
            )
        else:
            self.downsample = None

        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


if __name__ == "__main__":
    import sys
    from rich.console import Console

    console = Console()

    stage = sys.argv[1] if len(sys.argv) >= 2 else "conv3_x"
    assert stage in ["conv2_x", "conv3_x", "conv4_x", "conv5_x"]

    if stage == "conv2_x":
        PREV_PLANES = 64
        PREV_OUTPUT_SIZE = 56
        NUM_PLANES = 64
        BLOCK_REPEATS = 3
        FIRST_LAYER_STRIDE = 1
    elif stage == "conv3_x":
        PREV_PLANES = 64
        PREV_OUTPUT_SIZE = 56
        NUM_PLANES = 128
        BLOCK_REPEATS = 4
        FIRST_LAYER_STRIDE = 2
    elif stage == "conv4_x":
        PREV_PLANES = 512
        PREV_OUTPUT_SIZE = 28
        NUM_PLANES = 256
        BLOCK_REPEATS = 6
        FIRST_LAYER_STRIDE = 2
    elif stage == "conv5_x":
        PREV_PLANES = 1024
        PREV_OUTPUT_SIZE = 14
        NUM_PLANES = 512
        BLOCK_REPEATS = 3
        FIRST_LAYER_STRIDE = 2

    console.rule("Configurations")
    console.print("Stage: {}".format(stage))
    console.print("Previous planes: {}".format(PREV_PLANES))
    console.print("Previous output: {}".format(PREV_OUTPUT_SIZE))
    console.print("Number of planes: {}".format(NUM_PLANES))
    console.print("Number of repeated layers: {}".format(BLOCK_REPEATS))
    console.print("First layer stride: {}".format(FIRST_LAYER_STRIDE))

    console.rule("First layer of a stage (with downsample)")

    m_1 = Bottleneck(PREV_PLANES, NUM_PLANES, FIRST_LAYER_STRIDE)
    console.print(m_1)
    for k, v in m_1.named_parameters():
        if k.startswith("bn"):
            continue
        console.print("{}: {}".format(k, str(v.shape)))

    console.rule("Repeated layers of a stage (with expansion of 4 for bottleneck)")
    m_2 = Bottleneck(NUM_PLANES * 4, NUM_PLANES, 1)
    console.print(m_2)
    for k, v in m_2.named_parameters():
        if k.startswith("bn"):
            continue
        console.print("{}: {}".format(k, str(v.shape)))

    # Prepare forward pass
    BATCH_SIZE = 5
    console.rule("Forward pass")
    input_tensor = torch.randn(
        BATCH_SIZE, PREV_PLANES, PREV_OUTPUT_SIZE, PREV_OUTPUT_SIZE
    )
    console.print("Input tensor shape: {}".format(input_tensor.shape))

    output_tensor = m_1(input_tensor)
    console.print("First block output shape: {}".format(output_tensor.shape))

    for _ in range(BLOCK_REPEATS - 1):
        output_tensor = m_2(output_tensor)

    console.print("Final stage output shape: {}".format(output_tensor.shape))