import torch
import torch.nn as nn


class SRModel(nn.Module):
    """
    Simple CNN-based Super-Resolution model.

    Input:
        3 x 160 x 160

    Output:
        3 x 480 x 480
    """

    def __init__(self, scale_factor=3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(32, 32 * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.ReLU(inplace=True),
        )

        self.output = nn.Conv2d(
            32,
            3,
            kernel_size=5,
            padding=2
        )

    def forward(self, x):
        x = self.features(x)
        x = self.upsample(x)
        x = self.output(x)

        return x


if __name__ == "__main__":
    model = SRModel(scale_factor=3)

    test_input = torch.randn(1, 3, 160, 160)
    test_output = model(test_input)

    print("INPUT SHAPE :", test_input.shape)
    print("OUTPUT SHAPE:", test_output.shape)