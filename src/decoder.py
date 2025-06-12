# Code obtained from: https://github.com/naoto0804/pytorch-AdaIN
import torch.nn as nn


class SwinDecoder(nn.Module):
    def __init__(self):
        super(SwinDecoder, self).__init__()
        self.model = nn.Sequential(
                                nn.ReflectionPad2d((1, 1, 1, 1)),
                                nn.Conv2d(512, 256, (3, 3)),
                                nn.ReLU(),
                                nn.Upsample(scale_factor=2, mode='nearest'),
                                nn.ReflectionPad2d((1, 1, 1, 1)),
                                nn.Conv2d(256, 256, (3, 3)),
                                nn.ReLU(),
                                nn.ReflectionPad2d((1, 1, 1, 1)),
                                nn.Conv2d(256, 256, (3, 3)),
                                nn.ReLU(),
                                nn.ReflectionPad2d((1, 1, 1, 1)),
                                nn.Conv2d(256, 256, (3, 3)),
                                nn.ReLU(),
                                nn.ReflectionPad2d((1, 1, 1, 1)),
                                nn.Conv2d(256, 128, (3, 3)),
                                nn.ReLU(),
                                nn.Upsample(scale_factor=2, mode='nearest'),
                                nn.ReflectionPad2d((1, 1, 1, 1)),
                                nn.Conv2d(128, 128, (3, 3)),
                                nn.ReLU(),
                                nn.ReflectionPad2d((1, 1, 1, 1)),
                                nn.Conv2d(128, 64, (3, 3)),
                                nn.ReLU(),
                                nn.Upsample(scale_factor=2, mode='nearest'),
                                nn.ReflectionPad2d((1, 1, 1, 1)),
                                nn.Conv2d(64, 64, (3, 3)),
                                nn.ReLU(),
                                nn.ReflectionPad2d((1, 1, 1, 1)),
                                nn.Conv2d(64, 3, (3, 3)),
                            )

    def forward(self, x):
        return self.model(x)
