# Code adapted from: https://github.com/naoto0804/pytorch-AdaIN
import torch.nn as nn


class SwinDecoder(nn.Module):
    def __init__(self):
        super(SwinDecoder, self).__init__()
        self.n_channels=int(96)
        self.model = nn.Sequential(
                        # Block 1: HxW → 2Hx2W
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(self.n_channels, self.n_channels // 2, 3, 1, 0),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(self.n_channels // 2, self.n_channels // 2, 3, 1, 0),
                        nn.ReLU(inplace=True),
                        nn.Upsample(scale_factor=2, mode='nearest'),

                        # Block 2: 2Hx2W → 4Hx4W
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(self.n_channels // 2, self.n_channels // 4, 3, 1, 0),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(self.n_channels // 4, self.n_channels // 4, 3, 1, 0),
                        nn.ReLU(inplace=True),
                        nn.Upsample(scale_factor=2, mode='nearest'),

                        # Block 3: 4Hx4W → 8Hx8W
                        #nn.ReflectionPad2d(1),
                        #nn.Conv2d(self.n_channels // 4, self.n_channels // 8, 3, 1, 0),
                        #nn.ReLU(inplace=True),
                        #nn.ReflectionPad2d(1),
                        #nn.Conv2d(self.n_channels // 8, self.n_channels // 8, 3, 1, 0),
                        #nn.ReLU(inplace=True),
                        #nn.Upsample(scale_factor=2, mode='nearest'),


                        # Block 4: 8Hx8W → 16Hx16W
                        #nn.ReflectionPad2d(1),
                        #nn.Conv2d(self.n_channels // 8, self.n_channels // 16, 3, 1, 0),
                        #nn.ReLU(inplace=True),
                        #nn.ReflectionPad2d(1),
                        #nn.Conv2d(self.n_channels // 16, self.n_channels // 16, 3, 1, 0),
                        #nn.ReLU(inplace=True),
                        #nn.Upsample(scale_factor=2, mode='nearest'),

                        ## Block 5: 16Hx16W → 32Hx32W
                        #nn.ReflectionPad2d(1),
                        #nn.Conv2d(self.n_channels // 16, self.n_channels // 32, 3, 1, 0),
                        #nn.ReLU(inplace=True),
                        #nn.ReflectionPad2d(1),
                        #nn.Conv2d(self.n_channels // 32, self.n_channels // 32, 3, 1, 0),
                        #nn.ReLU(inplace=True),
                        #nn.Upsample(scale_factor=2, mode='nearest'),


                        # Output layer
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(self.n_channels // 4, 3, 3, 1, 0),
        )

    def forward(self, x):
        return self.model(x)





class SwinDecoder_old(nn.Module):
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
