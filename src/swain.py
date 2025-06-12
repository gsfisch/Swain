# Majority of the code here was obtained from: https://github.com/naoto0804/pytorch-AdaIN
import torch
import torch.nn as nn
import encoder
from function import adaptive_instance_normalization as adain
from function import calc_mean_std


# (Sw)in Transformer (A)da(IN) neural network
class Swain(nn.Module):
    def __init__(self, encoder, decoder):
        super(Swain, self).__init__()
        self.encoder = encoder  # Swin encoder
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()

        # Project swin output (768 channels) to 512 channels for decoder
        self.input_proj = nn.Conv2d(768, 512, kernel_size=1)

        # Project features used for loss back to original Swin feature shapes
        self.loss_proj = nn.Conv2d(512, 768, kernel_size=1)

    def encode_with_intermediate(self, x):
        return self.encoder(x)  # returns list of 4 features

    def encode(self, x):
        return self.encoder(x)[-1]  # deepest feature for AdaIN

    def calc_content_loss(self, input, target):
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + self.mse_loss(input_std, target_std)

    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1

        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)  # [B, 768, H/32, W/32]

        print("On AdaIN: ")
        print("content: ")
        print(content_feat.shape)
        print("Style feat (last):")
        print(style_feats[-1].shape)
        # AdaIN works on deepest features
        t = adain(content_feat, style_feats[-1])
        t = alpha * t + (1 - alpha) * content_feat
        print("output shape:")
        print(t.shape)

        # Project to decoder input size
        t_proj = self.input_proj(t)  # [B, 768, H/32, W/32] -> [B, 512, H/32, W/32]
        #g_t = self.decoder(t_proj)   # generated stylized image
        g_t = self.decoder(t_proj)   # generated stylized image

        print("On decoder")
        print(g_t.shape)

        return g_t

        # Re-encode stylized image
        g_t_feats = self.encode_with_intermediate(g_t)


        # Compute losses
        # Project back features for fair comparison
        g_t_deep = self.loss_proj(self.encode(g_t))
        loss_c = self.calc_content_loss(g_t_deep, t)

        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        return loss_c, loss_s
