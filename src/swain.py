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

    def encode(self, x):
        return self.encoder(x)  # returns list of 4 features

    def calc_content_loss(self, input, target):
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + self.mse_loss(input_std, target_std)

    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1

        content = content.to(torch.float32)
        style = style.to(torch.float32)

        style_feats = self.encode(style)
        content_feat = self.encode(content)


        # AdaIN
        t = adain(content_feat[0], style_feats[0])
        t = alpha * t + (1 - alpha) * content_feat[0]

        g_t = self.decoder(t)   # generated stylized image

        # Re-encode stylized image
        g_t_feats = self.encode(g_t)


        loss_c = self.calc_content_loss(g_t_feats[0], t)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])

        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])

        return g_t, loss_c, loss_s


        # Compute losses
        # Project back features for fair comparison
        g_t_deep = self.encode(g_t)
        loss_c = self.calc_content_loss(g_t_deep, t)

        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])

        return g_t
        return loss_c, loss_s
