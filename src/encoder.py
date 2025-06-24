import torch
import torch.nn as nn
import timm


class SwinEncoder(nn.Module):
    def __init__(self, model_name='swin_tiny_patch4_window7_224'):
        super(SwinEncoder, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True, features_only=True)

        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Return a list of 4 feature maps
        feature_maps_list = self.model(x)
        new_feature_maps_list = []

        # Reshape output to (B, C, H, W)
        for feature_map in feature_maps_list:
            new_feature_maps_list.append(feature_map.permute(0, 3, 1, 2))

        return new_feature_maps_list
