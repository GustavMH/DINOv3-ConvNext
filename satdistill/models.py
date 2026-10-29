"""timm backbones with their own input normalisation and the last-layer feature map used by the probes."""
import timm
import torch
import torch.nn as nn


class Backbone(nn.Module):
    def __init__(self, name=None, checkpoint=None):
        """Released weights by timm name, or a student saved by satdistill.distill."""
        super().__init__()
        saved = torch.load(checkpoint, map_location="cpu") if checkpoint else None
        self.model = timm.create_model(saved["student_name"] if saved else name, pretrained=saved is None,
                                       num_classes=0)
        cfg = dict(self.model.pretrained_cfg)
        if saved:
            self.model.load_state_dict(saved["student"])
            cfg.update(saved["normalize"])
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1))
        self.is_vit = hasattr(self.model, "blocks")

    def dense_features(self, x):
        """(B, C, H/16, W/16) for a ViT, (B, C, H/32, W/32) for a ConvNeXt."""
        x = (x - self.mean) / self.std
        if self.is_vit:
            return self.model.forward_intermediates(x, indices=1, norm=True, intermediates_only=True,
                                                    output_fmt="NCHW")[-1]
        return self.model.forward_features(x)