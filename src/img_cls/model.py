"""
Building a linear classifier model.

To test the file, execute as:
python -m src.img_cls.model
"""

import torch
import torch.nn as nn

from collections import OrderedDict
from src.models.vision_transformer import vit_huge

def load_model(weights):
    ckpt = torch.load(weights, map_location='cpu')
    print(ckpt.keys())

    ckpt_encoder = ckpt['encoder']

    model = vit_huge(patch_size=14)
    print('#'*30, 'Model', '#'*30, )
    print(model)
    print('#'*67)

    for k, v in ckpt_encoder.items():
        model.state_dict()[k[len('module.'):]].copy_(v)

    model = model.eval()

    return model

class LinearClassifier(nn.Module):
    def __init__(self, num_classes=10, fine_tune=False, weights=None):
        super(LinearClassifier, self).__init__()

        if weights is not None:
            backbone_model = load_model(weights=weights)
        else:
            backbone_model = vit_huge(patch_size=14)

        self.model = torch.nn.Sequential(OrderedDict([
            ('backbone', backbone_model),
            ('head', torch.nn.Linear(
                in_features=1280, out_features=num_classes, bias=True
            ))
        ]))
    
        if not fine_tune:
            for params in self.model.backbone.parameters():
                params.requires_grad = False

    def forward(self, x):
        backbone_out = self.model.backbone(x)
        avg_features = backbone_out.mean(dim=1)
        out = self.model.head(avg_features)

        return out


if __name__ == '__main__':
    from PIL import Image
    from torchvision import transforms
    from torchinfo import summary

    import numpy as np

    sample_size = 224

    # Define image transformation
    transform = transforms.Compose([
        transforms.Resize(
            sample_size, 
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.CenterCrop(sample_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5), 
            std=(0.5, 0.5, 0.5)
        )
    ])

    # Loading the pretrained model without classification head.
    # The following weights path should be relative to the directory
    # from where this module is being executed from.
    weights = 'weights/IN1K-vit.h.14-300e.pth.tar'

    # Testing forward pass.
    pil_image = Image.fromarray(np.ones((
        sample_size, sample_size, 3), dtype=np.uint8)
    )
    model_input = transform(pil_image).unsqueeze(0)

    model = LinearClassifier(num_classes=10, fine_tune=False, weights=weights)

    summary(
        model,
        input_data=model_input,
        col_names=('input_size', 'output_size', 'num_params'),
        row_settings=['var_names']
    )