import torch
import torch.nn as nn
import math

from collections import OrderedDict
from src.models.vision_transformer import vit_huge 
from torchinfo import summary

def load_model(weights):
    model = vit_huge(patch_size=16, img_size=[448])
    print('#'*30, 'Model', '#'*30, )
    print(model)
    print('#'*67)

    if weights is not None:
        ckpt = torch.load(weights, map_location='cpu')
        print(ckpt.keys())
        ckpt_encoder = ckpt['encoder']
    
        for k, v in ckpt_encoder.items():
            model.state_dict()[k[len('module.'):]].copy_(v)

    model = model.eval()

    return model

class SimpleDecoder(nn.Module):
    def __init__(self, in_channels, nc=1):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, nc, kernel_size=3)
        )

    def forward(self, x):
        return self.decode(x)

class JepaSegmentation(nn.Module):
    def __init__(self, fine_tune=False, weights=None, num_classes=2):
        super(JepaSegmentation, self).__init__()

        self.backbone_model = load_model(weights=weights)

        self.num_classes = num_classes

        if fine_tune:
            for name, param in self.backbone_model.named_parameters():
                param.requires_grad = True
        else:
            for name, param in self.backbone_model.named_parameters():
                param.requires_grad = False

        self.decode_head = SimpleDecoder(
            in_channels=1280, nc=self.num_classes
        )

        self.model = nn.Sequential(OrderedDict([
            ('backbone', self.backbone_model),
            ('decode_head', self.decode_head)
        ]))

    def forward(self, x):
        # Backbone forward pass
        features = self.model.backbone(x)

        # Reshape patch tokens to (B, EmbeddingDim, patch_h, patch_w)
        B, N, D = features.shape
        tokenH = tokenW = int(math.sqrt(N))
        
        # Need to correctly resize and permute.
        x = features.view(B, tokenH, tokenW, D)
        x = x.permute(0, 3, 1, 2)  # (B, EmbeddingDim, patch_h, patch_w)

        # Decoder forward pass
        classifier_out = self.model.decode_head(x)
        return classifier_out
    

if __name__ == '__main__':
    from PIL import Image
    from torchvision import transforms
    import numpy as np

    input_size = 448

    transform = transforms.Compose([
        transforms.Resize(
            input_size, 
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5), 
            std=(0.5, 0.5, 0.5)
        )
    ])

    # Loading the pretrained model without classification head.
    # The following weights path should be relative to the directory
    # from where this module is being executed from.
    weights = 'weights/IN1K-vit.h.16-448px-300e.pth.tar'

    model = JepaSegmentation(fine_tune=False, weights=weights, num_classes=2)
    model.eval()
    print(model)

    random_image = Image.fromarray(np.ones(
        (input_size, input_size, 3), dtype=np.uint8)
    )
    x = transform(random_image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(x)
    
    print(outputs.shape)

    summary(
        model, 
        input_data=x,
        col_names=('input_size', 'output_size', 'num_params'),
        row_settings=['var_names'],
    )