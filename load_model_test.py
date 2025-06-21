"""
Check weight loading and transferring weights to model.
"""

import torch

from src.models.vision_transformer import vit_huge

ckpt = torch.load(
    'weights/IN1K-vit.h.14-300e.pth.tar',
    map_location='cpu'
)
print(ckpt.keys())

ckpt_encoder = ckpt['encoder']

model = vit_huge(patch_size=14)
print('#'*30, 'Model', '#'*30, )
print(model)
print('#'*67)

for k, v in ckpt_encoder.items():
    model.state_dict()[k[len('module.'):]].copy_(v)

_ = model.cuda()

if __name__ == '__main__':
    rnd_tensor = torch.randn((1, 3, 224, 224)).cuda()
    
    print('Doing forward pass...')
    _ = model.eval()
    with torch.no_grad():
        outputs = model(rnd_tensor)

    print('Output shape')
    print(outputs.shape)