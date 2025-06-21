import torch
import cv2

from src.models.vision_transformer import vit_huge
from torch.nn.functional import cosine_similarity
from torchvision import transforms

device = 'cuda'

img_transform = transforms.Compose([
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])

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

    model = model.to(device).eval()

    return model


def get_embeddings(image_path, model):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float32')

    image = torch.tensor(image) / 255.
    image = torch.permute(image, (2, 0, 1))
    image = img_transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        embed = model(image)
    
    avg_embed = embed.mean(dim=1)

    print(f"Output shape: {embed.shape}")
    print(f"Mean shape: {avg_embed.shape}")

    return avg_embed


def check_cosine_similarity(embed_1, embed_2):
    similarity = cosine_similarity(embed_1, embed_2)
    
    return similarity


weights_path = 'weights/IN1K-vit.h.14-300e.pth.tar'
model = load_model(weights=weights_path)

embed_1 = get_embeddings('input/image_1.jpg', model)
embed_2 = get_embeddings('input/image_4.jpg', model)

similarity = check_cosine_similarity(embed_1, embed_2)

print(similarity)