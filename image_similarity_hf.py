"""
Script to compute image similarity with I-JEPA.
"""

from PIL import Image
from torch.nn.functional import cosine_similarity
from transformers import AutoModel, AutoProcessor

import torch

def load_model(model_id):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).eval()

    return model, processor


def get_embeddings(image_path, model, processor):
    image = Image.open(image_path)

    inputs = processor(image, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    print(f"Outputs shape: {outputs.last_hidden_state.shape}")
    hidden_states = outputs.last_hidden_state.mean(dim=1)
    print(f"Mean shape: {hidden_states.shape}")

    return hidden_states


def check_cosine_similarity(embed_1, embed_2):
    similarity = cosine_similarity(embed_1, embed_2)
    
    return similarity


model_id = 'facebook/ijepa_vith14_1k'
model, processor = load_model(model_id)


embed_1 = get_embeddings('input/image_3.jpg', model, processor)
embed_2 = get_embeddings('input/image_4.jpg', model, processor)

similarity = cosine_similarity(embed_1, embed_2)
print(similarity)