# FashionCLIP.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

class FashionCLIP(nn.Module):
    """CLIP backbone + 256-D projection heads for image/text embeddings."""
    def __init__(self, model_name: str, embedding_dim: int = 256):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        clip_dim = self.clip.config.projection_dim

        self.image_projection = nn.Sequential(
            nn.Linear(clip_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(clip_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.embedding_dim = embedding_dim

    def forward(self, pixel_values, input_ids, attention_mask):
        outputs = self.clip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        image_embeds = self.image_projection(outputs.image_embeds)
        text_embeds  = self.text_projection(outputs.text_embeds)

        # L2-normalize for cosine/inner-product similarity
        image_embeds = F.normalize(image_embeds, p=2, dim=1)
        text_embeds  = F.normalize(text_embeds,  p=2, dim=1)
        return image_embeds, text_embeds
