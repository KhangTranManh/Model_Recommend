# FashionCLIP.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

class FashionCLIP(nn.Module):
    """
    FashionCLIP model = CLIP backbone + 256-D projection heads.
    This aligns image and text embeddings into a shared space for fashion retrieval.
    """

    def __init__(self, model_name: str, embedding_dim: int = 256, use_projection=True, use_layer_norm=True):
        super().__init__()

        # Load pretrained CLIP
        self.clip = CLIPModel.from_pretrained(model_name)
        self.use_projection = use_projection

        if use_projection:
            clip_dim = self.clip.config.projection_dim

            if use_layer_norm:
                # Projection + normalization improves stability
                self.image_projection = nn.Sequential(
                    nn.Linear(clip_dim, embedding_dim),
                    nn.LayerNorm(embedding_dim)
                )
                self.text_projection = nn.Sequential(
                    nn.Linear(clip_dim, embedding_dim),
                    nn.LayerNorm(embedding_dim)
                )
            else:
                # Simple linear projections (no normalization)
                self.image_projection = nn.Linear(clip_dim, embedding_dim)
                self.text_projection = nn.Linear(clip_dim, embedding_dim)

            self.embedding_dim = embedding_dim
        else:
            # Directly use CLIP’s own projection layer
            self.image_projection = nn.Identity()
            self.text_projection = nn.Identity()
            self.embedding_dim = self.clip.config.projection_dim

    def forward(self, pixel_values, input_ids, attention_mask):
        """
        Forward pass — returns normalized image and text embeddings.
        """
        outputs = self.clip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Get CLIP embeddings
        image_embeds = outputs.image_embeds
        text_embeds  = outputs.text_embeds

        # Apply projection heads if enabled
        if self.use_projection:
            image_embeds = self.image_projection(image_embeds)
            text_embeds  = self.text_projection(text_embeds)

        # Normalize to unit vectors (important for cosine similarity)
        image_embeds = F.normalize(image_embeds, p=2, dim=1)
        text_embeds  = F.normalize(text_embeds,  p=2, dim=1)

        return image_embeds, text_embeds
