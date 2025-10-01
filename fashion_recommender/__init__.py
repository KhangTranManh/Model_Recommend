"""
Fashion Recommender System

A comprehensive fashion recommendation system using CLIP embeddings 
and FAISS indexing for image similarity search and personalized recommendations.
"""

__version__ = "1.0.0"
__author__ = "Fashion Recommender Team"

# Import main components for easy access
from .config.config import config
from .models.FashionCLIP import FashionCLIP

__all__ = [
    "config",
    "FashionCLIP"
]
