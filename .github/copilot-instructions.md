# FashionCLIP AI Agent Instructions

## Project Overview
This is a fashion image similarity search system using CLIP embeddings with FAISS indexing. The system enables both text-to-image and image-to-image similarity search through a Gradio web interface.

## Architecture Components

### Core Models
- **FashionCLIP.py**: Simple CLIP backbone with 256-D projection heads (baseline model)
- **EnhancedFashionCLIP** (in last.py): Advanced version with configurable projection and layer normalization
- Both models normalize embeddings for cosine similarity search

### Key Files & Structure
- **last.py**: Main Gradio demo application with enhanced model and UI
- **demo_retrieval.py**: Command-line interface for batch processing and embedding generation
- **fashion_clip_best.pt/fashion_clip_final.pt**: Trained model checkpoints with config metadata
- **gallery_embeddings.npz**: Pre-computed image embeddings and file paths
- **gallery_ip.index**: FAISS inner-product index for fast similarity search
- **pic/**: Image gallery directory (10000.jpg, 10001.jpg, etc.)

## Development Patterns

### Model Loading Convention
```python
# Always load with config validation
ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
config = ckpt.get("config", {})
model_name = config.get("model_name", "openai/clip-vit-base-patch32")
embedding_dim = config.get("embedding_dim", 256)
```

### Embedding Generation
- Use `@torch.no_grad()` decorator for inference
- Always normalize embeddings: `F.normalize(embeds, p=2, dim=1)`
- Use mixed precision: `torch.amp.autocast('cuda', enabled=(device.type=="cuda"))`
- Batch processing for efficiency (see demo_retrieval.py)

### FAISS Index Management
```python
# Fallback pattern for missing index files
try:
    index = faiss.read_index(INDEX_PATH)
except Exception:
    vecs = npz["vecs"].astype("float32")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
```

### Path Handling for Gradio
- Images must be in allowed directories or temp folders
- Use `copy_image_to_temp()` helper for gallery display
- Always include `allowed_paths` parameter in `demo.launch()`

## Common Workflows

### Running Similarity Search
```bash
# Text query
python demo_retrieval.py --query_text "red floral summer dress" --k 6 --show

# Image query  
python demo_retrieval.py --query_image "path/to/image.jpg" --k 6 --show

# Web interface
python last.py
```

### Model Training Considerations
- Save checkpoints with full config metadata
- Include projection head settings: `use_projection_heads`, `use_layer_norm`
- Store processor settings: `max_length`, `image_size`

## Critical Dependencies
- torch (with CUDA support)
- transformers (CLIPProcessor, CLIPModel)
- faiss-cpu or faiss-gpu
- gradio (for web interface)
- PIL (image processing)

## Error Prevention
- Always validate image inputs with `load_and_validate_image()`
- Check file existence before processing
- Use proper exception handling for FAISS operations
- Ensure tensor device consistency (CPU/CUDA)

## Performance Notes
- Pre-compute embeddings for large galleries (see demo_retrieval.py)
- Use FAISS IndexFlatIP for inner product similarity
- Batch embedding generation for efficiency
- Consider image size limits for memory management