#!/usr/bin/env python3
"""
Rebuild Embeddings Script
Re-generates embeddings for a specific set of images (e.g., 100 images)
Creates new NPZ file and FAISS index.
"""

import sys
import os
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import faiss

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def rebuild_embeddings(image_dir, output_npz, output_index, checkpoint_path, batch_size=32, max_images=None):
    """
    Rebuild embeddings for images in a directory
    
    Args:
        image_dir: Directory containing images
        output_npz: Output NPZ file path
        output_index: Output FAISS index path
        checkpoint_path: Model checkpoint path
        batch_size: Batch size for embedding generation
        max_images: Maximum number of images to process (None = all)
    """
    from fashion_recommender.models.similarity import load_model
    from transformers import CLIPProcessor
    
    print("🔄 Rebuilding Fashion Embeddings")
    print("=" * 60)
    
    # Load model
    print(f"\n🤖 Loading model from: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor, cfg = load_model(checkpoint_path, device)
    print(f"✅ Model loaded on {device}")
    
    # Get image paths
    print(f"\n📁 Scanning images in: {image_dir}")
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(Path(image_dir).glob(f"*{ext}"))
        image_paths.extend(Path(image_dir).glob(f"*{ext.upper()}"))
    
    image_paths = sorted([str(p) for p in image_paths])
    
    if max_images:
        image_paths = image_paths[:max_images]
    
    print(f"✅ Found {len(image_paths)} images")
    
    if len(image_paths) == 0:
        print("❌ No images found!")
        return False
    
    # Generate embeddings
    print(f"\n🔄 Generating embeddings (batch_size={batch_size})...")
    
    all_embeddings = []
    max_length = cfg.get("max_length", 77)
    
    @torch.no_grad()
    def embed_batch(image_batch):
        """Embed a batch of images"""
        enc = processor(
            text=[""] * len(image_batch),
            images=image_batch,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        
        pv = enc["pixel_values"].to(device)
        ids = enc["input_ids"].to(device)
        am = enc["attention_mask"].to(device)
        
        img_embeds, _ = model(pv, ids, am)
        return img_embeds.cpu().numpy().astype("float32")
    
    # Process in batches
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Embedding"):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        
        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                batch_images.append(img)
            except Exception as e:
                print(f"⚠️ Error loading {img_path}: {e}")
                # Use blank image as fallback
                batch_images.append(Image.new("RGB", (224, 224), "white"))
        
        if batch_images:
            embeddings = embed_batch(batch_images)
            all_embeddings.append(embeddings)
    
    # Combine all embeddings
    all_embeddings = np.vstack(all_embeddings)
    print(f"\n✅ Generated {all_embeddings.shape[0]} embeddings")
    print(f"   Shape: {all_embeddings.shape}")
    
    # Save NPZ
    print(f"\n💾 Saving embeddings to: {output_npz}")
    os.makedirs(os.path.dirname(output_npz), exist_ok=True)
    np.savez(
        output_npz,
        vecs=all_embeddings,
        paths=np.array(image_paths, dtype=object)
    )
    print(f"✅ NPZ saved ({os.path.getsize(output_npz) / 1024 / 1024:.2f} MB)")
    
    # Build FAISS index
    print(f"\n🔍 Building FAISS index...")
    embedding_dim = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)  # Inner product (cosine similarity for normalized vectors)
    index.add(all_embeddings)
    
    # Save FAISS index
    print(f"💾 Saving FAISS index to: {output_index}")
    os.makedirs(os.path.dirname(output_index), exist_ok=True)
    faiss.write_index(index, output_index)
    print(f"✅ FAISS index saved ({os.path.getsize(output_index) / 1024 / 1024:.2f} MB)")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 Rebuild Complete!")
    print("=" * 60)
    print(f"📊 Statistics:")
    print(f"  • Total images: {len(image_paths)}")
    print(f"  • Embedding dimension: {embedding_dim}")
    print(f"  • NPZ file: {output_npz}")
    print(f"  • Index file: {output_index}")
    print(f"\n💡 Next steps:")
    print(f"  1. Update config to point to new files (or backup old ones)")
    print(f"  2. Run test_recommendation.py to test")
    print("=" * 60)
    
    return True


def main():
    """Main function with user prompts"""
    from fashion_recommender.config.config import config
    
    print("🎯 Fashion Embeddings Rebuild Tool")
    print("=" * 60)
    print("\nThis script will:")
    print("  1. Load images from a directory")
    print("  2. Generate embeddings using your trained model")
    print("  3. Create new NPZ and FAISS index files")
    print("=" * 60)
    
    # Get user input
    print("\n📂 Current gallery directory:", config.images_dir_path)
    use_current = input("Use current gallery? (y/n, default=y): ").strip().lower()
    
    if use_current in ['n', 'no']:
        image_dir = input("Enter image directory path: ").strip()
    else:
        image_dir = config.images_dir_path
    
    if not os.path.exists(image_dir):
        print(f"❌ Directory not found: {image_dir}")
        return False
    
    # Max images
    max_images_input = input("\nMaximum images to process (press Enter for all): ").strip()
    max_images = int(max_images_input) if max_images_input else None
    
    # Output paths - always use config paths
    print("\n📝 Output location:")
    print(f"  NPZ: {config.embeddings_path}")
    print(f"  Index: {config.index_path}")
    
    # Create backup option
    backup_suffix = input("\nCreate backup of existing files? Enter suffix (e.g., '_old' or press Enter to overwrite): ").strip()
    
    if backup_suffix:
        import shutil
        if os.path.exists(config.embeddings_path):
            backup_npz = config.embeddings_path.replace('.npz', f'{backup_suffix}.npz')
            shutil.copy(config.embeddings_path, backup_npz)
            print(f"✅ Backed up NPZ to: {backup_npz}")
        
        if os.path.exists(config.index_path):
            backup_index = config.index_path.replace('.index', f'{backup_suffix}.index')
            shutil.copy(config.index_path, backup_index)
            print(f"✅ Backed up index to: {backup_index}")
    
    output_npz = config.embeddings_path
    output_index = config.index_path
    
    # Batch size
    batch_size_input = input("\nBatch size (default=32): ").strip()
    batch_size = int(batch_size_input) if batch_size_input else 32
    
    # Confirm
    print("\n" + "=" * 60)
    print("📋 Configuration:")
    print(f"  Image directory: {image_dir}")
    print(f"  Max images: {max_images if max_images else 'All'}")
    print(f"  Output NPZ: {output_npz}")
    print(f"  Output Index: {output_index}")
    print(f"  Batch size: {batch_size}")
    print(f"  Model: {config.checkpoint_path}")
    print("=" * 60)
    
    confirm = input("\nProceed? (y/n): ").strip().lower()
    
    if confirm not in ['y', 'yes']:
        print("❌ Cancelled")
        return False
    
    # Run rebuild
    success = rebuild_embeddings(
        image_dir=image_dir,
        output_npz=output_npz,
        output_index=output_index,
        checkpoint_path=config.checkpoint_path,
        batch_size=batch_size,
        max_images=max_images
    )
    
    return success


if __name__ == "__main__":
    try:
        success = main()
        if not success:
            print("\n❌ Rebuild failed")
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nPress Enter to exit...")
    input()

