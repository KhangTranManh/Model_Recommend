#!/usr/bin/env python3
"""
CLI interface for Fashion Recommendation System
Usage: python run_cli.py [options]
"""

import sys
import os
import argparse

def main():
    print("🔍 Fashion Recommendation System - CLI Mode")
    
    # Add the project root to Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    parser = argparse.ArgumentParser(description="Fashion Recommendation System CLI")
    parser.add_argument("--query", "-q", type=str, help="Text query for fashion search")
    parser.add_argument("--image", "-i", type=str, help="Path to image file for similarity search")
    parser.add_argument("--k", type=int, default=6, help="Number of results to return (default: 6)")
    parser.add_argument("--show", action="store_true", help="Display results with images")
    
    args = parser.parse_args()
    
    if not args.query and not args.image:
        print("❌ Please provide either --query or --image parameter")
        parser.print_help()
        return
    
    try:
        from fashion_recommender.models.similarity import search, embed_text, embed_one_image, load_model
        from fashion_recommender.config.config import DEFAULT_CHECKPOINT, DEFAULT_NPZ, DEFAULT_INDEX, DEFAULT_IMAGES_DIR
        import torch
        
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Loading model on {device}...")
        model, processor, cfg = load_model(DEFAULT_CHECKPOINT, device)
        
        # Perform search
        if args.query:
            print(f"🔍 Searching for: '{args.query}'")
            results = search(
                model=model, 
                processor=processor, 
                query_text=args.query,
                k=args.k,
                device=device,
                cfg=cfg,
                npz_path=DEFAULT_NPZ,
                index_path=DEFAULT_INDEX,
                images_dir=DEFAULT_IMAGES_DIR
            )
        else:  # args.image
            print(f"🖼️ Finding similar images to: {args.image}")
            results = search(
                model=model,
                processor=processor, 
                query_image=args.image,
                k=args.k,
                device=device,
                cfg=cfg,
                npz_path=DEFAULT_NPZ,
                index_path=DEFAULT_INDEX,
                images_dir=DEFAULT_IMAGES_DIR
            )
        
        # Display results
        print(f"\n🎯 Found {len(results)} similar items:")
        for i, (score, img_path) in enumerate(results, 1):
            print(f"  {i}. Score: {score:.4f} - {os.path.basename(img_path)}")
        
        if args.show:
            try:
                import matplotlib.pyplot as plt
                from PIL import Image
                
                fig, axes = plt.subplots(1, len(results), figsize=(15, 3))
                if len(results) == 1:
                    axes = [axes]
                    
                for i, (score, img_path) in enumerate(results):
                    img = Image.open(img_path)
                    axes[i].imshow(img)
                    axes[i].set_title(f"Score: {score:.3f}")
                    axes[i].axis('off')
                
                plt.tight_layout()
                plt.show()
                
            except ImportError:
                print("⚠️ Install matplotlib to display images: pip install matplotlib")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have:")
        print("  1. Installed required packages: pip install -r requirements.txt")
        print("  2. Model files in data/models/ directory") 
        print("  3. Embedding files in data/embeddings/ directory")

if __name__ == "__main__":
    main()