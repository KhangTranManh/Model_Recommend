#!/usr/bin/env python3
"""
Interactive Recommendation Test
Type your searches and purchases, get personalized recommendations with images.
"""

import sys
import os
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def display_recommendations(recommendations, title="Recommendations", max_display=8):
    """Display recommended images in a grid without duplicates"""
    try:
        from fashion_recommender.config.config import resolve_image_path
        
        # Remove duplicates while preserving order
        seen = set()
        unique_items = []
        for score, path in recommendations:
            # Use basename as unique identifier
            basename = os.path.basename(path)
            if basename not in seen:
                seen.add(basename)
                unique_items.append((score, path))
                if len(unique_items) >= max_display:
                    break
        
        if len(unique_items) == 0:
            print("⚠️ No items to display")
            return
        
        # Create grid layout
        cols = 4
        rows = (len(unique_items) + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Flatten axes array for easier iteration
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes_flat = axes.flatten()
        
        # Display each image
        for idx, (score, path) in enumerate(unique_items):
            ax = axes_flat[idx]
            
            # Resolve and load image
            try:
                img_path = resolve_image_path(path)
                img = Image.open(img_path)
                ax.imshow(img)
                
                # Set title with score and filename
                filename = os.path.basename(path)
                ax.set_title(f"{idx + 1}. {filename}\nScore: {score:.4f}", 
                           fontsize=10)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading\n{os.path.basename(path)}", 
                       ha='center', va='center')
                ax.set_title(f"{idx + 1}. Error", fontsize=10)
            
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(len(unique_items), len(axes_flat)):
            axes_flat[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("⚠️ matplotlib or PIL not available - skipping image display")
    except Exception as e:
        print(f"⚠️ Error displaying images: {e}")


def interactive_search(model, processor, device, cfg, user_id="interactive_user"):
    """Interactive search and recommendation session"""
    from fashion_recommender.models.similarity import embed_text
    from fashion_recommender.user.history import save_query
    from fashion_recommender.config.config import config
    from fashion_recommender.api.recommendations import RecommendationEngine
    
    print("\n" + "=" * 60)
    print("🛍️ INTERACTIVE FASHION RECOMMENDATION")
    print("=" * 60)
    print(f"👤 User: {user_id}")
    print("\nInstructions:")
    print("  • Type search queries (e.g., 'red dress', 'blue jeans')")
    print("  • Type 'purchase: <item_name>' to mark items as purchased")
    print("  • Type 'recommend' to see personalized recommendations")
    print("  • Type 'quit' to exit")
    print("=" * 60 + "\n")
    
    # Load gallery data
    print("📁 Loading gallery data...")
    embeddings_path = config.embeddings_path
    npz = np.load(embeddings_path, allow_pickle=True)
    gallery_embeddings = npz["vecs"].astype("float32")
    gallery_paths = list(npz["paths"])
    print(f"✅ Loaded {len(gallery_embeddings)} items\n")
    
    # Initialize recommendation engine
    rec_engine = RecommendationEngine(model=model, processor=processor, config=cfg)
    
    search_count = 0
    purchase_count = 0
    
    while True:
        user_input = input("💬 Your input: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        # Handle purchase
        if user_input.lower().startswith('purchase:'):
            item_name = user_input[9:].strip()
            
            # Find matching item in gallery
            matching_items = [p for p in gallery_paths if item_name.lower() in os.path.basename(p).lower()]
            
            if matching_items:
                selected_item = matching_items[0]
                
                # Get embedding for the purchased item
                item_idx = gallery_paths.index(selected_item)
                item_embedding = gallery_embeddings[item_idx]
                
                # Save as purchase
                save_query(
                    user_id=user_id,
                    query_type="purchase",
                    product_id=os.path.basename(selected_item),
                    embedding=item_embedding
                )
                
                purchase_count += 1
                print(f"✅ Purchased: {os.path.basename(selected_item)}")
                print(f"   (Total purchases: {purchase_count})\n")
            else:
                print(f"❌ Item '{item_name}' not found in gallery")
                print(f"   Available items: {len(gallery_paths)}")
                print(f"   Try searching first to see available items\n")
        
        # Handle recommendation request
        elif user_input.lower() in ['recommend', 'rec', 'show']:
            print("\n🎯 Generating personalized recommendations...")
            
            if search_count == 0 and purchase_count == 0:
                print("⚠️ No search or purchase history yet!")
                print("   Try searching for some items first.\n")
                continue
            
            # Get personalized recommendations
            recommendations = rec_engine.get_personalized_recommendations(
                user_id=user_id,
                gallery_embeddings=gallery_embeddings,
                gallery_paths=gallery_paths,
                top_k=16  # Get more to handle deduplication
            )
            
            if recommendations:
                print(f"✅ Top personalized recommendations based on your profile:\n")
                for i, (score, path) in enumerate(recommendations[:8], 1):
                    print(f"  {i}. {os.path.basename(path)} (score: {score:.4f})")
                
                # Display images
                print("\n🖼️ Displaying recommendations...")
                display_recommendations(
                    recommendations, 
                    f"Personalized for {user_id} ({search_count} searches, {purchase_count} purchases)",
                    max_display=8
                )
                
                # Show analytics
                try:
                    analytics = rec_engine.get_user_analytics(user_id)
                    if analytics:
                        print("\n📊 Your Profile Analytics:")
                        if isinstance(analytics, str):
                            print(analytics)
                        elif isinstance(analytics, dict):
                            for key, value in analytics.items():
                                print(f"  {key}: {value}")
                except:
                    pass
                
                print()
            else:
                print("⚠️ Could not generate recommendations\n")
        
        # Handle search query
        else:
            search_query = user_input
            
            print(f"🔍 Searching for: '{search_query}'...")
            
            # Generate embedding
            text_embedding = embed_text(
                model=model,
                processor=processor,
                query=search_query,
                device=device,
                cfg=cfg
            )
            
            # Save to history
            save_query(
                user_id=user_id,
                query_type="search",
                query_text=search_query,
                embedding=text_embedding[0]
            )
            
            search_count += 1
            
            # Show search results
            import faiss
            index = faiss.read_index(config.index_path)
            sims, idxs = index.search(text_embedding, 8)
            
            # Deduplicate results
            seen = set()
            unique_results = []
            for idx, score in zip(idxs[0], sims[0]):
                basename = os.path.basename(gallery_paths[idx])
                if basename not in seen:
                    seen.add(basename)
                    unique_results.append((score, gallery_paths[idx]))
            
            print(f"✅ Found {len(unique_results)} results:\n")
            for i, (score, path) in enumerate(unique_results[:5], 1):
                print(f"  {i}. {os.path.basename(path)} (score: {score:.4f})")
            
            print(f"\n   Saved to your profile (Total searches: {search_count})")
            print(f"   Tip: Type 'recommend' to see personalized suggestions\n")


def main():
    """Main interactive test"""
    print("🎯 Fashion Recommendation System - Interactive Test")
    print("=" * 60)
    
    try:
        # Import modules
        from fashion_recommender.models.similarity import load_model
        from fashion_recommender.config.config import config
        
        print("✅ Modules imported successfully\n")
        
        # Check required files
        print("📂 Checking required files:")
        model_path = config.checkpoint_path
        embeddings_path = config.embeddings_path
        index_path = config.index_path
        
        files_ok = True
        for name, path in [
            ("Model", model_path),
            ("Embeddings", embeddings_path),
            ("Index", index_path)
        ]:
            exists = os.path.exists(path)
            print(f"  {name}: {'✅' if exists else '❌ MISSING'}")
            if not exists:
                files_ok = False
        
        if not files_ok:
            print("\n❌ Missing required files. Please ensure:")
            print("  - data/models/fashion_clip_best.pt")
            print("  - data/embeddings/gallery_embeddings.npz")
            print("  - data/embeddings/gallery_ip.index")
            return False
        
        # Load model
        print(f"\n🤖 Loading model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, processor, cfg = load_model(model_path, device)
        print(f"✅ Model loaded on {device}")
        
        # Start interactive session
        interactive_search(model, processor, device, cfg)
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    
    if not success:
        print("\n❌ Common issues:")
        print("  • Missing model/data files in data/ directory")
        print("  • Missing dependencies: pip install -r requirements.txt")
        print("  • Wrong working directory (run from project root)")
    
    print("\nPress Enter to exit...")
    input()
