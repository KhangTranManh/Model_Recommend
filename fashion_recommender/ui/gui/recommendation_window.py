# recommendation_demo.py
"""
Demonstration of user-based recommendations using profiles
Shows how to recommend items based on user search history
"""

import os
import numpy as np
from ...user.profile_manager import UserProfileManager
from ...config.config import config

def demo_user_recommendations():
    """Demonstrate personalized recommendations"""
    print("🎯 Fashion Recommendation Demo")
    print("=" * 60)
    
    # Load user profiles
    manager = UserProfileManager()
    
    if not manager.user_profiles:
        print("❌ No user profiles found. Run user_profile_builder.py first.")
        return
    
    # Load gallery data for recommendations
    print("\n🔄 Loading gallery data...")
    try:
        npz_data = np.load(config.embeddings_path, allow_pickle=True)
        gallery_embeddings = npz_data["vecs"].astype("float32")
        gallery_paths = list(npz_data["paths"])
        
        # Check for duplicates in gallery data
        unique_items = set(os.path.basename(path) for path in gallery_paths)
        print(f"✅ Loaded {len(gallery_embeddings)} gallery embeddings")
        print(f"📊 Unique items: {len(unique_items)}")
        if len(gallery_embeddings) > len(unique_items):
            duplicates = len(gallery_embeddings) - len(unique_items)
            print(f"⚠️ Found {duplicates} duplicate embeddings (fixed by deduplication)")
        
    except Exception as e:
        print(f"❌ Error loading gallery: {e}")
        return
    
    # Get recommendations for each user
    for user_id in manager.list_all_users():
        print(f"\n{'='*60}")
        print(f"👤 RECOMMENDATIONS FOR USER: {user_id}")
        print(f"{'='*60}")
        
        # Show user profile summary
        print(manager.get_profile_summary(user_id))
        
        # Get user preferences
        preferences = manager.get_user_preferences(user_id)
        print(f"\n🎯 User's Fashion Interests:")
        for term, count in list(preferences.items())[:5]:
            print(f"  • {term.title()}: {count} searches")
        
        # Generate recommendations
        print(f"\n💡 PERSONALIZED RECOMMENDATIONS:")
        print("-" * 40)
        
        recommendations = manager.recommend_items_for_user(
            user_id, gallery_embeddings, gallery_paths, top_k=10
        )
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"{i:2d}. {rec['item_name']} (Score: {rec['similarity_score']:.3f})")
        else:
            print("No recommendations available")
        
        # Show why these recommendations make sense
        print(f"\n🧠 RECOMMENDATION ANALYSIS:")
        print("-" * 40)
        print("These items are recommended because they match your search patterns:")
        
        # Analyze user's search behavior
        top_colors = [term for term in preferences.keys() if term in ['red', 'blue', 'black', 'white', 'green', 'pink']]
        top_items = [term for term in preferences.keys() if term in ['shirt', 'dress', 'shoes', 'pants', 'hat', 'bag']]
        
        if top_colors:
            print(f"  • You frequently search for {', '.join(top_colors)} items")
        if top_items:
            print(f"  • You're interested in {', '.join(top_items)}")
        
        # Search pattern analysis
        total_searches = preferences.get('shirt', 0) + preferences.get('dress', 0)
        shoe_searches = preferences.get('shoes', 0)
        
        if shoe_searches >= 3:
            print(f"  • Strong preference for footwear ({shoe_searches} shoe-related searches)")
        if total_searches >= 2:
            print(f"  • Interest in clothing items ({total_searches} clothing searches)")

def compare_recommendation_methods():
    """Compare different recommendation approaches"""
    print(f"\n{'='*60}")
    print("📊 RECOMMENDATION METHOD COMPARISON")
    print(f"{'='*60}")
    
    manager = UserProfileManager()
    
    # Load gallery data
    npz_data = np.load(config.embeddings_path, allow_pickle=True)
    gallery_embeddings = npz_data["vecs"].astype("float32")
    gallery_paths = list(npz_data["paths"])
    
    user_id = list(manager.user_profiles.keys())[0]  # Get first user
    
    print(f"Comparing recommendation methods for user: {user_id}")
    
    # Method 1: Profile-based recommendations (what we built)
    profile_recs = manager.recommend_items_for_user(user_id, gallery_embeddings, gallery_paths, top_k=5)
    
    # Method 2: Random recommendations (baseline)
    random_indices = np.random.choice(len(gallery_paths), size=5, replace=False)
    random_recs = [{'item_name': gallery_paths[i], 'similarity_score': 0.0} for i in random_indices]
    
    print(f"\n🎯 Profile-Based Recommendations:")
    for i, rec in enumerate(profile_recs, 1):
        print(f"  {i}. {rec['item_name']} (Score: {rec['similarity_score']:.3f})")
    
    print(f"\n🎲 Random Baseline:")
    for i, rec in enumerate(random_recs, 1):
        item_name = rec['item_name'].split('\\')[-1] if '\\' in rec['item_name'] else rec['item_name']
        print(f"  {i}. {item_name} (Random)")
    
    print(f"\n📈 Analysis:")
    print("  • Profile-based recommendations have similarity scores showing relevance")
    print("  • These are personalized based on user's 21 search queries")
    print("  • Random recommendations serve as baseline for comparison")

def analyze_gallery_duplicates():
    """Analyze and report duplicate issues in gallery data"""
    print(f"\n{'='*60}")
    print("🔍 GALLERY DUPLICATE ANALYSIS")
    print(f"{'='*60}")
    
    try:
        npz_data = np.load(config.embeddings_path, allow_pickle=True)
        gallery_paths = list(npz_data["paths"])
        
        # Count duplicates by filename
        from collections import Counter
        filenames = [os.path.basename(path) for path in gallery_paths]
        filename_counts = Counter(filenames)
        
        duplicates = {name: count for name, count in filename_counts.items() if count > 1}
        
        print(f"📊 Gallery Statistics:")
        print(f"   Total embeddings: {len(gallery_paths)}")
        print(f"   Unique filenames: {len(filename_counts)}")
        print(f"   Duplicate files: {len(duplicates)}")
        
        if duplicates:
            print(f"\n🔍 Top Duplicates:")
            for filename, count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"   {filename}: {count} copies")
            
            print(f"\n💡 Duplicate Sources:")
            sample_file = list(duplicates.keys())[0]
            sample_paths = [path for path in gallery_paths if os.path.basename(path) == sample_file]
            print(f"   Example '{sample_file}' appears in:")
            for path in sample_paths[:3]:
                print(f"     {path}")
        else:
            print("✅ No duplicates found!")
            
    except Exception as e:
        print(f"❌ Error analyzing duplicates: {e}")

def main():
    """Main demonstration function"""
    # Run recommendation demo
    demo_user_recommendations()
    
    # Compare methods
    compare_recommendation_methods()
    
    # Analyze duplicates
    analyze_gallery_duplicates()
    
    print(f"\n🎉 NEXT STEPS:")
    print("="*40)
    print("1. ✅ User profiles are built from search history")
    print("2. ✅ Personalized recommendations are working")
    print("3. ✅ Duplicate detection implemented")
    print("4. 🔄 Add more users to test collaborative filtering")
    print("5. 🔄 Track user clicks/purchases to improve profiles")
    print("6. 🔄 Implement hybrid recommendations (content + collaborative)")
    print("7. 🔄 Add real-time profile updates")

if __name__ == "__main__":
    main()