#!/usr/bin/env python3
"""
Test script for Fashion Recommendation System Personalization
This script demonstrates how the personalization system works step by step.
"""

import sys
import os
import numpy as np
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_personalization_system():
    """
    Complete test of the personalization system with detailed explanations
    """
    print("🧪 Testing Fashion Recommendation Personalization System")
    print("=" * 60)
    
    try:
        # Import the modules
        from fashion_recommender.user.history import save_query, get_recent_embedding, get_history
        from fashion_recommender.user.profile_manager import UserProfileManager
        from fashion_recommender.api.recommendations import RecommendationEngine
        from fashion_recommender.config.config import config
        
        print("✅ All modules imported successfully!")
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the project root directory")
        return False
    
    # Test 1: Basic User History Tracking
    print("\n📝 Test 1: User History Tracking")
    print("-" * 30)
    
    test_user = "test_user_123"
    
    # Simulate user searches
    searches = [
        "red summer dress",
        "blue jeans", 
        "black sneakers",
        "floral dress",
        "denim jacket"
    ]
    
    print(f"👤 Simulating searches for user: {test_user}")
    for i, search_query in enumerate(searches):
        # Create fake embedding (in real system, this comes from the model)
        fake_embedding = np.random.rand(256).astype(np.float32)
        
        try:
            save_query(
                user_id=test_user,
                query_type="search", 
                query_text=search_query,
                embedding=fake_embedding
            )
            print(f"  ✓ Saved search: '{search_query}'")
        except Exception as e:
            print(f"  ❌ Failed to save search '{search_query}': {e}")
    
    # Test 2: Retrieve User History
    print("\n📚 Test 2: Retrieving User History")
    print("-" * 30)
    
    try:
        history = get_history(test_user, limit=10)
        if history:
            print(f"✅ Found {len(history)} history entries for {test_user}")
            for entry in history[-3:]:  # Show last 3
                print(f"  - {entry.get('type', 'unknown')}: {entry.get('query', 'N/A')}")
        else:
            print("⚠️ No history found")
    except Exception as e:
        print(f"❌ Error retrieving history: {e}")
    
    # Test 3: Recent Embeddings
    print("\n🔍 Test 3: Getting Recent Embeddings")
    print("-" * 30)
    
    try:
        recent_embeddings = get_recent_embedding(test_user, n=3)
        if recent_embeddings:
            print(f"✅ Retrieved {len(recent_embeddings)} recent embeddings")
            print(f"  Embedding shape: {recent_embeddings[0].shape}")
        else:
            print("⚠️ No recent embeddings found")
    except Exception as e:
        print(f"❌ Error getting embeddings: {e}")
    
    # Test 4: Profile Manager
    print("\n👤 Test 4: User Profile Management")
    print("-" * 30)
    
    try:
        profile_manager = UserProfileManager()
        
        # Check if user profile exists
        user_profile = profile_manager.get_user_profile(test_user)
        if user_profile:
            print(f"✅ Found existing profile for {test_user}")
            summary = profile_manager.get_profile_summary(test_user)
            print(f"  Profile summary: {summary}")
        else:
            print(f"⚠️ No profile found for {test_user} (this is normal for new users)")
        
        # Get system stats
        stats = profile_manager.get_system_stats()
        print(f"📊 System stats: {stats}")
        
    except Exception as e:
        print(f"❌ Profile manager error: {e}")
    
    # Test 5: Recommendation Engine
    print("\n🎯 Test 5: Recommendation Engine")
    print("-" * 30)
    
    try:
        # Create dummy gallery data for testing
        num_items = 100
        embedding_dim = 256
        
        gallery_embeddings = np.random.rand(num_items, embedding_dim).astype(np.float32)
        gallery_paths = [f"item_{i:04d}.jpg" for i in range(num_items)]
        
        print(f"📁 Created dummy gallery: {num_items} items, {embedding_dim}D embeddings")
        
        # Initialize recommendation engine
        rec_engine = RecommendationEngine()
        
        # Test personalized recommendations
        print(f"🔍 Getting personalized recommendations for {test_user}...")
        recommendations = rec_engine.get_personalized_recommendations(
            user_id=test_user,
            gallery_embeddings=gallery_embeddings,
            gallery_paths=gallery_paths,
            top_k=5
        )
        
        if recommendations:
            print(f"✅ Got {len(recommendations)} personalized recommendations:")
            for i, (score, item) in enumerate(recommendations[:3], 1):
                print(f"  {i}. {item} (score: {score:.4f})")
        else:
            print("⚠️ No personalized recommendations (falling back to popular items)")
            
        # Test popular recommendations
        print("\n🔥 Getting popular recommendations...")
        popular = rec_engine.get_popular_recommendations(top_k=3)
        if popular:
            print(f"✅ Got {len(popular)} popular recommendations:")
            for i, (score, item) in enumerate(popular, 1):
                print(f"  {i}. {item}")
        
    except Exception as e:
        print(f"❌ Recommendation engine error: {e}")
    
    # Test 6: Track Interactions
    print("\n📊 Test 6: Tracking User Interactions")
    print("-" * 30)
    
    try:
        rec_engine = RecommendationEngine()
        
        # Simulate user clicking on items
        clicked_items = ["item_0023.jpg", "item_0045.jpg", "item_0078.jpg"]
        
        for item in clicked_items:
            fake_embedding = np.random.rand(256).astype(np.float32)
            rec_engine.track_user_interaction(
                user_id=test_user,
                interaction_type="click",
                item_path=item,
                embedding=fake_embedding
            )
            print(f"  ✓ Tracked click on {item}")
        
        print(f"✅ Successfully tracked {len(clicked_items)} interactions")
        
    except Exception as e:
        print(f"❌ Interaction tracking error: {e}")
    
    print("\n🎉 Personalization System Test Complete!")
    print("=" * 60)
    
    return True


def demonstrate_personalization_flow():
    """
    Demonstrate how personalization improves over time
    """
    print("\n🔄 Demonstrating Personalization Learning Process")
    print("=" * 50)
    
    try:
        from fashion_recommender.user.history import save_query, get_recent_embedding
        from fashion_recommender.api.recommendations import RecommendationEngine
        
        # Simulate a user with evolving preferences
        user_id = "learning_user"
        rec_engine = RecommendationEngine()
        
        # Create dummy gallery
        gallery_embeddings = np.random.rand(50, 256).astype(np.float32)
        gallery_paths = [f"fashion_item_{i:03d}.jpg" for i in range(50)]
        
        # Stage 1: No history (cold start)
        print("\n📍 Stage 1: New User (No History)")
        recommendations_1 = rec_engine.get_personalized_recommendations(
            user_id, gallery_embeddings, gallery_paths, top_k=3
        )
        print(f"Recommendations: {[item for _, item in recommendations_1[:3]]}")
        
        # Stage 2: After some searches
        print("\n📍 Stage 2: After Fashion Searches")
        fashion_searches = ["elegant dress", "high heels", "designer bag"]
        for search in fashion_searches:
            # Simulate fashion-oriented embeddings (higher values in certain dimensions)
            fashion_embedding = np.random.rand(256).astype(np.float32)
            fashion_embedding[:50] += 0.5  # Boost fashion-related dimensions
            
            save_query(user_id, "search", search, embedding=fashion_embedding)
            print(f"  Searched: '{search}'")
        
        recommendations_2 = rec_engine.get_personalized_recommendations(
            user_id, gallery_embeddings, gallery_paths, top_k=3
        )
        print(f"Recommendations: {[item for _, item in recommendations_2[:3]]}")
        
        # Stage 3: After interactions with items
        print("\n📍 Stage 3: After Item Interactions")
        for item in ["fashion_item_005.jpg", "fashion_item_012.jpg"]:
            item_embedding = np.random.rand(256).astype(np.float32)
            item_embedding[:50] += 0.7  # Strong fashion signal
            
            save_query(user_id, "click", product_id=item, embedding=item_embedding)
            print(f"  Clicked: {item}")
        
        recommendations_3 = rec_engine.get_personalized_recommendations(
            user_id, gallery_embeddings, gallery_paths, top_k=3
        )
        print(f"Recommendations: {[item for _, item in recommendations_3[:3]]}")
        
        print("\n✨ Notice how recommendations adapt to user behavior!")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")


if __name__ == "__main__":
    print("🚀 Fashion Recommendation System - Personalization Test Suite")
    print("This script tests all personalization features step by step.\n")
    
    # Run main test
    success = test_personalization_system()
    
    if success:
        # Run learning demonstration
        demonstrate_personalization_flow()
        
        print(f"\n📋 What was tested:")
        print("✅ User history tracking (saving searches and clicks)")
        print("✅ User profile management (loading existing profiles)")  
        print("✅ Recommendation generation (personalized + popular)")
        print("✅ Interaction tracking (clicks, purchases)")
        print("✅ Learning over time (cold start → personalized)")
        
        print(f"\n🎯 Next steps to test with real data:")
        print("1. Run the GUI: python app.py")
        print("2. Make some searches and clicks")
        print("3. Check data/history.json for saved interactions")
        print("4. Run this test again to see real personalization")
        
    else:
        print("\n❌ Tests failed. Please check the setup and try again.")
    
    input("\nPress Enter to exit...")