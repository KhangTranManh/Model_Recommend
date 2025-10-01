#!/usr/bin/env python3
"""
Interactive Demo: Fashion Recommendation Personalization
This script provides an interactive way to test personalization features
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def interactive_personalization_demo():
    """
    Interactive demo where you can test personalization in real-time
    """
    print("🎮 Interactive Fashion Personalization Demo")
    print("=" * 50)
    
    try:
        from fashion_recommender.user.history import save_query, get_recent_embedding, get_history
        from fashion_recommender.api.recommendations import RecommendationEngine
        
        print("✅ System loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading system: {e}")
        return
    
    # Get user ID
    user_id = input("\n👤 Enter your user ID (or press Enter for 'demo_user'): ").strip()
    if not user_id:
        user_id = "demo_user"
    
    print(f"Welcome, {user_id}! Let's test personalization.")
    
    # Create recommendation engine
    rec_engine = RecommendationEngine()
    
    # Create dummy gallery for demo
    print("\n📁 Setting up demo gallery...")
    gallery_items = [
        "red_summer_dress.jpg", "blue_denim_jeans.jpg", "black_leather_boots.jpg",
        "white_cotton_shirt.jpg", "floral_midi_dress.jpg", "navy_blazer.jpg",
        "brown_leather_bag.jpg", "gold_jewelry_set.jpg", "pink_sweater.jpg",
        "black_high_heels.jpg", "casual_sneakers.jpg", "vintage_coat.jpg"
    ]
    
    gallery_embeddings = np.random.rand(len(gallery_items), 256).astype(np.float32)
    print(f"✅ Created gallery with {len(gallery_items)} fashion items")
    
    while True:
        print("\n" + "="*50)
        print("🎯 What would you like to test?")
        print("1. 🔍 Make a search (builds your profile)")
        print("2. 👆 Click on an item (shows interest)")
        print("3. 📋 View your search history") 
        print("4. 🎨 Get personalized recommendations")
        print("5. 🔥 See popular items")
        print("6. 📊 View your profile summary")
        print("7. ❌ Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == "1":
            # Search functionality
            print("\n🔍 SEARCH MODE")
            query = input("What fashion item are you looking for? ")
            
            if query:
                # Create search embedding (simulate different types of searches)
                search_embedding = np.random.rand(256).astype(np.float32)
                
                # Add some personality to different search types
                if any(word in query.lower() for word in ['dress', 'elegant', 'formal']):
                    search_embedding[:50] += 0.8  # Formal wear signal
                elif any(word in query.lower() for word in ['casual', 'jeans', 'sneakers']):
                    search_embedding[50:100] += 0.8  # Casual wear signal
                elif any(word in query.lower() for word in ['shoes', 'boots', 'heels']):
                    search_embedding[100:150] += 0.8  # Footwear signal
                
                # Save the search
                try:
                    rec_engine.track_user_interaction(
                        user_id=user_id,
                        interaction_type="search",
                        query_text=query,
                        embedding=search_embedding
                    )
                    print(f"✅ Saved search: '{query}'")
                    print("💡 This search will influence your future recommendations!")
                except Exception as e:
                    print(f"❌ Error saving search: {e}")
        
        elif choice == "2":
            # Item interaction
            print("\n👆 ITEM INTERACTION")
            print("Available items:")
            for i, item in enumerate(gallery_items, 1):
                print(f"  {i}. {item}")
            
            try:
                item_choice = int(input("\nEnter item number to click on: ")) - 1
                if 0 <= item_choice < len(gallery_items):
                    selected_item = gallery_items[item_choice]
                    item_embedding = gallery_embeddings[item_choice]
                    
                    rec_engine.track_user_interaction(
                        user_id=user_id,
                        interaction_type="click", 
                        item_path=selected_item,
                        embedding=item_embedding
                    )
                    print(f"✅ Clicked on: {selected_item}")
                    print("💡 This interaction strengthens your preference for similar items!")
                else:
                    print("❌ Invalid item number")
            except ValueError:
                print("❌ Please enter a valid number")
        
        elif choice == "3":
            # View history
            print("\n📋 YOUR SEARCH HISTORY")
            try:
                history = get_history(user_id, limit=10)
                if history:
                    print(f"Found {len(history)} interactions:")
                    for i, entry in enumerate(history[-5:], 1):  # Show last 5
                        entry_type = entry.get('type', 'unknown')
                        if entry_type == 'search':
                            print(f"  {i}. 🔍 Searched: '{entry.get('query', 'N/A')}'")
                        elif entry_type == 'click':
                            print(f"  {i}. 👆 Clicked: {entry.get('product', 'N/A')}")
                        else:
                            print(f"  {i}. 📌 {entry_type}: {entry.get('query', entry.get('product', 'N/A'))}")
                else:
                    print("No history found. Try making some searches first!")
            except Exception as e:
                print(f"❌ Error retrieving history: {e}")
        
        elif choice == "4":
            # Personalized recommendations
            print("\n🎨 YOUR PERSONALIZED RECOMMENDATIONS")
            try:
                recommendations = rec_engine.get_personalized_recommendations(
                    user_id=user_id,
                    gallery_embeddings=gallery_embeddings,
                    gallery_paths=gallery_items,
                    top_k=5
                )
                
                if recommendations:
                    print("Based on your searches and clicks, we recommend:")
                    for i, (score, item) in enumerate(recommendations, 1):
                        print(f"  {i}. {item} (similarity: {score:.3f})")
                else:
                    print("No personalized recommendations yet.")
                    print("💡 Make some searches and clicks to build your profile!")
            except Exception as e:
                print(f"❌ Error getting recommendations: {e}")
        
        elif choice == "5":
            # Popular recommendations
            print("\n🔥 POPULAR ITEMS")
            try:
                popular = rec_engine.get_popular_recommendations(top_k=5)
                if popular:
                    print("Most popular items across all users:")
                    for i, (score, item) in enumerate(popular, 1):
                        print(f"  {i}. {item}")
                else:
                    print("No popular items data available yet.")
            except Exception as e:
                print(f"❌ Error getting popular items: {e}")
        
        elif choice == "6":
            # Profile summary
            print(f"\n📊 PROFILE SUMMARY for {user_id}")
            try:
                analytics = rec_engine.get_user_analytics(user_id)
                if analytics:
                    # Handle both string and dict formats
                    if isinstance(analytics, str):
                        print(analytics)
                    elif isinstance(analytics, dict):
                        print("Your profile information:")
                        for key, value in analytics.items():
                            print(f"  {key}: {value}")
                    else:
                        print("Your profile information:")
                        print(str(analytics))
                else:
                    print("No profile data available yet.")
                    print("💡 Build your profile by making searches and interactions!")
            except Exception as e:
                print(f"❌ Error getting profile: {e}")
        
        elif choice == "7":
            print("\n👋 Thanks for testing the personalization system!")
            print("💾 All your interactions have been saved for future sessions.")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-7.")


def show_personalization_explanation():
    """
    Explain how the personalization system works
    """
    print("\n🧠 HOW PERSONALIZATION WORKS")
    print("=" * 40)
    print("""
🔍 SEARCH TRACKING:
   • Every search query you make is converted to an embedding vector
   • These vectors capture the semantic meaning of what you're looking for
   • The system learns your style preferences from your search patterns

👆 INTERACTION TRACKING:
   • When you click on items, the system notes which items interest you
   • Item embeddings are combined with your profile
   • This creates a more accurate picture of your taste

🎯 RECOMMENDATION GENERATION:
   • Your recent searches and clicks are averaged to create your "preference vector"
   • The system finds items in the gallery most similar to your preferences
   • Recommendations get better as you interact more

🔄 CONTINUOUS LEARNING:
   • Each interaction refines your profile
   • The system adapts to changes in your taste over time
   • Popular items serve as fallback when you're new

📊 PROFILE EVOLUTION:
   Cold Start → Basic Preferences → Rich Profile → Expert Recommendations
   """)


if __name__ == "__main__":
    print("🎮 Fashion Recommendation System - Interactive Demo")
    print("This demo lets you experience how personalization works in real-time.")
    
    # Show explanation first
    show_personalization_explanation()
    
    input("\nPress Enter to start the interactive demo...")
    
    # Run interactive demo
    interactive_personalization_demo()
    
    print(f"\n📝 Files created during this session:")
    print("• data/history.json - Your interaction history")
    print("• data/global_stats.json - System-wide statistics")
    print("\n🔄 Run the demo again to see how your profile has evolved!")