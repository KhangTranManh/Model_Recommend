# profile_integration_example.py
"""
Example of how to integrate user profiles into your main application
Shows how to add personalized recommendations to demo_recommend.py
"""

from user_profile_manager import UserProfileManager
import numpy as np
from fashion_recommender.config.config import config

def integrate_recommendations_into_gui():
    """
    Example code to add to your demo_recommend.py FashionRecommendationApp class
    """
    
    example_code = '''
# Add this to your FashionRecommendationApp.__init__() method:
self.profile_manager = UserProfileManager()

# Add this as a new method in FashionRecommendationApp class:
def get_personalized_recommendations(self, user_id="user1", max_recommendations=6):
    """Get personalized recommendations for a user"""
    try:
        if not hasattr(self, 'profile_manager') or not self.profile_manager.user_profiles:
            return []
        
        # Load gallery data if not already loaded
        if not hasattr(self, 'gallery_vecs') or self.gallery_vecs is None:
            npz = np.load(self.NPZ_PATH, allow_pickle=True)
            self.gallery_vecs = npz["vecs"].astype("float32")
            self.keep_paths = list(npz["paths"])
        
        # Get recommendations
        recommendations = self.profile_manager.recommend_items_for_user(
            user_id, self.gallery_vecs, self.keep_paths, top_k=max_recommendations
        )
        
        return recommendations
        
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return []

# Add this method to display personalized recommendations:
def display_personalized_recommendations(self):
    """Display personalized recommendations in the analytics panel"""
    try:
        recommendations = self.get_personalized_recommendations()
        
        if recommendations:
            self.recommendations_text.delete(1.0, tk.END)
            self.recommendations_text.insert(tk.END, "🎯 For You:\\n\\n")
            
            for i, rec in enumerate(recommendations[:5], 1):
                item_name = rec['item_name']
                score = rec['similarity_score']
                self.recommendations_text.insert(tk.END, f"{i}. {item_name}\\n")
                self.recommendations_text.insert(tk.END, f"   Match: {score:.1%}\\n\\n")
        else:
            self.recommendations_text.delete(1.0, tk.END)
            self.recommendations_text.insert(tk.END, "Build your profile by\\nsearching more items!")
            
    except Exception as e:
        print(f"Error displaying recommendations: {e}")

# Modify your update_analytics() method to include:
def update_analytics(self):
    """Update the analytics panel"""
    # ... existing code ...
    
    # Add personalized recommendations
    self.display_personalized_recommendations()
'''
    
    return example_code

def create_profile_updater():
    """
    Shows how to update profiles when users interact with items
    """
    
    updater_code = '''
# Add this method to update profiles when users click/buy items:
def update_user_profile_on_interaction(self, user_id, item_path, interaction_type="click"):
    """Update user profile when they interact with an item"""
    try:
        # Add interaction to history
        from user_history import save_query
        
        if interaction_type == "purchase":
            # Save purchase
            item_name = os.path.basename(item_path)
            save_query(user_id, "purchase", product_id=item_name)
            
            # Update analytics
            self.log_app_usage("item_purchase", {
                "item": item_name,
                "user_profile_exists": user_id in self.profile_manager.user_profiles
            })
        
        # Trigger profile rebuild (you might want to do this periodically instead)
        # self.rebuild_user_profiles_if_needed()
        
    except Exception as e:
        print(f"Error updating profile: {e}")

# Add this to your buy_item() method:
def buy_item(self, img_path):
    """Handle item purchase with profile update"""
    try:
        # ... existing buy logic ...
        
        # Update user profile
        self.update_user_profile_on_interaction(self.USER_ID, img_path, "purchase")
        
        # Refresh recommendations
        self.display_personalized_recommendations()
        
    except Exception as e:
        messagebox.showerror("Error", f"Purchase failed: {str(e)}")
'''
    
    return updater_code

def main():
    """Demonstrate integration examples"""
    print("🔗 USER PROFILE INTEGRATION GUIDE")
    print("="*60)
    
    # Check current system status
    manager = UserProfileManager()
    print(f"📊 Current System Status:")
    print(f"   Users with profiles: {len(manager.user_profiles)}")
    print(f"   Profile types: {set(p['profile_type'] for p in manager.user_profiles.values())}")
    
    # Show integration example
    print(f"\n📝 INTEGRATION CODE FOR demo_recommend.py:")
    print("-"*60)
    code = integrate_recommendations_into_gui()
    print(code)
    
    print(f"\n🔄 PROFILE UPDATE CODE:")
    print("-"*60)
    updater = create_profile_updater()
    print(updater)
    
    print(f"\n🎯 IMPLEMENTATION STEPS:")
    print("="*40)
    print("1. ✅ User profiles built from search history")
    print("2. 🔄 Add profile_manager to FashionRecommendationApp")
    print("3. 🔄 Add personalized recommendations to UI")
    print("4. 🔄 Update profiles on user interactions")
    print("5. 🔄 Periodically rebuild profiles from updated history")
    
    print(f"\n💡 RECOMMENDATION SYSTEM ARCHITECTURE:")
    print("="*50)
    print("Search History → User Profiles → Personalized Recommendations")
    print("      ↓               ↓                    ↓")
    print("  history.json  user_profiles.npz    recommendation_demo.py")
    print("      ↓               ↓                    ↓")
    print("Text Embeddings  Averaged Profiles   FAISS Similarity Search")
    
    print(f"\n🚀 NEXT FEATURES TO BUILD:")
    print("="*30)
    print("• Real-time profile updates")
    print("• Collaborative filtering (user-user similarity)")
    print("• Hybrid recommendations (content + collaborative)")
    print("• A/B testing for recommendation algorithms")
    print("• Recommendation explanation system")
    print("• Cold start recommendations for new users")

if __name__ == "__main__":
    main()