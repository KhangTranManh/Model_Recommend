#!/usr/bin/env python3
"""
Test script to verify profile summary fix
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_profile_summary():
    """Test that profile summary works without errors"""
    print("🧪 Testing Profile Summary Fix")
    print("=" * 40)
    
    try:
        from fashion_recommender.api.recommendations import RecommendationEngine
        
        # Create recommendation engine
        rec_engine = RecommendationEngine()
        
        # Test with a user ID
        test_user_id = "test_profile_user"
        
        print(f"👤 Testing profile for: {test_user_id}")
        
        # First, create some interactions for the user
        print("\n🔍 Creating test interactions...")
        search_embedding = np.random.rand(256).astype(np.float32)
        
        rec_engine.track_user_interaction(
            user_id=test_user_id,
            interaction_type="search",
            query_text="red dress",
            embedding=search_embedding
        )
        
        rec_engine.track_user_interaction(
            user_id=test_user_id,
            interaction_type="click",
            item_path="test_item.jpg",
            embedding=search_embedding
        )
        
        print("✅ Created test interactions")
        
        # Now test getting the profile analytics
        print("\n📊 Testing get_user_analytics...")
        analytics = rec_engine.get_user_analytics(test_user_id)
        
        print("✅ Got analytics result")
        print(f"📝 Analytics type: {type(analytics)}")
        
        # Test the handling logic from interactive_demo
        if analytics:
            print("\n🎯 Testing display logic:")
            if isinstance(analytics, str):
                print("📄 String format detected - displaying as text:")
                print(analytics)
            elif isinstance(analytics, dict):
                print("📋 Dictionary format detected - displaying as key-value pairs:")
                for key, value in analytics.items():
                    print(f"  {key}: {value}")
            else:
                print("📌 Other format detected - converting to string:")
                print(str(analytics))
        else:
            print("⚠️ No analytics data returned")
        
        print("\n✅ Profile summary test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in profile summary test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_profile_summary()