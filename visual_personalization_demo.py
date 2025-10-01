#!/usr/bin/env python3
"""
Visual Personalization Demo
Shows how recommendations change as user profile evolves
"""

import sys
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def visual_personalization_demo():
    """
    Visual demonstration of how personalization evolves
    """
    print("📊 Visual Personalization Evolution Demo")
    print("=" * 40)
    
    try:
        from fashion_recommender.api.recommendations import RecommendationEngine  
        from fashion_recommender.user.history import save_query
        import matplotlib.pyplot as plt
        
        print("✅ Modules loaded successfully")
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Try: pip install matplotlib")
        return
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Create test scenario
    rec_engine = RecommendationEngine()
    user_id = "visual_demo_user"
    
    # Create themed gallery items (simulate different fashion categories)
    categories = {
        'dresses': np.random.rand(10, 256) + np.array([0.8, 0.1, 0.1] + [0]*253),  # High in "dress" dimension
        'casual': np.random.rand(10, 256) + np.array([0.1, 0.8, 0.1] + [0]*253),   # High in "casual" dimension  
        'formal': np.random.rand(10, 256) + np.array([0.1, 0.1, 0.8] + [0]*253),   # High in "formal" dimension
        'shoes': np.random.rand(10, 256) + np.array([0.1, 0.1, 0.1, 0.8] + [0]*252) # High in "shoes" dimension
    }
    
    gallery_embeddings = np.vstack([cat_emb.astype(np.float32) for cat_emb in categories.values()])
    gallery_paths = []
    for cat, emb in categories.items():
        gallery_paths.extend([f"{cat}_{i:02d}.jpg" for i in range(len(emb))])
    
    print(f"📁 Created themed gallery: {len(gallery_paths)} items across 4 categories")
    
    # Simulation stages
    stages = [
        {
            'name': 'New User (Cold Start)',
            'searches': [],
            'description': 'No history - shows popular items'
        },
        {
            'name': 'Casual Interest', 
            'searches': [
                ('jeans and t-shirt', np.array([0.2, 0.9, 0.1] + [0]*253)),
                ('comfortable sneakers', np.array([0.1, 0.8, 0.1, 0.7] + [0]*252)),
            ],
            'description': 'User shows interest in casual wear'
        },
        {
            'name': 'Formal Transition',
            'searches': [
                ('business attire', np.array([0.1, 0.3, 0.9] + [0]*253)),
                ('professional dress', np.array([0.9, 0.1, 0.8] + [0]*253)),
                ('office shoes', np.array([0.1, 0.1, 0.7, 0.8] + [0]*252)),
            ],
            'description': 'User needs formal wear - preferences shifting'
        },
        {
            'name': 'Established Profile',
            'searches': [
                ('elegant evening wear', np.array([0.8, 0.1, 0.9] + [0]*253)),
                ('designer heels', np.array([0.1, 0.1, 0.8, 0.9] + [0]*252)),
            ],
            'description': 'Clear preference for formal/elegant items'
        }
    ]
    
    # Track recommendations through each stage
    stage_recommendations = []
    
    for stage_num, stage in enumerate(stages):
        print(f"\n📍 Stage {stage_num + 1}: {stage['name']}")
        print(f"   {stage['description']}")
        
        # Add searches for this stage
        for search_text, search_embedding in stage['searches']:
            save_query(
                user_id=user_id,
                query_type="search",
                query_text=search_text, 
                embedding=search_embedding.astype(np.float32)
            )
            print(f"   🔍 Searched: '{search_text}'")
        
        # Get recommendations for this stage
        try:
            recommendations = rec_engine.get_personalized_recommendations(
                user_id=user_id,
                gallery_embeddings=gallery_embeddings,
                gallery_paths=gallery_paths,
                top_k=5
            )
            
            stage_recommendations.append(recommendations)
            
            print("   🎯 Top recommendations:")
            for i, (score, item) in enumerate(recommendations[:3], 1):
                category = item.split('_')[0]
                print(f"     {i}. {item} ({category}) - score: {score:.3f}")
        
        except Exception as e:
            print(f"   ❌ Error getting recommendations: {e}")
            stage_recommendations.append([])
    
    # Create visualization
    try:
        print(f"\n📊 Creating visualization...")
        
        # Analyze recommendation categories over time
        category_evolution = {cat: [] for cat in categories.keys()}
        
        for stage_recs in stage_recommendations:
            stage_counts = {cat: 0 for cat in categories.keys()}
            
            for score, item in stage_recs[:5]:  # Top 5 recommendations
                category = item.split('_')[0]
                if category in stage_counts:
                    stage_counts[category] += 1
            
            for cat in categories.keys():
                category_evolution[cat].append(stage_counts[cat])
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Category distribution over time
        stage_names = [stage['name'] for stage in stages]
        x_pos = range(len(stage_names))
        
        bottom = np.zeros(len(stage_names))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, (category, counts) in enumerate(category_evolution.items()):
            ax1.bar(x_pos, counts, bottom=bottom, label=category.title(), color=colors[i % len(colors)])
            bottom += counts
        
        ax1.set_xlabel('User Journey Stages')
        ax1.set_ylabel('Number of Recommendations')  
        ax1.set_title('How Personalization Adapts to User Preferences Over Time')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(stage_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Recommendation scores over time
        if stage_recommendations:
            for i, stage_recs in enumerate(stage_recommendations):
                if stage_recs:
                    scores = [score for score, _ in stage_recs[:5]]
                    positions = [i] * len(scores)
                    ax2.scatter(positions, scores, alpha=0.7, s=50)
        
        ax2.set_xlabel('User Journey Stages')
        ax2.set_ylabel('Recommendation Confidence Score')
        ax2.set_title('Recommendation Quality Improvement Over Time') 
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(stage_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('personalization_evolution.png', dpi=150, bbox_inches='tight')
        print("✅ Visualization saved as 'personalization_evolution.png'")
        
        # Show the plot
        plt.show()
        
    except Exception as e:
        print(f"❌ Visualization error: {e}")
    
    # Print summary
    print(f"\n📋 PERSONALIZATION JOURNEY SUMMARY")
    print("=" * 40)
    
    for i, (stage, recs) in enumerate(zip(stages, stage_recommendations)):
        print(f"\n{i+1}. {stage['name']}")
        print(f"   Searches: {len(stage['searches'])}")
        if recs:
            categories_recommended = {}
            for _, item in recs[:5]:
                cat = item.split('_')[0]
                categories_recommended[cat] = categories_recommended.get(cat, 0) + 1
            
            print("   Top categories recommended:")
            for cat, count in sorted(categories_recommended.items(), key=lambda x: x[1], reverse=True):
                print(f"     • {cat.title()}: {count} items")
        
    print(f"\n💡 Key Insights:")
    print("• Cold start: Random/popular recommendations")
    print("• Early interactions: System begins to learn preferences")
    print("• Profile building: Recommendations become more targeted")
    print("• Mature profile: Highly personalized, confident recommendations")
    
    return True


if __name__ == "__main__":
    print("📊 Fashion Recommendation System - Visual Personalization Demo")
    print("This demo shows how recommendations evolve as the system learns about a user.\n")
    
    success = visual_personalization_demo()
    
    if success:
        print(f"\n🎯 What this demo showed:")
        print("✅ Cold start problem and how it's handled")
        print("✅ How user searches build a preference profile")
        print("✅ Adaptation to changing user interests")
        print("✅ Visual representation of personalization evolution")
        
        print(f"\n📁 Files created:")
        print("• personalization_evolution.png - Visual chart")
        print("• Updated user history in data/history.json")
        
    else:
        print("❌ Demo failed. Check dependencies and try again.")
    
    input("\nPress Enter to exit...")