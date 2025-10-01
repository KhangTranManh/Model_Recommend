
"""
Simple runner for the Fashion Recommendation System
Usage: python run_demo.py
"""

import sys
import os

def main():
    print("🚀 Starting Fashion Recommendation System...")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required")
        return
        
    # Import and run the main application
    try:
        # Add the project root to Python path
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, project_root)
        
        from fashion_recommender.ui.gui.main_window import FashionRecommendationApp
        import tkinter as tk
        
        # Create and run the GUI application
        root = tk.Tk()
        app = FashionRecommendationApp(root)
        print("✅ Fashion Recommendation System GUI started successfully!")
        root.mainloop()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all required packages are installed:")
        print("  pip install -r requirements.txt")
        print("\nOr install individually:")
        print("  pip install torch transformers pillow numpy faiss-cpu")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Check that model files exist in data/models/ directory")

if __name__ == "__main__":
    main()