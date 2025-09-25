
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
        from demo_recommend import main as run_app
        run_app()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all required files are in the same directory:")
        print("  - demo_recommend.py")
        print("  - last.py") 
        print("  - user_history.py")
        print("  - FashionCLIP.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()