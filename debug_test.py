#!/usr/bin/env python3
"""
Test script to debug import issues
Run this to check if all imports work correctly
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

def test_imports():
    """Test all major imports"""
    print("🔍 Testing Fashion Recommender System Imports...")
    print("=" * 50)
    
    try:
        print("✓ Testing config import...")
        from fashion_recommender.config.config import config
        print(f"  ✓ Config loaded, project root: {config.PROJECT_ROOT}")
        
        print("✓ Testing model imports...")
        from fashion_recommender.models.FashionCLIP import FashionCLIP
        from fashion_recommender.models.similarity import load_model
        print("  ✓ Model imports successful")
        
        print("✓ Testing user imports...")
        from fashion_recommender.user.history import save_query
        from fashion_recommender.user.profile_manager import UserProfileManager
        print("  ✓ User imports successful")
        
        print("✓ Testing GUI imports...")
        from fashion_recommender.ui.gui.main_window import FashionRecommendationApp
        print("  ✓ GUI imports successful")
        
        print("\n🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False

def check_files():
    """Check if required files exist"""
    print("\n📁 Checking required files...")
    print("=" * 50)
    
    required_files = [
        "data/models/fashion_clip_best.pt",
        "data/models/fashion_clip_final.pt", 
        "data/embeddings/gallery_embeddings.npz",
        "data/embeddings/gallery_ip.index",
        "assets/images/gallery"
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = Path(file_path)
        if full_path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ❌ {file_path} (missing)")
            all_exist = False
    
    return all_exist

def check_dependencies():
    """Check if required packages are installed"""
    print("\n📦 Checking dependencies...")
    print("=" * 50)
    
    required_packages = [
        "torch", "transformers", "numpy", "PIL", "faiss", "tkinter"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == "PIL":
                import PIL
            elif package == "tkinter":
                import tkinter
            elif package == "faiss":
                import faiss
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ❌ {package} (not installed)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n💡 To install missing packages:")
        if "faiss" in missing_packages:
            print("  pip install faiss-cpu")
        print("  pip install -r requirements.txt")
    
    return len(missing_packages) == 0

if __name__ == "__main__":
    print("🧪 Fashion Recommendation System - Debug Test")
    print("=" * 60)
    
    imports_ok = test_imports()
    files_ok = check_files() 
    deps_ok = check_dependencies()
    
    print("\n" + "=" * 60)
    if imports_ok and files_ok and deps_ok:
        print("🎉 Everything looks good! You can run the application.")
        print("\nTry: python app.py")
    else:
        print("⚠️ Some issues found. Please fix them and try again.")
        
    input("\nPress Enter to exit...")