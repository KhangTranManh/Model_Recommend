#!/usr/bin/env python3
"""
Setup script for Fashion Recommendation System
Run this script on a new computer to verify everything is configured correctly
"""

import os
import sys
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required. Current version:", sys.version)
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'torch', 'transformers', 'PIL', 'numpy', 
        'tkinter', 'matplotlib'
    ]
    
    optional_packages = [
        'faiss-cpu',  # or faiss-gpu
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'tkinter':
                import tkinter
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    # Check optional packages
    try:
        import faiss
        print("✅ faiss (for fast similarity search)")
    except ImportError:
        print("⚠️  faiss not found - similarity search will be slower")
        print("   Install with: pip install faiss-cpu")
    
    if missing:
        print(f"\n❌ Missing required packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    return True

def check_project_structure():
    """Verify project files and directories exist"""
    from fashion_recommender.config.config import config, ensure_directories
    
    print("\n📁 Checking project structure...")
    
    # Check critical files
    files_to_check = [
        ("Model checkpoint", config.checkpoint_path),
        ("Gallery embeddings", config.embeddings_path),
        ("Gallery index", config.index_path),
    ]
    
    missing_files = []
    for name, path in files_to_check:
        if os.path.exists(path):
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: {path}")
            missing_files.append((name, path))
    
    # Check directories
    dirs_to_check = [
        ("Images directory", config.images_dir_path),
    ]
    
    for name, path in dirs_to_check:
        if os.path.exists(path) and os.path.isdir(path):
            count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            print(f"✅ {name}: {path} ({count} images)")
        else:
            print(f"❌ {name}: {path}")
            missing_files.append((name, path))
    
    # Create data directories
    try:
        ensure_directories()
        print("✅ Data directories created/verified")
    except Exception as e:
        print(f"❌ Error creating directories: {e}")
    
    if missing_files:
        print("\n❌ Missing files/directories:")
        for name, path in missing_files:
            print(f"   {name}: {path}")
        return False
    
    return True

def run_basic_test():
    """Run a basic functionality test"""
    print("\n🧪 Running basic functionality test...")
    
    try:
        # Test model loading
        from last import load_model
        from fashion_recommender.config.config import config
        import torch
        
        device = torch.device("cpu")  # Use CPU for setup test
        model, processor, cfg = load_model(config.checkpoint_path, device)
        print("✅ Model loads successfully")
        
        # Test embedding generation
        from last import embed_text
        embedding = embed_text(model, processor, "test query", device, cfg)
        print(f"✅ Text embedding generation works (shape: {embedding.shape})")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False

def main():
    """Main setup verification"""
    print("🚀 Fashion Recommendation System - Setup Verification")
    print("=" * 60)
    
    all_good = True
    
    # Check Python version
    if not check_python_version():
        all_good = False
    
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        all_good = False
    
    # Check project structure
    if not check_project_structure():
        all_good = False
    
    # Run basic test
    if all_good and not run_basic_test():
        all_good = False
    
    print("\n" + "=" * 60)
    if all_good:
        print("🎉 Setup verification completed successfully!")
        print("You can now run the application with: python run_demo.py")
    else:
        print("❌ Setup verification failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()