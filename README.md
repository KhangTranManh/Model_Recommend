# Fashion Recommendation System

A fashion image similarity search system using CLIP embeddings with FAISS indexing. The system enables both text-to-image and image-to-image similarity search through a GUI interface.

## 🚀 Quick Start

### 1. Setup on Any Computer

1. **Clone/Copy the project** to your desired location
2. **Install Python 3.7+** if not already installed  
3. **Install dependencies**:
```bash
pip install torch transformers pillow numpy matplotlib
pip install faiss-cpu  # or faiss-gpu if you have CUDA
```

4. **Run setup verification**:
```bash
python setup_check.py
```

5. **Start the application**:
```bash
python run_demo.py
```

### 2. File Structure
```
Model_Recommend/
├── config.py              # 🆕 Portable configuration system
├── setup_check.py          # 🆕 Setup verification script
├── demo_recommend.py       # Main GUI application
├── run_demo.py            # Application launcher
├── last.py                # CLI interface & core functions
├── FashionCLIP.py         # Model architecture
├── emb.py                 # Batch embedding generation
├── user_history.py        # User tracking & analytics
├── fashion_clip_best.pt   # Trained model checkpoint
├── gallery_embeddings.npz # Pre-computed image embeddings
├── gallery_ip.index       # FAISS similarity search index
├── pic/                   # Fashion image gallery
└── data/                  # User data & analytics
```

## 🔧 What Was Fixed

### Path Portability Issues ✅
- **Before**: Hardcoded paths like `r"D:\Secret\duan\..."`
- **After**: Relative paths using `pathlib.Path` and project root detection

### Key Improvements:
1. **`config.py`** - Central configuration with portable paths
2. **Automatic path resolution** - Works regardless of where you place the project
3. **Directory auto-creation** - Creates required folders automatically
4. **Setup verification** - `setup_check.py` validates everything works

### Usage Examples

**Text to Image Search:**
```bash
python last.py --query_text "red floral summer dress" --k 6
```

**Image to Image Search:**
```bash  
python last.py --query_image "path/to/your/image.jpg" --k 6
```

**GUI Application:**
```bash
python run_demo.py
```

## 🛠️ Technical Details

- **Model**: CLIP backbone with 256-D projection heads
- **Search**: FAISS inner-product similarity index
- **UI**: Tkinter-based GUI with analytics tracking
- **Storage**: NPZ format for embeddings, JSON for user history

## 📋 Requirements

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- FAISS (CPU or GPU)
- PIL/Pillow
- NumPy
- Matplotlib
- Tkinter (usually included with Python)

## 🔍 Troubleshooting

Run `python setup_check.py` to diagnose issues automatically.