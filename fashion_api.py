# fashion_api.py
import os, numpy as np, torch
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import CLIPProcessor
import faiss

# ===== Load model + index =====
from fashion_recommender.models.FashionCLIP import FashionCLIP  # model class definition
from fashion_recommender.config.config import DEFAULT_CHECKPOINT, DEFAULT_INDEX, DEFAULT_NPZ, DEFAULT_IMAGES_DIR

# Centralized, portable paths
CHECKPOINT = DEFAULT_CHECKPOINT
INDEX_PATH = DEFAULT_INDEX
NPZ_PATH = DEFAULT_NPZ
IMAGES_DIR = DEFAULT_IMAGES_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load checkpoint ---
ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
cfg = ckpt["config"]
processor = CLIPProcessor.from_pretrained(cfg["model_name"])
model = FashionCLIP(
    cfg["model_name"],
    cfg.get("embedding_dim", 256),
    use_projection=True,
    use_layer_norm=True,
    enable_compile=False,
).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# --- Load FAISS + paths ---
npz = np.load(NPZ_PATH, allow_pickle=True)
paths = list(npz["paths"])
basenames = [os.path.basename(p) for p in paths]
local_paths = [os.path.join(IMAGES_DIR, b) for b in basenames]
index = faiss.read_index(INDEX_PATH)

app = FastAPI(title="Fashion Recommendation API")

# ===== Embedding helpers =====
@torch.no_grad()
def embed_text(query: str):
    blank = Image.new("RGB", (cfg["image_size"], cfg["image_size"]), "white")
    enc = processor(text=[query], images=[blank], return_tensors="pt",
                    padding="max_length", truncation=True, max_length=cfg["max_length"])
    pv, ids, am = enc["pixel_values"].to(device), enc["input_ids"].to(device), enc["attention_mask"].to(device)
    _, txt = model(pv, ids, am)
    return txt.cpu().numpy().astype("float32")

@torch.no_grad()
def embed_image(img: Image.Image):
    enc = processor(text=[""], images=[img.convert("RGB")], return_tensors="pt",
                    padding="max_length", truncation=True, max_length=cfg["max_length"])
    pv, ids, am = enc["pixel_values"].to(device), enc["input_ids"].to(device), enc["attention_mask"].to(device)
    img_e, _ = model(pv, ids, am)
    return img_e.cpu().numpy().astype("float32")

# ===== Routes =====
@app.get("/")
def root():
    return {"message": "Fashion Recommendation API is running."}

@app.post("/search/text")
def search_text(query: str = Form(...), k: int = Form(5)):
    k = max(1, min(int(k), 50))
    q = embed_text(query)
    sims, idxs = index.search(q, k)
    results = [{"path": local_paths[i], "score": float(s)} for i, s in zip(idxs[0], sims[0])]
    return JSONResponse(results)

@app.post("/search/image")
async def search_image(file: UploadFile, k: int = Form(5)):
    k = max(1, min(int(k), 50))
    img = Image.open(file.file)
    q = embed_image(img)
    sims, idxs = index.search(q, k)
    results = [{"path": local_paths[i], "score": float(s)} for i, s in zip(idxs[0], sims[0])]
    return JSONResponse(results)
