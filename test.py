import json
import numpy as np
import faiss
import os
from PIL import Image
import matplotlib.pyplot as plt
from config import config, resolve_image_path

# --- Config ---
NPZ_PATH = config.embeddings_path   # your saved gallery
QUERY_JSON = os.path.join(config.data_dir_path, "user_searches", "search_2025-09-25T00-42-29.721242.json")  # file containing the query embedding JSON
TOP_K = 5

# --- Load gallery embeddings ---
npz = np.load(NPZ_PATH, allow_pickle=True)
gallery_vecs = npz["vecs"].astype("float32")          # shape: [N, 256]
gallery_paths = list(npz["paths"])                    # image file paths used during embedding

print(f"Loaded {len(gallery_vecs)} image embeddings from {NPZ_PATH}")

# --- Build FAISS index ---
d = gallery_vecs.shape[1]
index = faiss.IndexFlatIP(d)                          # inner product (cosine if normalized)
index.add(gallery_vecs)

# --- Load query embedding from JSON ---
with open(QUERY_JSON, "r") as f:
    data = json.load(f)

query_vec = np.array(data["embedding"], dtype="float32").reshape(1, -1)

print(f"Query: {data['query']} (dim={query_vec.shape[1]})")

# --- Search in FAISS ---
sims, idxs = index.search(query_vec, TOP_K)

# --- Show results ---
print("\nTop results:")
for rank, (i, s) in enumerate(zip(idxs[0], sims[0]), 1):
    print(f"{rank}. {gallery_paths[i]}   score={s:.3f}")

# Optional: visualize
plt.figure(figsize=(15, 3))
for j, i in enumerate(idxs[0], 1):
    try:
        img_path = resolve_image_path(gallery_paths[i])
        img = Image.open(img_path).convert("RGB")
        plt.subplot(1, TOP_K, j)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"{j}. {sims[0][j-1]:.2f}")
    except Exception as e:
        print(f"Error loading {gallery_paths[i]}: {e}")
plt.show()
