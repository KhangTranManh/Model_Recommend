import os, json
import numpy as np
from collections import defaultdict, Counter

HISTORY_FILE = "user_history.json"
GLOBAL_FILE = "global_stats.json"

# -------------------
# Helpers
# -------------------
def _load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def _save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# -------------------
# User History
# -------------------
def save_query(user_id, query_type, query_text=None, product_id=None, embedding=None):
    """
    Save a search or purchase into the user history.
    query_type: "search" or "purchase"
    """
    data = _load_json(HISTORY_FILE, {})

    if user_id not in data:
        data[user_id] = {"history": []}

    entry = {
        "type": query_type,
        "query": query_text,
        "product": product_id,
        "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
    }
    data[user_id]["history"].append(entry)

    _save_json(HISTORY_FILE, data)

    # Update global stats
    if query_type == "search" and query_text:
        add_global_query(query_text)
    if query_type == "purchase" and product_id:
        add_global_item(product_id)

def get_history(user_id, limit=10):
    data = _load_json(HISTORY_FILE, {})
    if user_id not in data:
        return []
    return data[user_id]["history"][-limit:]

def get_recent_embedding(user_id, n=5):
    """
    Average the last n embeddings (searches/purchases).
    Returns None if no embeddings exist.
    """
    history = get_history(user_id, limit=n)
    embs = [np.array(h["embedding"], dtype="float32")
            for h in history if h.get("embedding") is not None]
    if not embs:
        return None
    return np.mean(embs, axis=0)

# -------------------
# Global Analytics
# -------------------
def add_global_item(product_id):
    stats = _load_json(GLOBAL_FILE, {"popular_items": {}, "popular_queries": {}})
    stats["popular_items"][product_id] = stats["popular_items"].get(product_id, 0) + 1
    _save_json(GLOBAL_FILE, stats)

def add_global_query(query_text):
    stats = _load_json(GLOBAL_FILE, {"popular_items": {}, "popular_queries": {}})
    stats["popular_queries"][query_text] = stats["popular_queries"].get(query_text, 0) + 1
    _save_json(GLOBAL_FILE, stats)

def get_top_items(k=5):
    stats = _load_json(GLOBAL_FILE, {"popular_items": {}, "popular_queries": {}})
    counter = Counter(stats["popular_items"])
    return counter.most_common(k)

def get_top_queries(k=5):
    stats = _load_json(GLOBAL_FILE, {"popular_items": {}, "popular_queries": {}})
    counter = Counter(stats["popular_queries"])
    return counter.most_common(k)

# -------------------
# Example Usage
# -------------------
if __name__ == "__main__":
    # Save some fake data
    save_query("user1", "search", query_text="red dress", embedding=np.random.rand(256))
    save_query("user1", "purchase", product_id="10001.jpg", embedding=np.random.rand(256))

    print("User1 history:", get_history("user1"))
    print("User1 vector (avg):", get_recent_embedding("user1", n=5)[:5], "...")

    print("Top Items:", get_top_items())
    print("Top Queries:", get_top_queries())
