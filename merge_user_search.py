import os, json, glob

SEARCH_DIR = r"D:\Secret\duan\Model_Recommend\data\user_searches"
OUTPUT = r"D:\Secret\duan\Model_Recommend\data\history.json"

def merge_searches():
    all_entries = []

    for path in glob.glob(os.path.join(SEARCH_DIR, "*.json")):
        try:
            with open(path, "r") as f:
                data = json.load(f)

                if isinstance(data, dict):
                    entry = {
                        "user_id": data.get("user_id"),
                        "timestamp": data.get("timestamp"),
                        "query": data.get("query"),
                        # drop "embedding" if exists
                        "clicked_item": data.get("product_id") or data.get("clicked_item")
                    }
                    all_entries.append(entry)

                elif isinstance(data, list):
                    for d in data:
                        entry = {
                            "user_id": d.get("user_id"),
                            "timestamp": d.get("timestamp"),
                            "query": d.get("query"),
                            "clicked_item": d.get("product_id") or d.get("clicked_item")
                        }
                        all_entries.append(entry)

        except Exception as e:
            print(f"⚠️ Error reading {path}: {e}")

    # Sort by timestamp
    all_entries.sort(key=lambda x: x.get("timestamp", ""))

    with open(OUTPUT, "w") as f:
        json.dump(all_entries, f, indent=2)

    print(f"✅ Merged {len(all_entries)} entries into {OUTPUT}")

if __name__ == "__main__":
    merge_searches()
