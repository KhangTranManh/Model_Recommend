import os
import sys
import json
import numpy as np
from datetime import datetime
from collections import Counter
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch
import faiss
import socket
import requests

# Import your existing modules
from last import load_model, embed_one_image, embed_text, search
from user_history import save_query, get_recent_embedding, get_top_items, get_top_queries

class FashionRecommendationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fashion Recommendation System")
        self.root.geometry("1400x900")
        
        # Configuration
        self.CHECKPOINT = r"D:\Secret\duan\fashion_clip_best.pt"
        self.IMAGES_DIR = r"D:\Secret\duan\pic"
        self.NPZ_PATH = r"D:\Secret\duan\gallery_embeddings.npz"
        self.DATA_DIR = "data"
        self.USER_ID = "user1"
        
        # Get user IP for tracking
        self.user_ip = self.get_user_ip()
        
        # Create data directory structure
        self.setup_data_directories()
        
        # Create analytics directories
        os.makedirs(os.path.join(self.DATA_DIR, "analytics"), exist_ok=True)
        os.makedirs(os.path.join(self.DATA_DIR, "logs"), exist_ok=True)
        
        # Initialize model and data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.cfg = None
        self.index = None
        self.keep_paths = []
        
        self.setup_ui()
        self.load_system()
        
        # Create privacy information file
        self.create_privacy_info_file()
        
        # Log app startup
        self.log_app_usage("app_startup", {
            "model_checkpoint": self.CHECKPOINT,
            "images_directory": self.IMAGES_DIR
        })
        
    def setup_data_directories(self):
        """Create data directory structure"""
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(os.path.join(self.DATA_DIR, "user_searches"), exist_ok=True)
        os.makedirs(os.path.join(self.DATA_DIR, "user_purchases"), exist_ok=True)
        os.makedirs(os.path.join(self.DATA_DIR, "global"), exist_ok=True)
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top panel - Search
        search_frame = ttk.LabelFrame(main_frame, text="Search Fashion Items", padding=10)
        search_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Text search
        ttk.Label(search_frame, text="Text Search:").pack(anchor=tk.W)
        text_frame = ttk.Frame(search_frame)
        text_frame.pack(fill=tk.X, pady=(5, 10))
        
        self.search_entry = ttk.Entry(text_frame, font=('Arial', 12))
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.search_entry.bind('<Return>', lambda e: self.perform_search())
        
        ttk.Button(text_frame, text="Search", command=self.perform_search).pack(side=tk.RIGHT)
        
        # Image search
        ttk.Label(search_frame, text="Image Search:").pack(anchor=tk.W)
        image_frame = ttk.Frame(search_frame)
        image_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(image_frame, text="Upload Image", command=self.upload_image_search).pack(side=tk.LEFT)
        self.image_path_label = ttk.Label(image_frame, text="No image selected")
        self.image_path_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Middle panel - Results and Analytics
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side - Search Results
        results_frame = ttk.LabelFrame(content_frame, text="Search Results", padding=10)
        results_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Results canvas with scrollbar
        canvas_frame = ttk.Frame(results_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_canvas = tk.Canvas(canvas_frame, bg='white')
        results_scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.results_canvas.yview)
        self.scrollable_frame = ttk.Frame(self.results_canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))
        )
        
        self.results_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.results_canvas.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_canvas.pack(side="left", fill="both", expand=True)
        results_scrollbar.pack(side="right", fill="y")
        
        # Right side - Analytics
        analytics_frame = ttk.LabelFrame(content_frame, text="Analytics", padding=10)
        analytics_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        # Top 5 Purchases
        ttk.Label(analytics_frame, text="🔥 Top 5 Purchases:", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        self.top_purchases_text = tk.Text(analytics_frame, height=8, width=30, wrap=tk.WORD)
        self.top_purchases_text.pack(pady=(0, 10))
        
        # Recent Searches
        ttk.Label(analytics_frame, text="🔍 Recent Searches:", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        self.recent_searches_text = tk.Text(analytics_frame, height=6, width=30, wrap=tk.WORD)
        self.recent_searches_text.pack(pady=(0, 10))
        
        # Personalized Recommendations
        ttk.Label(analytics_frame, text="💡 For You:", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        self.recommendations_text = tk.Text(analytics_frame, height=6, width=30, wrap=tk.WORD)
        self.recommendations_text.pack()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, pady=(10, 0))
        
    def load_system(self):
        """Load the model and image embeddings"""
        try:
            self.status_var.set("Loading model...")
            self.root.update()
            
            # Load model
            self.model, self.processor, self.cfg = load_model(self.CHECKPOINT, self.device)
            
            # Load embeddings
            if os.path.exists(self.NPZ_PATH):
                npz = np.load(self.NPZ_PATH, allow_pickle=True)
                gallery_vecs = torch.tensor(npz["vecs"], dtype=torch.float32)
                self.keep_paths = list(npz["paths"])
                
                # Store gallery vectors for duplicate detection
                self.gallery_vecs = gallery_vecs
                
                # Build FAISS index
                d = gallery_vecs.shape[1]
                self.index = faiss.IndexFlatIP(d)
                self.index.add(gallery_vecs.numpy().astype("float32"))
                
                self.status_var.set(f"Loaded {len(self.keep_paths)} items")
                self.update_analytics()
            else:
                messagebox.showerror("Error", f"Embeddings file not found: {self.NPZ_PATH}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load system: {str(e)}")
            
    def get_user_ip(self):
        """Get user's IP address for tracking purposes"""
        try:
            # Try to get public IP first
            response = requests.get('https://api.ipify.org', timeout=5)
            if response.status_code == 200:
                return response.text.strip()
        except Exception:
            pass
            
        try:
            # Fallback to local IP
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            return local_ip
        except Exception:
            pass
            
        try:
            # Another fallback method
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "unknown"
            
    def get_session_info(self):
        """Get additional session information for tracking"""
        import platform
        
        try:
            session_info = {
                "platform": platform.system(),
                "platform_version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "hostname": socket.gethostname(),
                "user_agent": f"FashionRecommendationApp/1.0 ({platform.system()})"
            }
            return session_info
        except Exception:
            return {"platform": "unknown"}
            
    def perform_search(self):
        """Perform text search"""
        query_text = self.search_entry.get().strip()
        if not query_text:
            messagebox.showwarning("Warning", "Please enter a search query")
            return
            
        try:
            self.status_var.set(f"Searching for '{query_text}'...")
            self.root.update()
            
            # Embed text query
            text_vec = embed_text(self.model, self.processor, query_text, self.device, self.cfg)
            sims, idxs = search(self.index, text_vec, 20)  # Get more results to filter
            
            # Remove duplicates
            unique_idxs, unique_sims = self.remove_duplicates(idxs, sims)
            
            # Save search query with IP and session info
            self.save_user_search(query_text, text_vec[0], search_type="text")
            
            # Log search action
            self.log_app_usage("text_search", {"query": query_text, "results_count": len(unique_idxs)})
            
            # Display results
            self.display_results(unique_idxs, unique_sims, f"Text Search: '{query_text}'")
            
            self.status_var.set(f"Found {len(unique_idxs)} unique results for '{query_text}'")
            self.update_analytics()
            
        except Exception as e:
            messagebox.showerror("Error", f"Search failed: {str(e)}")
            
    def upload_image_search(self):
        """Upload and search by image"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp *.bmp *.tiff")]
        )
        
        if not file_path:
            return
            
        try:
            self.image_path_label.config(text=os.path.basename(file_path))
            self.status_var.set("Processing image...")
            self.root.update()
            
            # Embed image
            image_vec = embed_one_image(self.model, self.processor, file_path, self.device, self.cfg)
            sims, idxs = search(self.index, image_vec, 20)  # Get more results to filter
            
            # Remove duplicates
            unique_idxs, unique_sims = self.remove_duplicates(idxs, sims)
            
            # Save image search (using filename as query)
            query_name = os.path.basename(file_path)
            self.save_user_search(f"[IMAGE] {query_name}", image_vec[0], search_type="image", image_path=file_path)
            
            # Log image search action
            self.log_app_usage("image_search", {"image_file": query_name, "results_count": len(unique_idxs)})
            
            # Display results
            self.display_results(unique_idxs, unique_sims, f"Image Search: {query_name}")
            
            self.status_var.set(f"Found {len(unique_idxs)} similar items")
            self.update_analytics()
            
        except Exception as e:
            messagebox.showerror("Error", f"Image search failed: {str(e)}")
            
    def display_results(self, idxs, sims, title):
        """Display search results with images"""
        # Clear previous results
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
            
        # Title
        title_label = ttk.Label(self.scrollable_frame, text=title, font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # Results
        for rank, (idx, sim) in enumerate(zip(idxs, sims), 1):
            if idx >= len(self.keep_paths):
                continue
                
            img_path = self.keep_paths[idx]
            
            # Result frame
            result_frame = ttk.Frame(self.scrollable_frame, relief=tk.RIDGE, borderwidth=1)
            result_frame.pack(fill=tk.X, pady=5, padx=10)
            
            # Image and info frame
            content_frame = ttk.Frame(result_frame)
            content_frame.pack(fill=tk.X, padx=10, pady=10)
            
            # Load and display image
            try:
                img = Image.open(img_path)
                img.thumbnail((150, 150), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                img_label = ttk.Label(content_frame, image=photo)
                img_label.image = photo  # Keep a reference
                img_label.pack(side=tk.LEFT, padx=(0, 10))
            except Exception:
                # If image can't be loaded, show placeholder
                placeholder = ttk.Label(content_frame, text="[Image]", width=20, relief=tk.SUNKEN)
                placeholder.pack(side=tk.LEFT, padx=(0, 10))
            
            # Info frame
            info_frame = ttk.Frame(content_frame)
            info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            # Item info
            ttk.Label(info_frame, text=f"#{rank}", font=('Arial', 12, 'bold')).pack(anchor=tk.W)
            ttk.Label(info_frame, text=f"File: {os.path.basename(img_path)}", wraplength=300).pack(anchor=tk.W)
            ttk.Label(info_frame, text=f"Similarity: {sim:.3f}", font=('Arial', 10)).pack(anchor=tk.W)
            
            # Buy button
            buy_btn = ttk.Button(info_frame, text="🛒 Buy", 
                               command=lambda p=img_path: self.buy_item(p))
            buy_btn.pack(anchor=tk.W, pady=(5, 0))
            
    def remove_duplicates(self, idxs, sims, similarity_threshold=0.95, max_results=10):
        """Remove duplicate/highly similar results"""
        if len(idxs) == 0:
            return idxs, sims
            
        unique_idxs = []
        unique_sims = []
        seen_files = set()
        
        for idx, sim in zip(idxs, sims):
            if idx >= len(self.keep_paths):
                continue
                
            img_path = self.keep_paths[idx]
            filename = os.path.basename(img_path)
            
            # Skip exact filename duplicates
            if filename in seen_files:
                continue
                
            # Check if this image is too similar to already selected ones
            is_duplicate = False
            if len(unique_idxs) > 0:
                try:
                    # Get embeddings for comparison
                    current_vec = None
                    if hasattr(self, 'gallery_vecs') and idx < len(self.gallery_vecs):
                        current_vec = self.gallery_vecs[idx]
                    else:
                        # Fallback: compute embedding on the fly
                        current_vec = embed_one_image(self.model, self.processor, img_path, self.device, self.cfg)
                        current_vec = torch.tensor(current_vec).flatten()
                    
                    for prev_idx in unique_idxs[-3:]:  # Check against last 3 items only for speed
                        if prev_idx >= len(self.keep_paths):
                            continue
                            
                        prev_path = self.keep_paths[prev_idx]
                        prev_vec = None
                        
                        if hasattr(self, 'gallery_vecs') and prev_idx < len(self.gallery_vecs):
                            prev_vec = self.gallery_vecs[prev_idx]
                        else:
                            prev_vec = embed_one_image(self.model, self.processor, prev_path, self.device, self.cfg)
                            prev_vec = torch.tensor(prev_vec).flatten()
                        
                        # Calculate cosine similarity
                        if current_vec is not None and prev_vec is not None:
                            cos_sim = torch.cosine_similarity(current_vec.unsqueeze(0), prev_vec.unsqueeze(0))
                            if cos_sim.item() > similarity_threshold:
                                is_duplicate = True
                                break
                                
                except Exception:
                    # If embedding comparison fails, use filename similarity as fallback
                    for prev_idx in unique_idxs[-3:]:
                        if prev_idx >= len(self.keep_paths):
                            continue
                        prev_filename = os.path.basename(self.keep_paths[prev_idx])
                        # Simple filename similarity check
                        if self.filename_similarity(filename, prev_filename) > 0.8:
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                unique_idxs.append(idx)
                unique_sims.append(sim)
                seen_files.add(filename)
                
                # Limit results
                if len(unique_idxs) >= max_results:
                    break
        
        return unique_idxs, unique_sims
        
    def filename_similarity(self, file1, file2):
        """Calculate similarity between two filenames (simple approach)"""
        # Remove extensions
        name1 = os.path.splitext(file1)[0]
        name2 = os.path.splitext(file2)[0]
        
        # If names are identical, return high similarity
        if name1 == name2:
            return 1.0
            
        # Check for similar numeric patterns (like 15977.jpg vs 15976.jpg)
        try:
            # Extract numbers from filenames
            import re
            nums1 = re.findall(r'\d+', name1)
            nums2 = re.findall(r'\d+', name2)
            
            if len(nums1) == 1 and len(nums2) == 1:
                num1, num2 = int(nums1[0]), int(nums2[0])
                # If numbers are very close, consider them similar
                if abs(num1 - num2) <= 2:
                    return 0.9
                    
        except:
            pass
            
        # Basic string similarity
        common_chars = set(name1.lower()) & set(name2.lower())
        total_chars = set(name1.lower()) | set(name2.lower())
        
        if len(total_chars) == 0:
            return 0.0
            
        return len(common_chars) / len(total_chars)
            
    def buy_item(self, img_path):
        """Handle item purchase"""
        try:
            # Get item embedding
            item_vec = embed_one_image(self.model, self.processor, img_path, self.device, self.cfg)
            
            # Save purchase
            item_name = os.path.basename(img_path)
            self.save_user_purchase(item_name, item_vec[0], img_path)
            
            # Log purchase action
            self.log_app_usage("purchase", {"item": item_name})
            
            messagebox.showinfo("Purchase", f"Purchased: {item_name}")
            self.update_analytics()
            
        except Exception as e:
            messagebox.showerror("Error", f"Purchase failed: {str(e)}")
            
    def save_user_search(self, query, embedding, search_type="text", image_path=None):
        """Save user search to data/user_searches with IP tracking"""
        timestamp = datetime.now().isoformat()
        session_info = self.get_session_info()
        
        search_data = {
            "user_id": self.USER_ID,
            "timestamp": timestamp,
            "query": query,
            "search_type": search_type,
            "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
            
            # IP and tracking information (not shown in GUI)
            "ip_address": self.user_ip,
            "session_info": session_info,
            "image_path": image_path if image_path else None,
            
            # Additional metadata for analytics
            "metadata": {
                "app_version": "1.0",
                "search_source": "desktop_app",
                "embedding_model": "fashion_clip_best.pt"
            }
        }
        
        # Save individual search file
        filename = f"search_{timestamp.replace(':', '-')}.json"
        filepath = os.path.join(self.DATA_DIR, "user_searches", filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(search_data, f, indent=2)
            
        # Update search analytics
        self.update_search_analytics(search_type)
            
    def save_user_purchase(self, item_name, embedding, img_path):
        """Save user purchase to data/user_purchases with IP tracking"""
        timestamp = datetime.now().isoformat()
        session_info = self.get_session_info()
        
        purchase_data = {
            "user_id": self.USER_ID,
            "timestamp": timestamp,
            "item": item_name,
            "item_path": img_path,
            "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
            
            # IP and tracking information (not shown in GUI)
            "ip_address": self.user_ip,
            "session_info": session_info,
            
            # Additional metadata for analytics
            "metadata": {
                "app_version": "1.0",
                "purchase_source": "desktop_app",
                "embedding_model": "fashion_clip_best.pt"
            }
        }
        
        # Save individual purchase file
        filename = f"purchase_{timestamp.replace(':', '-')}.json"
        filepath = os.path.join(self.DATA_DIR, "user_purchases", filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(purchase_data, f, indent=2)
            
        # Update global purchases
        self.update_global_purchases(item_name)
        
        # Update purchase analytics
        self.update_purchase_analytics()
        
    def update_global_purchases(self, item_name):
        """Update global purchase statistics"""
        global_file = os.path.join(self.DATA_DIR, "global", "top_purchases.json")
        
        # Load existing data
        if os.path.exists(global_file):
            with open(global_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {}
            
        # Update count
        data[item_name] = data.get(item_name, 0) + 1
        
        # Save updated data
        with open(global_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
    def update_search_analytics(self, search_type):
        """Update search analytics data"""
        analytics_file = os.path.join(self.DATA_DIR, "global", "search_analytics.json")
        
        # Load existing data
        if os.path.exists(analytics_file):
            with open(analytics_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {
                "total_searches": 0,
                "search_types": {},
                "ips": {},
                "daily_searches": {}
            }
            
        # Update counts
        data["total_searches"] = data.get("total_searches", 0) + 1
        data["search_types"][search_type] = data["search_types"].get(search_type, 0) + 1
        data["ips"][self.user_ip] = data["ips"].get(self.user_ip, 0) + 1
        
        # Daily search count
        today = datetime.now().strftime("%Y-%m-%d")
        data["daily_searches"][today] = data["daily_searches"].get(today, 0) + 1
        
        # Keep only last 30 days of daily data
        if len(data["daily_searches"]) > 30:
            dates = sorted(data["daily_searches"].keys())
            for old_date in dates[:-30]:
                del data["daily_searches"][old_date]
        
        # Save updated data
        with open(analytics_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
    def update_purchase_analytics(self):
        """Update purchase analytics data"""
        analytics_file = os.path.join(self.DATA_DIR, "global", "purchase_analytics.json")
        
        # Load existing data
        if os.path.exists(analytics_file):
            with open(analytics_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {
                "total_purchases": 0,
                "ips": {},
                "daily_purchases": {}
            }
            
        # Update counts
        data["total_purchases"] = data.get("total_purchases", 0) + 1
        data["ips"][self.user_ip] = data["ips"].get(self.user_ip, 0) + 1
        
        # Daily purchase count
        today = datetime.now().strftime("%Y-%m-%d")
        data["daily_purchases"][today] = data["daily_purchases"].get(today, 0) + 1
        
        # Keep only last 30 days of daily data
        if len(data["daily_purchases"]) > 30:
            dates = sorted(data["daily_purchases"].keys())
            for old_date in dates[:-30]:
                del data["daily_purchases"][old_date]
        
        # Save updated data
        with open(analytics_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
    def get_top_purchases(self, k=5):
        """Get top k purchased items"""
        global_file = os.path.join(self.DATA_DIR, "global", "top_purchases.json")
        
        if not os.path.exists(global_file):
            return []
            
        with open(global_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        counter = Counter(data)
        return counter.most_common(k)
        
    def get_recent_user_searches(self, k=5):
        """Get recent user searches"""
        searches_dir = os.path.join(self.DATA_DIR, "user_searches")
        
        if not os.path.exists(searches_dir):
            return []
            
        # Get all search files
        search_files = [f for f in os.listdir(searches_dir) if f.endswith('.json')]
        search_files.sort(reverse=True)  # Most recent first
        
        recent_searches = []
        for filename in search_files[:k]:
            filepath = os.path.join(searches_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    recent_searches.append(data['query'])
            except Exception:
                continue
                
        return recent_searches
        
    def get_user_recommendations(self, k=5):
        """Get personalized recommendations based on user history"""
        try:
            # Get user's recent activity embeddings
            purchases_dir = os.path.join(self.DATA_DIR, "user_purchases")
            searches_dir = os.path.join(self.DATA_DIR, "user_searches")
            
            embeddings = []
            
            # Collect recent purchase embeddings
            if os.path.exists(purchases_dir):
                purchase_files = [f for f in os.listdir(purchases_dir) if f.endswith('.json')]
                purchase_files.sort(reverse=True)
                
                for filename in purchase_files[:3]:  # Last 3 purchases
                    filepath = os.path.join(purchases_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            embeddings.append(np.array(data['embedding']))
                    except Exception:
                        continue
                        
            # Collect recent search embeddings
            if os.path.exists(searches_dir):
                search_files = [f for f in os.listdir(searches_dir) if f.endswith('.json')]
                search_files.sort(reverse=True)
                
                for filename in search_files[:2]:  # Last 2 searches
                    filepath = os.path.join(searches_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            embeddings.append(np.array(data['embedding']))
                    except Exception:
                        continue
                        
            if not embeddings:
                return []
                
            # Average embeddings to create user profile
            user_profile = np.mean(embeddings, axis=0)
            user_vec = np.expand_dims(user_profile, axis=0).astype("float32")
            
            # Search for similar items
            sims, idxs = search(self.index, user_vec, k)
            
            recommendations = []
            for idx, sim in zip(idxs, sims):
                if idx < len(self.keep_paths):
                    item_name = os.path.basename(self.keep_paths[idx])
                    recommendations.append(f"{item_name} ({sim:.3f})")
                    
            return recommendations
            
        except Exception:
            return []
            
    def log_app_usage(self, action, details=None):
        """Log application usage for analytics (includes IP tracking)"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "user_id": self.USER_ID,
            "ip_address": self.user_ip,
            "action": action,
            "details": details or {},
            "session_info": self.get_session_info()
        }
        
        # Create daily log file
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(self.DATA_DIR, "logs", f"usage_{today}.json")
        
        # Load existing logs or create new
        logs = []
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            except Exception:
                logs = []
        
        # Add new log entry
        logs.append(log_entry)
        
        # Save logs
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2)
            
    def create_privacy_info_file(self):
        """Create a privacy information file explaining data collection"""
        privacy_file = os.path.join(self.DATA_DIR, "PRIVACY_INFO.txt")
        
        privacy_content = """FASHION RECOMMENDATION SYSTEM - DATA COLLECTION NOTICE

This application collects the following information for analytics and personalization:

COLLECTED DATA:
- Search queries (text and image-based)
- Purchase history
- IP addresses
- System information (OS, platform, etc.)
- Timestamps of all actions
- Image embeddings and similarity data

PURPOSE:
- Improve recommendation accuracy
- Analyze usage patterns
- Provide personalized fashion suggestions
- System performance monitoring

DATA STORAGE:
- All data is stored locally in the 'data' directory
- Individual files for searches, purchases, and analytics
- No data is transmitted to external servers (except IP detection)

IP ADDRESS USAGE:
- Used for session tracking and analytics
- Helps identify unique users and usage patterns
- Not displayed in the application interface
- Stored in JSON files for analysis

DATA RETENTION:
- Search and purchase history: Indefinitely (local storage)
- Daily analytics: Last 30 days
- Usage logs: Organized by date

USER CONTROL:
- All data files are in human-readable JSON format
- Users can delete the 'data' directory to remove all tracking data
- No personal information beyond usage patterns is collected

This is a local application - your data remains on your computer.
Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """
"""
        
        with open(privacy_file, 'w', encoding='utf-8') as f:
            f.write(privacy_content)
            
    def update_analytics(self):
        """Update the analytics panel"""
        # Update top purchases
        self.top_purchases_text.delete(1.0, tk.END)
        top_purchases = self.get_top_purchases(5)
        if top_purchases:
            for rank, (item, count) in enumerate(top_purchases, 1):
                self.top_purchases_text.insert(tk.END, f"{rank}. {item}\n   ({count} purchases)\n\n")
        else:
            self.top_purchases_text.insert(tk.END, "No purchases yet")
            
        # Update recent searches
        self.recent_searches_text.delete(1.0, tk.END)
        recent_searches = self.get_recent_user_searches(5)
        if recent_searches:
            for search in recent_searches:
                self.recent_searches_text.insert(tk.END, f"• {search}\n")
        else:
            self.recent_searches_text.insert(tk.END, "No searches yet")
            
        # Update recommendations
        self.recommendations_text.delete(1.0, tk.END)
        recommendations = self.get_user_recommendations(5)
        if recommendations:
            for rec in recommendations:
                self.recommendations_text.insert(tk.END, f"• {rec}\n")
        else:
            self.recommendations_text.insert(tk.END, "No recommendations yet\n(Search or buy items first)")

def main():
    # Check if required files exist
    required_files = [
        "last.py",
        "user_history.py",
        "FashionCLIP.py"  # Assuming this exists
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Please ensure all required modules are in the same directory.")
        return
        
    root = tk.Tk()
    app = FashionRecommendationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()