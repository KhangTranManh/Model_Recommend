#!/usr/bin/env python3
"""
Model Efficiency Test Suite
Tests FashionCLIP model performance, speed, accuracy, and memory usage.
"""

import sys
import os
import time
import numpy as np
import torch
import psutil
import gc
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class ModelEfficiencyTester:
    """Comprehensive model efficiency testing"""
    
    def __init__(self):
        self.results = {}
        # Force GPU if available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"🔧 Testing on: {self.device} (GPU)")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print(f"   CUDA Version: {torch.version.cuda}")
        else:
            self.device = torch.device("cpu")
            print(f"🔧 Testing on: {self.device} (CPU)")
            print("   ⚠️ GPU not available - using CPU")
            print("   💡 To use GPU: Install CUDA and PyTorch with CUDA support")
        
    def test_model_loading(self):
        """Test model loading time and memory usage"""
        print("\n" + "="*60)
        print("📦 TEST 1: Model Loading")
        print("="*60)
        
        from fashion_recommender.models.similarity import load_model
        from fashion_recommender.config.config import config
        
        # Memory before loading
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Time model loading
        start_time = time.time()
        model, processor, cfg = load_model(config.checkpoint_path, self.device)
        load_time = time.time() - start_time
        
        # Memory after loading
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        self.results['model_loading'] = {
            'load_time': load_time,
            'memory_used_mb': memory_used,
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'device': str(self.device)
        }
        
        print(f"✅ Model loaded in {load_time:.3f} seconds")
        print(f"✅ Memory used: {memory_used:.1f} MB")
        print(f"✅ Model parameters: {self.results['model_loading']['model_parameters']:,}")
        
        return model, processor, cfg
    
    def test_embedding_speed(self, model, processor, cfg):
        """Test text and image embedding generation speed"""
        print("\n" + "="*60)
        print("⚡ TEST 2: Embedding Generation Speed")
        print("="*60)
        
        from fashion_recommender.models.similarity import embed_text, embed_one_image
        
        # Test text embeddings
        text_queries = [
            "red summer dress",
            "blue jeans",
            "black leather jacket",
            "white sneakers",
            "elegant evening gown"
        ]
        
        print("🔤 Testing text embeddings...")
        text_times = []
        
        for query in text_queries:
            start_time = time.time()
            embedding = embed_text(model, processor, query, self.device, cfg)
            end_time = time.time()
            
            text_times.append(end_time - start_time)
            print(f"  '{query}': {text_times[-1]:.4f}s")
        
        # Test image embeddings
        print("\n🖼️ Testing image embeddings...")
        image_times = []
        
        # Create test images and save them temporarily
        test_image_paths = []
        temp_dir = Path("temp_test_images")
        temp_dir.mkdir(exist_ok=True)
        
        for i in range(5):
            # Create random test image
            img = Image.new("RGB", (224, 224), (i*50, 100, 200-i*30))
            img_path = temp_dir / f"test_image_{i}.jpg"
            img.save(img_path)
            test_image_paths.append(str(img_path))
        
        for i, img_path in enumerate(test_image_paths):
            start_time = time.time()
            embedding = embed_one_image(model, processor, img_path, self.device, cfg)
            end_time = time.time()
            
            image_times.append(end_time - start_time)
            print(f"  Image {i+1}: {image_times[-1]:.4f}s")
        
        # Clean up temp images
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        self.results['embedding_speed'] = {
            'text_avg_time': np.mean(text_times),
            'text_min_time': np.min(text_times),
            'text_max_time': np.max(text_times),
            'image_avg_time': np.mean(image_times),
            'image_min_time': np.min(image_times),
            'image_max_time': np.max(image_times),
            'text_queries': len(text_queries),
            'image_tests': len(test_images)
        }
        
        print(f"\n📊 Text Embedding Summary:")
        print(f"  Average: {np.mean(text_times):.4f}s")
        print(f"  Min: {np.min(text_times):.4f}s")
        print(f"  Max: {np.max(text_times):.4f}s")
        
        print(f"\n📊 Image Embedding Summary:")
        print(f"  Average: {np.mean(image_times):.4f}s")
        print(f"  Min: {np.min(image_times):.4f}s")
        print(f"  Max: {np.max(image_times):.4f}s")
    
    def test_batch_processing(self, model, processor, cfg):
        """Test batch processing efficiency"""
        print("\n" + "="*60)
        print("📦 TEST 3: Batch Processing")
        print("="*60)
        
        from fashion_recommender.models.similarity import embed_text
        
        batch_sizes = [1, 4, 8, 16, 32]
        text_queries = [
            "red dress", "blue jeans", "black shoes", "white shirt",
            "green jacket", "purple skirt", "brown boots", "pink blouse"
        ] * 4  # 32 queries total
        
        batch_results = {}
        
        for batch_size in batch_sizes:
            print(f"\n🔄 Testing batch size: {batch_size}")
            
            # Prepare batches
            batches = [text_queries[i:i+batch_size] for i in range(0, len(text_queries), batch_size)]
            
            total_time = 0
            total_embeddings = 0
            
            for batch in batches:
                start_time = time.time()
                
                # Process batch
                for query in batch:
                    embedding = embed_text(model, processor, query, self.device, cfg)
                    total_embeddings += 1
                
                batch_time = time.time() - start_time
                total_time += batch_time
            
            avg_time_per_query = total_time / total_embeddings
            queries_per_second = total_embeddings / total_time
            
            batch_results[batch_size] = {
                'total_time': total_time,
                'avg_time_per_query': avg_time_per_query,
                'queries_per_second': queries_per_second,
                'total_queries': total_embeddings
            }
            
            print(f"  Total time: {total_time:.3f}s")
            print(f"  Avg per query: {avg_time_per_query:.4f}s")
            print(f"  Queries/second: {queries_per_second:.2f}")
        
        self.results['batch_processing'] = batch_results
    
    def test_search_efficiency(self, model, processor, cfg):
        """Test FAISS search efficiency"""
        print("\n" + "="*60)
        print("🔍 TEST 4: Search Efficiency")
        print("="*60)
        
        from fashion_recommender.config.config import config
        from fashion_recommender.models.similarity import embed_text
        import faiss
        
        # Load gallery data
        print("📁 Loading gallery data...")
        npz = np.load(config.embeddings_path, allow_pickle=True)
        gallery_embeddings = npz["vecs"].astype("float32")
        gallery_paths = list(npz["paths"])
        
        print(f"✅ Loaded {len(gallery_embeddings)} items")
        
        # Load FAISS index
        print("🔍 Loading FAISS index...")
        index = faiss.read_index(config.index_path)
        
        # Test search speeds
        search_queries = [
            "red dress", "blue jeans", "black shoes", "elegant gown",
            "casual shirt", "formal jacket", "summer dress", "winter coat"
        ]
        
        k_values = [5, 10, 20, 50, 100]
        search_results = {}
        
        for k in k_values:
            print(f"\n🔍 Testing k={k} (top-{k} results)")
            
            times = []
            for query in search_queries:
                # Embed query
                start_time = time.time()
                query_embedding = embed_text(model, processor, query, self.device, cfg)
                
                # Search
                sims, idxs = index.search(query_embedding, k)
                search_time = time.time() - start_time
                
                times.append(search_time)
            
            avg_time = np.mean(times)
            searches_per_second = len(search_queries) / sum(times)
            
            search_results[k] = {
                'avg_search_time': avg_time,
                'searches_per_second': searches_per_second,
                'total_queries': len(search_queries)
            }
            
            print(f"  Average search time: {avg_time:.4f}s")
            print(f"  Searches per second: {searches_per_second:.2f}")
        
        self.results['search_efficiency'] = search_results
    
    def test_memory_usage(self, model, processor, cfg):
        """Test memory usage during operations"""
        print("\n" + "="*60)
        print("💾 TEST 5: Memory Usage")
        print("="*60)
        
        process = psutil.Process()
        
        # Baseline memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Test with different operations
        from fashion_recommender.models.similarity import embed_text
        
        # Single embedding
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        single_memory = process.memory_info().rss / 1024 / 1024
        embedding = embed_text(model, processor, "test query", self.device, cfg)
        after_single = process.memory_info().rss / 1024 / 1024
        
        # Batch embeddings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        batch_memory = process.memory_info().rss / 1024 / 1024
        for i in range(10):
            embedding = embed_text(model, processor, f"test query {i}", self.device, cfg)
        after_batch = process.memory_info().rss / 1024 / 1024
        
        # GPU memory if available
        gpu_memory = {}
        if torch.cuda.is_available():
            gpu_memory = {
                'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                'cached_mb': torch.cuda.memory_reserved() / 1024 / 1024,
                'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024
            }
        
        self.results['memory_usage'] = {
            'baseline_memory_mb': baseline_memory,
            'single_embedding_mb': after_single - single_memory,
            'batch_embedding_mb': after_batch - batch_memory,
            'gpu_memory': gpu_memory
        }
        
        print(f"📊 Memory Usage:")
        print(f"  Baseline: {baseline_memory:.1f} MB")
        print(f"  Single embedding: {after_single - single_memory:.1f} MB")
        print(f"  Batch (10 queries): {after_batch - batch_memory:.1f} MB")
        
        if gpu_memory:
            print(f"  GPU allocated: {gpu_memory['allocated_mb']:.1f} MB")
            print(f"  GPU cached: {gpu_memory['cached_mb']:.1f} MB")
    
    def test_accuracy_consistency(self, model, processor, cfg):
        """Test embedding consistency and accuracy"""
        print("\n" + "="*60)
        print("🎯 TEST 6: Accuracy & Consistency")
        print("="*60)
        
        from fashion_recommender.models.similarity import embed_text
        
        # Test consistency (same input should give same output)
        query = "red summer dress"
        embeddings = []
        
        print(f"🔄 Testing consistency for: '{query}'")
        for i in range(5):
            embedding = embed_text(model, processor, query, self.device, cfg)
            embeddings.append(embedding[0])  # Remove batch dimension
        
        # Calculate similarity between embeddings
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j])
                similarities.append(sim)
        
        avg_similarity = np.mean(similarities)
        min_similarity = np.min(similarities)
        
        print(f"  Average similarity: {avg_similarity:.6f}")
        print(f"  Min similarity: {min_similarity:.6f}")
        print(f"  Consistency: {'✅ Good' if avg_similarity > 0.999 else '⚠️ Poor'}")
        
        # Test semantic similarity
        print(f"\n🧠 Testing semantic similarity:")
        
        similar_queries = [
            ("red dress", "crimson gown"),
            ("blue jeans", "denim pants"),
            ("black shoes", "dark footwear")
        ]
        
        semantic_similarities = []
        for query1, query2 in similar_queries:
            emb1 = embed_text(model, processor, query1, self.device, cfg)[0]
            emb2 = embed_text(model, processor, query2, self.device, cfg)[0]
            sim = np.dot(emb1, emb2)
            semantic_similarities.append(sim)
            print(f"  '{query1}' vs '{query2}': {sim:.4f}")
        
        self.results['accuracy_consistency'] = {
            'consistency_avg_similarity': avg_similarity,
            'consistency_min_similarity': min_similarity,
            'semantic_similarities': semantic_similarities,
            'is_consistent': avg_similarity > 0.999
        }
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*60)
        print("📊 EFFICIENCY TEST REPORT")
        print("="*60)
        
        # Model loading
        ml = self.results['model_loading']
        print(f"\n📦 Model Loading:")
        print(f"  Time: {ml['load_time']:.3f}s")
        print(f"  Memory: {ml['memory_used_mb']:.1f} MB")
        print(f"  Parameters: {ml['model_parameters']:,}")
        print(f"  Device: {ml['device']}")
        
        # Embedding speed
        es = self.results['embedding_speed']
        print(f"\n⚡ Embedding Speed:")
        print(f"  Text avg: {es['text_avg_time']:.4f}s")
        print(f"  Image avg: {es['image_avg_time']:.4f}s")
        print(f"  Text queries/sec: {1/es['text_avg_time']:.2f}")
        print(f"  Image queries/sec: {1/es['image_avg_time']:.2f}")
        
        # Batch processing
        bp = self.results['batch_processing']
        best_batch = max(bp.keys(), key=lambda k: bp[k]['queries_per_second'])
        print(f"\n📦 Batch Processing:")
        print(f"  Best batch size: {best_batch}")
        print(f"  Best throughput: {bp[best_batch]['queries_per_second']:.2f} queries/sec")
        
        # Search efficiency
        se = self.results['search_efficiency']
        print(f"\n🔍 Search Efficiency:")
        for k, data in se.items():
            print(f"  Top-{k}: {data['searches_per_second']:.2f} searches/sec")
        
        # Memory usage
        mu = self.results['memory_usage']
        print(f"\n💾 Memory Usage:")
        print(f"  Baseline: {mu['baseline_memory_mb']:.1f} MB")
        print(f"  Single embedding: {mu['single_embedding_mb']:.1f} MB")
        if mu['gpu_memory']:
            print(f"  GPU allocated: {mu['gpu_memory']['allocated_mb']:.1f} MB")
        
        # Accuracy
        ac = self.results['accuracy_consistency']
        print(f"\n🎯 Accuracy & Consistency:")
        print(f"  Consistency: {ac['consistency_avg_similarity']:.6f}")
        print(f"  Status: {'✅ Good' if ac['is_consistent'] else '⚠️ Poor'}")
        
        # Save detailed report
        report_file = f"efficiency_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed report saved to: {report_file}")
        
        # Performance summary
        print(f"\n🏆 PERFORMANCE SUMMARY:")
        print(f"  Model loading: {'✅ Fast' if ml['load_time'] < 5 else '⚠️ Slow'}")
        print(f"  Text embedding: {'✅ Fast' if es['text_avg_time'] < 0.1 else '⚠️ Slow'}")
        print(f"  Image embedding: {'✅ Fast' if es['image_avg_time'] < 0.1 else '⚠️ Slow'}")
        print(f"  Consistency: {'✅ Good' if ac['is_consistent'] else '⚠️ Poor'}")
        print(f"  Memory usage: {'✅ Low' if mu['single_embedding_mb'] < 50 else '⚠️ High'}")


def main():
    """Run all efficiency tests"""
    print("🚀 FashionCLIP Model Efficiency Test Suite")
    print("=" * 60)
    
    tester = ModelEfficiencyTester()
    
    try:
        # Test 1: Model loading
        model, processor, cfg = tester.test_model_loading()
        
        # Test 2: Embedding speed
        tester.test_embedding_speed(model, processor, cfg)
        
        # Test 3: Batch processing
        tester.test_batch_processing(model, processor, cfg)
        
        # Test 4: Search efficiency
        tester.test_search_efficiency(model, processor, cfg)
        
        # Test 5: Memory usage
        tester.test_memory_usage(model, processor, cfg)
        
        # Test 6: Accuracy consistency
        tester.test_accuracy_consistency(model, processor, cfg)
        
        # Generate report
        tester.generate_report()
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    
    if not success:
        print("\n❌ Common issues:")
        print("  • Missing model/data files")
        print("  • CUDA out of memory (try CPU)")
        print("  • Missing dependencies")
    
    print("\nPress Enter to exit...")
    input()
