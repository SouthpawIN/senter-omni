#!/usr/bin/env python3
"""
GPU Memory Optimizer for Senter-Omni

This script provides comprehensive GPU memory management and optimization
for running large language models with minimal memory fragmentation.
"""

import torch
import os
import gc
import psutil
from typing import Dict, Any
import warnings

class GPUMemoryOptimizer:
    """Comprehensive GPU memory management and optimization"""

    def __init__(self):
        self.original_alloc_config = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')

    def set_optimized_environment(self):
        """Set optimized PyTorch memory management environment variables"""
        print("🔧 Setting optimized GPU memory environment...")

        # Key optimizations for reducing fragmentation
        alloc_config = [
            "expandable_segments:True",  # Allow expandable memory segments
            "max_split_size_mb:512",     # Limit maximum split size
            "garbage_collection_threshold:0.8"  # Lower GC threshold
        ]

        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = ','.join(alloc_config)
        print("✅ PyTorch memory optimization set")

        # Additional CUDA optimizations
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        os.environ['CUDA_CACHE_DISABLE'] = '0'  # Keep cache enabled but managed

    def clear_gpu_cache(self, thorough: bool = True):
        """Thoroughly clear GPU cache and free memory"""
        print("🧹 Clearing GPU cache...")

        # Clear PyTorch cache multiple times
        for i in range(3 if thorough else 1):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()

        print("✅ GPU cache cleared")

    def get_memory_status(self) -> Dict[str, Any]:
        """Get comprehensive memory status"""
        status = {
            'gpu_count': torch.cuda.device_count(),
            'gpus': []
        }

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = props.total_memory / 1024**3

            gpu_info = {
                'id': i,
                'name': props.name,
                'total_gb': total,
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'free_gb': total - allocated,
                'fragmentation_percent': (reserved - allocated) / total * 100 if total > 0 else 0,
                'utilization_percent': allocated / total * 100 if total > 0 else 0
            }
            status['gpus'].append(gpu_info)

        # System memory
        system_memory = psutil.virtual_memory()
        status['system'] = {
            'total_gb': system_memory.total / 1024**3,
            'available_gb': system_memory.available / 1024**3,
            'used_percent': system_memory.percent
        }

        return status

    def print_memory_status(self, status: Dict[str, Any] = None):
        """Print formatted memory status"""
        if status is None:
            status = self.get_memory_status()

        print("📊 MEMORY STATUS")
        print("=" * 60)

        for gpu in status['gpus']:
            print(f"GPU {gpu['id']} ({gpu['name']}):")
            print(f"  Total: {gpu['total_gb']:.2f} GB")
            print(f"  Allocated: {gpu['allocated_gb']:.2f} GB")
            print(f"  Reserved: {gpu['reserved_gb']:.2f} GB")
            print(f"  Free: {gpu['free_gb']:.2f} GB")
            print(f"  Fragmentation: {gpu['fragmentation_percent']:.1f}%")
            print(f"  Utilization: {gpu['utilization_percent']:.1f}%")
            print()

        system = status['system']
        print("💻 SYSTEM MEMORY:")
        print(f"  Total: {system['total_gb']:.2f} GB")
        print(f"  Available: {system['available_gb']:.2f} GB")
        print(f"  Used: {system['used_percent']:.1f}%")
    def optimize_for_large_models(self):
        """Apply optimizations specifically for large model loading"""
        print("🚀 Applying large model optimizations...")

        # Disable gradients globally for inference
        torch.set_grad_enabled(False)

        # Use inference mode for better memory efficiency
        torch.inference_mode()

        # Set memory format optimization
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.95)  # Use up to 95% of GPU memory

        print("✅ Large model optimizations applied")

    def create_memory_efficient_context(self):
        """Create a context manager for memory-efficient operations"""
        class MemoryEfficientContext:
            def __init__(self, optimizer):
                self.optimizer = optimizer

            def __enter__(self):
                self.optimizer.clear_gpu_cache()
                torch.cuda.empty_cache()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                torch.cuda.empty_cache()
                gc.collect()

        return MemoryEfficientContext(self)

    def diagnose_memory_issues(self):
        """Diagnose common memory issues and provide solutions"""
        print("🔍 MEMORY ISSUE DIAGNOSIS")
        print("=" * 60)

        status = self.get_memory_status()

        issues_found = []

        # Check for high fragmentation
        for gpu in status['gpus']:
            if gpu['fragmentation_percent'] > 50:
                issues_found.append(f"High fragmentation on GPU {gpu['id']}: {gpu['fragmentation_percent']:.1f}%")
            if gpu['free_gb'] < 8:
                issues_found.append(f"Low free memory on GPU {gpu['id']}: {gpu['free_gb']:.2f} GB")

        # Check system memory
        if status['system']['available_gb'] < 8:
            issues_found.append(f"Low system RAM: {status['system']['available_gb']:.2f} GB available")

        if not issues_found:
            print("✅ No major memory issues detected")
        else:
            print("⚠️ Issues found:")
            for issue in issues_found:
                print(f"  • {issue}")

        print("\n💡 RECOMMENDATIONS:")
        print("  • Use PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        print("  • Clear cache between model loads: torch.cuda.empty_cache()")
        print("  • Load models sequentially, not simultaneously")
        print("  • Consider closing GPU-intensive background applications")
        print("  • Use --gpu-memory-fraction 0.9 when running models")

    def reset_environment(self):
        """Reset environment to original state"""
        if self.original_alloc_config:
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = self.original_alloc_config
        else:
            os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)

        print("🔄 Environment reset to original state")

def main():
    """Main optimization function"""
    print("🎯 Senter-Omni GPU Memory Optimizer")
    print("=" * 60)

    optimizer = GPUMemoryOptimizer()

    # Apply optimizations
    optimizer.set_optimized_environment()
    optimizer.clear_gpu_cache(thorough=True)
    optimizer.optimize_for_large_models()

    # Show status
    optimizer.print_memory_status()

    # Diagnose issues
    optimizer.diagnose_memory_issues()

    print("\n🎉 GPU MEMORY OPTIMIZATION COMPLETE!")
    print("Your system is now optimized for large model loading.")
    print("\n💡 To use in your code:")
    print("   from gpu_memory_optimizer import GPUMemoryOptimizer")
    print("   optimizer = GPUMemoryOptimizer()")
    print("   optimizer.set_optimized_environment()")

if __name__ == "__main__":
    main()
