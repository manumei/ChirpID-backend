# Performance Monitor - DataLoader Performance Analysis
# Monitors and benchmarks DataLoader performance and utilization

import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.dataloader_factory import OptimalDataLoaderFactory


class DataLoaderPerformanceMonitor:
    """Monitor DataLoader performance and utilization."""
    
    def __init__(self):
        self.metrics = {
            'batch_loading_times': [],
            'gpu_utilization': [],
            'memory_usage': []
        }
    
    def time_batch_loading(self, dataloader, num_batches=10):
        """
        Time batch loading performance.
        
        Args:
            dataloader: PyTorch DataLoader to benchmark
            num_batches (int): Number of batches to time
            
        Returns:
            dict: Performance metrics
        """
        times = []
        start_total = time.time()
        
        print(f"Timing {num_batches} batches...")
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
                
            start_time = time.time()
            # Force tensor materialization and transfer to GPU if available
            if torch.cuda.is_available():
                _ = batch[0].cuda().shape
                torch.cuda.synchronize()  # Ensure GPU operations complete
            else:
                _ = batch[0].shape
            end_time = time.time()
            
            times.append(end_time - start_time)
            
        end_total = time.time()
        
        return {
            'mean_batch_time': np.mean(times),
            'std_batch_time': np.std(times),
            'min_batch_time': np.min(times),
            'max_batch_time': np.max(times),
            'total_time': end_total - start_total,
            'batches_per_second': num_batches / (end_total - start_total)
        }
    
    def benchmark_configurations(self, dataset, batch_size, num_batches=20):
        """
        Benchmark different DataLoader configurations.
        
        Args:
            dataset: PyTorch dataset to benchmark
            batch_size (int): Batch size to use
            num_batches (int): Number of batches per configuration
            
        Returns:
            dict: Results for each configuration
        """
        configs = [
            {
                'name': 'Single-threaded (baseline)',
                'config': {'num_workers': 0, 'pin_memory': False}
            },
            {
                'name': 'Conservative (4 workers)',
                'config': {'num_workers': 4, 'pin_memory': True, 'persistent_workers': True}
            },
            {
                'name': 'Optimized (8 workers)',
                'config': {'num_workers': 8, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 4}
            },
            {
                'name': 'Aggressive (12 workers)',
                'config': {'num_workers': 12, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 6}
            }
        ]
        
        results = {}
        
        for config_info in configs:
            name = config_info['name']
            config = config_info['config']
            
            print(f"\\n{'='*50}")
            print(f"Testing: {name}")
            print(f"Config: {config}")
            
            try:
                loader = DataLoader(
                    dataset, 
                    batch_size=batch_size, 
                    shuffle=False,  # Disable shuffle for consistent timing
                    **config
                )
                
                metrics = self.time_batch_loading(loader, num_batches)
                results[name] = {**config, **metrics}
                
                print(f"Results:")
                print(f"  Mean batch time: {metrics['mean_batch_time']:.4f}s")
                print(f"  Batches per second: {metrics['batches_per_second']:.2f}")
                print(f"  Total time: {metrics['total_time']:.2f}s")
                
            except Exception as e:
                print(f"Configuration failed: {e}")
                results[name] = {'error': str(e), **config}
        
        return results
    
    def compare_optimized_vs_baseline(self, dataset, batch_size, has_augmentation=False, has_standardization=False):
        """
        Compare optimized configuration against baseline.
        
        Args:
            dataset: PyTorch dataset
            batch_size (int): Batch size
            has_augmentation (bool): Whether dataset has augmentation  
            has_standardization (bool): Whether dataset has standardization
            
        Returns:
            dict: Comparison results
        """
        print("\\n" + "="*60)
        print("OPTIMIZED vs BASELINE COMPARISON")
        print("="*60)
        
        # Baseline configuration (conservative)
        baseline_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        # Optimized configuration using factory
        optimized_loader = OptimalDataLoaderFactory.create_training_loader(
            dataset,
            batch_size=batch_size,
            has_augmentation=has_augmentation,
            has_standardization=has_standardization,  
            shuffle=False  # Disable for consistent timing
        )
        
        print("\\nTesting baseline configuration...")
        baseline_metrics = self.time_batch_loading(baseline_loader, 20)
        
        print("\\nTesting optimized configuration...")
        optimized_metrics = self.time_batch_loading(optimized_loader, 20)
        
        # Calculate improvements
        improvement = {
            'speedup_factor': baseline_metrics['mean_batch_time'] / optimized_metrics['mean_batch_time'],
            'time_reduction_percent': ((baseline_metrics['mean_batch_time'] - optimized_metrics['mean_batch_time']) / baseline_metrics['mean_batch_time']) * 100,
            'throughput_increase': optimized_metrics['batches_per_second'] / baseline_metrics['batches_per_second']
        }
        
        results = {
            'baseline': baseline_metrics,
            'optimized': optimized_metrics,
            'improvement': improvement
        }
        
        # Print comparison
        print(f"\\n{'='*60}")
        print("PERFORMANCE COMPARISON RESULTS")
        print(f"{'='*60}")
        print(f"Baseline mean batch time:  {baseline_metrics['mean_batch_time']:.4f}s")
        print(f"Optimized mean batch time: {optimized_metrics['mean_batch_time']:.4f}s")
        print(f"Speedup factor: {improvement['speedup_factor']:.2f}x")
        print(f"Time reduction: {improvement['time_reduction_percent']:.1f}%")
        print(f"Throughput increase: {improvement['throughput_increase']:.2f}x")
        
        return results
    
    def plot_benchmark_results(self, results, save_path=None):
        """
        Plot benchmark results for visualization.
        
        Args:
            results (dict): Results from benchmark_configurations
            save_path (str, optional): Path to save the plot
        """
        # Extract data for plotting
        names = []
        batch_times = []
        throughputs = []
        
        for name, metrics in results.items():
            if 'error' in metrics:
                continue
            names.append(name)
            batch_times.append(metrics['mean_batch_time'])
            throughputs.append(metrics['batches_per_second'])
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Batch time comparison
        bars1 = ax1.bar(names, batch_times, color=['red', 'orange', 'green', 'blue'])
        ax1.set_title('Mean Batch Loading Time')
        ax1.set_ylabel('Time (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, batch_times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}s', ha='center', va='bottom')
        
        # Throughput comparison  
        bars2 = ax2.bar(names, throughputs, color=['red', 'orange', 'green', 'blue'])
        ax2.set_title('Throughput (Batches per Second)')
        ax2.set_ylabel('Batches/Second')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars2, throughputs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()


def quick_benchmark(dataset, batch_size=24, has_augmentation=False, has_standardization=False):
    """
    Quick benchmark function for easy testing.
    
    Args:
        dataset: PyTorch dataset
        batch_size (int): Batch size to test
        has_augmentation (bool): Whether augmentation is used
        has_standardization (bool): Whether standardization is used
    
    Returns:
        dict: Benchmark results
    """
    monitor = DataLoaderPerformanceMonitor()
    
    print(f"\\nQuick DataLoader Benchmark")
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Has augmentation: {has_augmentation}")  
    print(f"Has standardization: {has_standardization}")
    
    # Run comparison
    results = monitor.compare_optimized_vs_baseline(
        dataset, batch_size, has_augmentation, has_standardization
    )
    
    return results
