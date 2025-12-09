import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import triton

from contextlib import contextmanager


@contextmanager
def measure_memory():
    peak = [None]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        yield peak
        torch.cuda.synchronize()
        peak[0] = torch.cuda.max_memory_allocated() / 1024**2
    else:
        yield peak
        peak[0] = 0.0  # Memory tracking not available on CPU
    
    
def print_benchmark_results(avg_time_fused, avg_time_torch, mem_fused, mem_torch, title=""):
    header_length_pad = max(0, 35 - len(title) - 4)
    print(f"\n┌─ {title} " + "─"*header_length_pad + "┐")
    print("│")
    print(f"│  Runtime:")
    print(f"│    Fused Kernel  : {avg_time_fused:>8.2f} ms")
    print(f"│    PyTorch       : {avg_time_torch:>8.2f} ms")
    print(f"│    Speedup       : {avg_time_torch / avg_time_fused:>8.2f}×")
    print("│")
    print(f"│  Memory Usage:")
    print(f"│    Fused Kernel  : {mem_fused:>8.2f} MB")
    print(f"│    PyTorch       : {mem_torch:>8.2f} MB")
    print(f"│    Memory Ratio  : {mem_fused / mem_torch:>8.2f}×")
    print("│")
    print("└" + "─"*34 + "┘\n")


def run_benchmark(triton_fn, torch_fn, args, rep, warmup=200, verbose=True, return_mode='mean'):
    """Run forward and forward+backward benchmarks."""
    # Forward-only benchmark
    avg_time_fused = triton.testing.do_bench(lambda: triton_fn(*args), warmup, rep, return_mode=return_mode)
    avg_time_torch = triton.testing.do_bench(lambda: torch_fn(*args), warmup, rep, return_mode=return_mode)
    with measure_memory() as mem_fused_fwd: _ = triton_fn(*args)
    with measure_memory() as mem_torch_fwd: _ = torch_fn(*args)

    # Forward + backward benchmark
    args = [arg.clone().detach().requires_grad_(True) for arg in args]
    avg_time_fused_bwd = triton.testing.do_bench(lambda: triton_fn(*args).sum().backward(), warmup, rep, return_mode=return_mode)
    avg_time_torch_bwd = triton.testing.do_bench(lambda: torch_fn(*args).sum().backward(), warmup, rep, return_mode=return_mode)
    with measure_memory() as mem_fused_bwd: _ = triton_fn(*args)
    with measure_memory() as mem_torch_bwd: _ = torch_fn(*args)

    if verbose:
        print_benchmark_results(
            avg_time_fused, avg_time_torch, mem_fused_fwd[0], mem_torch_fwd[0],
            title="Forward Pass"
        )
        print_benchmark_results(
            avg_time_fused_bwd, avg_time_torch_bwd, mem_fused_bwd[0], mem_torch_bwd[0],
            title="Forward + Backward Pass"
        )
            
    return avg_time_fused, avg_time_torch, mem_fused_fwd[0], mem_torch_fwd[0], avg_time_fused_bwd, avg_time_torch_bwd, mem_fused_bwd[0], mem_torch_bwd[0]


def run_correctness_test(triton_fn, torch_fn, kwargs):
    """Run forward and backward correctness test."""
    # Forward correctness check
    out_triton = triton_fn(**kwargs)
    out_torch = torch_fn(**kwargs)

    max_diff = (out_torch - out_triton).abs().max().item()
    is_correct = torch.allclose(out_torch, out_triton, atol=1e-5)
    check_mark = " ✔" if is_correct else " ✘"
    print(f"Max absolute difference (fwd): {max_diff:.1e}{check_mark}")

    # Backward correctness check
    kwargs = {k: v.clone().detach().requires_grad_(True) for k, v in kwargs.items()}
    
    out_torch = torch_fn(**kwargs)
    out_triton = triton_fn(**kwargs)

    grad_output = torch.randn_like(out_torch).contiguous()
    out_torch.backward(grad_output)
    out_triton.backward(grad_output)

    for name, arg in kwargs.items():
        grad_diff = (arg.grad - arg.grad).abs().max().item()
        grad_correct = torch.allclose(arg.grad, arg.grad, atol=1e-2)
        print(f"grad {name} max diff: {grad_diff:.1e}" + (" ✔" if grad_correct else " ✘"))


def plot_heatmap(results, metric_key, title, save_path, cmap='RdYlGn', invert_cmap=False, vmin=None, vmax=None):
    """Heatmap visualization of benchmark results."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    batch_sizes = sorted(set(r['batch_size'] for r in results))
    num_features_list = sorted(set(r['num_features'] for r in results))

    matrix = np.zeros((len(batch_sizes), len(num_features_list)))
    for r in results:
        i = batch_sizes.index(r['batch_size'])
        j = num_features_list.index(r['num_features'])
        matrix[i, j] = r[metric_key] if r[metric_key] is not None else 0

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap_name = f'{cmap}_r' if invert_cmap else cmap
    im = ax.imshow(matrix, cmap=cmap_name, aspect='auto', vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(num_features_list)))
    ax.set_yticks(np.arange(len(batch_sizes)))
    ax.set_xticklabels(num_features_list)
    ax.set_yticklabels(batch_sizes)

    ax.set_xlabel('Number of Features', fontsize=12)
    ax.set_ylabel('Batch Size', fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)

    cbar = plt.colorbar(im, ax=ax)
    cbar_label = 'Speedup (x)' if 'speedup' in metric_key else 'Memory Ratio (x)'
    cbar.set_label(cbar_label, fontsize=12)

    # Add text annotations
    for i in range(len(batch_sizes)):
        for j in range(len(num_features_list)):
            value = matrix[i, j]
            text_str = f'{value:.2f}x' if value > 0 else 'OOM'
            ax.text(j, i, text_str, ha="center", va="center", color="black", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {save_path}")
    plt.close()


def print_results_table(results, title):
    """Print results as a formatted table."""
    separator = "=" * 140
    divider = "-" * 140

    # Forward pass results
    print(f"\n{separator}")
    print(f"FORWARD PASS RESULTS - {title}")
    print(separator)
    print(f"{'Batch':<8} {'Features':<10} {'Fused (ms)':<12} {'Torch (ms)':<12} "
          f"{'Speedup':<10} {'Max Diff':<12} {'Correct':<8}")
    print(divider)

    for r in results:
        speedup_str = f"{r['speedup_fwd']:.2f}x" if r['speedup_fwd'] else "N/A"
        correct_mark = '✔' if r['is_correct'] else '✘'
        print(f"{r['batch_size']:<8} {r['num_features']:<10} {r['time_fwd_fused']:<12.2f} "
              f"{r['time_fwd_torch']:<12.2f} {speedup_str:<10} {r['max_diff']:<12.1e} "
              f"{correct_mark:<8}")

    print(separator)

    # Forward + backward pass results
    print(f"\n{separator}")
    print(f"FORWARD + BACKWARD PASS RESULTS - {title}")
    print(separator)
    print(f"{'Batch':<8} {'Features':<10} {'Fused (ms)':<12} {'Torch (ms)':<12} {'Speedup':<10}")
    print(divider)

    for r in results:
        fused_time = f"{r['time_fwd_bwd_fused']:.2f}" if r['time_fwd_bwd_fused'] else "OOM"
        torch_time = f"{r['time_fwd_bwd_torch']:.2f}" if r['time_fwd_bwd_torch'] else "OOM"
        speedup = f"{r['speedup_fwd_bwd']:.2f}x" if r['speedup_fwd_bwd'] else "N/A"
        print(f"{r['batch_size']:<8} {r['num_features']:<10} {fused_time:<12} "
              f"{torch_time:<12} {speedup:<10}")

    print(separator)

    # Forward memory usage
    print(f"\n{separator}")
    print(f"FORWARD MEMORY USAGE - {title}")
    print(separator)
    print(f"{'Batch':<8} {'Features':<10} {'Fused (MB)':<12} {'Torch (MB)':<12} {'Ratio':<10}")
    print(divider)

    for r in results:
        fused_mem = f"{r['mem_fwd_fused']:.2f}" if r['mem_fwd_fused'] else "OOM"
        torch_mem = f"{r['mem_fwd_torch']:.2f}" if r['mem_fwd_torch'] else "OOM"
        ratio = f"{r['mem_ratio_fwd']:.2f}x" if r['mem_ratio_fwd'] else "N/A"
        print(f"{r['batch_size']:<8} {r['num_features']:<10} {fused_mem:<12} "
              f"{torch_mem:<12} {ratio:<10}")

    print(separator)

    # Forward + backward memory usage
    print(f"\n{separator}")
    print(f"FORWARD + BACKWARD MEMORY USAGE - {title}")
    print(separator)
    print(f"{'Batch':<8} {'Features':<10} {'Fused (MB)':<12} {'Torch (MB)':<12} {'Ratio':<10}")
    print(divider)

    for r in results:
        fused_mem = f"{r['mem_fwd_bwd_fused']:.2f}" if r['mem_fwd_bwd_fused'] else "OOM"
        torch_mem = f"{r['mem_fwd_bwd_torch']:.2f}" if r['mem_fwd_bwd_torch'] else "OOM"
        ratio = f"{r['mem_ratio_fwd_bwd']:.2f}x" if r['mem_ratio_fwd_bwd'] else "N/A"
        print(f"{r['batch_size']:<8} {r['num_features']:<10} {fused_mem:<12} "
              f"{torch_mem:<12} {ratio:<10}")

    print(separator)


def save_results_csv(results, path):
    """Save benchmark results to CSV file."""
    csv_path = os.path.join(path, 'results.csv')
    os.makedirs(path, exist_ok=True)
    
    if not results:
        return
    
    fieldnames = [
        'batch_size', 'num_features',
        'time_fwd_fused', 'time_fwd_torch', 'speedup_fwd',
        'time_fwd_bwd_fused', 'time_fwd_bwd_torch', 'speedup_fwd_bwd',
        'mem_fwd_fused', 'mem_fwd_torch', 'mem_ratio_fwd',
        'mem_fwd_bwd_fused', 'mem_fwd_bwd_torch', 'mem_ratio_fwd_bwd',
        'max_diff', 'is_correct'
    ]
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to: {csv_path}")


def run_single_benchmark(triton_fn, torch_fn, setup_fn, batch_size, num_features, rep, warmup=200, verify_correctness=True, return_mode='mean'):
    """Run a single benchmark configuration."""
    args = setup_fn(batch_size, num_features)

    out_triton = triton_fn(*args)
    out_torch = torch_fn(*args)

    if verify_correctness:
        is_correct = torch.allclose(out_torch, out_triton, atol=1e-5)
        max_diff = (out_torch - out_triton).abs().max().item()
    else:
        is_correct = False
        max_diff = 1e10

    time_fwd_fused, time_fwd_torch, mem_fwd_fused, mem_fwd_torch, \
    time_fwd_bwd_fused, time_fwd_bwd_torch, mem_fwd_bwd_fused, mem_fwd_bwd_torch = \
        run_benchmark(
            triton_fn, torch_fn, args, rep, 
            warmup=warmup, verbose=False, return_mode=return_mode
        )

    return {
        'batch_size': batch_size,
        'num_features': num_features,
        'time_fwd_fused': time_fwd_fused,
        'time_fwd_torch': time_fwd_torch,
        'speedup_fwd': time_fwd_torch / time_fwd_fused if time_fwd_fused else None,
        'time_fwd_bwd_fused': time_fwd_bwd_fused,
        'time_fwd_bwd_torch': time_fwd_bwd_torch,
        'speedup_fwd_bwd': time_fwd_bwd_torch / time_fwd_bwd_fused if time_fwd_bwd_fused else None,
        'mem_fwd_fused': mem_fwd_fused,
        'mem_fwd_torch': mem_fwd_torch,
        'mem_ratio_fwd': mem_fwd_fused / mem_fwd_torch if mem_fwd_torch else None,
        'mem_fwd_bwd_fused': mem_fwd_bwd_fused,
        'mem_fwd_bwd_torch': mem_fwd_bwd_torch,
        'mem_ratio_fwd_bwd': mem_fwd_bwd_fused / mem_fwd_bwd_torch if mem_fwd_bwd_torch else None,
        'max_diff': max_diff,
        'is_correct': is_correct,
    }
    

def run_sweep(triton_fn, torch_fn, setup_fn,
              batch_sizes=[1024, 2048, 4096, 8192],
              num_features_list=[128, 256, 512, 1024],
              rep=1000):
    """Run benchmark sweep across batch sizes and feature dimensions."""
    results = []

    print("Running benchmark sweep...")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Num features: {num_features_list}")

    for batch_size in batch_sizes:
        for num_features in num_features_list:
            print(f"Running batch_size={batch_size}, num_features={num_features}...", end=" ")
            result = run_single_benchmark(triton_fn, torch_fn, setup_fn, batch_size, num_features, rep)
            results.append(result)

            fwd_msg = (f"Fwd: {result['speedup_fwd']:.2f}x" 
                      if result['speedup_fwd'] else "Fwd: OOM")
            bwd_msg = (f"Fwd+Bwd: {result['speedup_fwd_bwd']:.2f}x" 
                      if result['speedup_fwd_bwd'] else "Fwd+Bwd: OOM")
            print(f"{fwd_msg}, {bwd_msg}")

    return results