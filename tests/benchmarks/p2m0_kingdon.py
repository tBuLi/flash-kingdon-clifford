import os
os.environ["TRITON_CACHE_DIR"] = "C:/triton_cache"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "C:/torch_cache"
import torch
torch.set_float32_matmul_precision('medium')
torch._dynamo.config.cache_size_limit = 512

from ops.p2m0_kingdon import fused_gelu_sgp_norm_2d
from tests.baselines import gelu_sgp_norm_2d_torch
from tests.utils import plot_heatmap, print_results_table, run_sweep, save_results_csv
from tests.config_loader import load_config


def setup_benchmark(batch_size, num_features):
    x = torch.randn(4, batch_size, num_features).cuda().contiguous()
    y = torch.randn(4, batch_size, num_features).cuda().contiguous()
    weight = torch.randn(num_features, 10).cuda().contiguous()
    return x, y, weight


if __name__ == "__main__":
    assert torch.cuda.is_available()

    config = load_config()
    path = "tests/benchmarks/results/p2m0_kingdon"

    results = run_sweep(
        fused_gelu_sgp_norm_2d,
        gelu_sgp_norm_2d_torch,
        setup_benchmark,
        batch_sizes=config['benchmarks']['batch_sizes'],
        num_features_list=config['benchmarks']['num_features_list'],
        rep=config['benchmarks']['rep']
    )

    print_results_table(results, "p2m0_kingdon")
    save_results_csv(results, path)

    plot_heatmap(results, 'speedup_fwd', 'Forward Pass Speedup: Triton vs PyTorch\nCl(2,0)',
                 path + '/speedup/fwd.png', vmin=1, vmax=6)
    plot_heatmap(results, 'speedup_fwd_bwd', 'Forward + Backward Pass Speedup: Triton vs PyTorch\nCl(2,0)',
                 path + '/speedup/fwd_bwd.png', vmin=1, vmax=6)
    plot_heatmap(results, 'mem_ratio_fwd', 'Forward Pass Memory Ratio: Fused / PyTorch\nCl(2,0)',
                 path + '/memory/fwd.png', invert_cmap=True, vmin=0, vmax=1)
    plot_heatmap(results, 'mem_ratio_fwd_bwd', 'Forward + Backward Pass Memory Ratio: Fused / PyTorch\nCl(2,0)',
                 path + '/memory/fwd_bwd.png', invert_cmap=True, vmin=0, vmax=1)
