import os
os.environ["TRITON_CACHE_DIR"] = "C:/triton_cache"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "C:/torch_cache"
import torch

torch.set_float32_matmul_precision('medium')

from ops.fc_p3m0 import fused_gelu_fcgp_norm_3d
from tests.baselines import gelu_fcgp_norm_3d_torch
from tests.utils import run_correctness_test, run_benchmark
from tests.config_loader import load_config


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = load_config()
    rep = config['tests']['rep']
    batch_size = config['tests']['batch_size']
    num_features = config['tests']['num_features']

    x = torch.randn(8, batch_size, num_features, device=device).contiguous()
    y = torch.randn(8, batch_size, num_features, device=device).contiguous()
    weight = torch.randn(20, num_features, num_features, device=device).contiguous()

    run_correctness_test(fused_gelu_fcgp_norm_3d, gelu_fcgp_norm_3d_torch, {'x': x, 'y': y, 'weight': weight})
    run_benchmark(fused_gelu_fcgp_norm_3d, gelu_fcgp_norm_3d_torch, (x, y, weight), rep, verbose=True)