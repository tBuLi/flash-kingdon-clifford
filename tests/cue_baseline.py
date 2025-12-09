import os
os.environ["TRITON_CACHE_DIR"] = "C:/triton_cache"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "C:/torch_cache"
import torch
import cuequivariance as cue
import cuequivariance_torch as cuet


def mvgelu(x):
    """Apply GELU activation gated by scalar component."""
    # x has shape (B, N * 8)
    b = x.shape[0]
    x = x.view(b, -1, 8)  # (B, N, 8)
    s = x[..., 0:1]  # scalar part
    gate = 0.5 * (1 + torch.erf(s * 0.7071067811865475))
    y = gate * x
    y = y.view(b, -1)  # (B, N * 8)
    return y


def initialize_linear(N: int):
    """
    Initialize MLP with linear layer + weighted GP + GELU + BatchNorm.
    Related: https://github.com/NVIDIA/cuEquivariance/issues/194
    """
    irreps = cue.Irreps("O3", f"{N}x0e + {N}x1o + {N}x1e + {N}x0o")

    ep_weighted = cue.descriptors.fully_connected_tensor_product(
        irreps.set_mul(1),
        irreps.set_mul(1),
        irreps.set_mul(1)
    )

    [(_, stp_weighted)] = ep_weighted.polynomial.operations
    stp_weighted = stp_weighted.append_modes_to_all_operands("n", dict(n=N))
    p_weighted = cue.SegmentedPolynomial.eval_last_operand(stp_weighted)

    weighted_gp = cuet.SegmentedPolynomial(p_weighted, method="uniform_1d").cuda()
    linear = cuet.Linear(irreps_in=irreps, irreps_out=irreps, layout=cue.ir_mul, layout_in=cue.ir_mul, layout_out=cue.ir_mul).cuda()
    norm = cuet.layers.BatchNorm(irreps=irreps, layout=cue.ir_mul).cuda()
    
    @torch.compile
    def mlp(x, w):
        y = linear(x)
        x = mvgelu(x)
        y = mvgelu(y)
        [x] = weighted_gp([w, x, y])
        x = norm(x)
        return x
    
    return mlp