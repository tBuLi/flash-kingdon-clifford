import torch
import triton
import triton.language as tl
from .vga2d import weighted_gp_kernel, gate_kernel, weighted_gp_grad_kernel

MV_DIM = 4
NUM_GRADES = 3
NUM_PRODUCT_WEIGHTS = 10
EPS = 1e-6

# tuned at RTX 4500
DEFAULT_BATCH_BLOCK = 4
DEFAULT_FEATURE_BLOCK = 128
DEFAULT_NUM_WARPS = 16
DEFAULT_NUM_STAGES = 1


@triton.jit
def compute_gelu_gate(x):
    """Compute the GELU gate Φ(x) := 0.5 * (1 + erf(x / sqrt(2)))"""
    return 0.5 * (1 + tl.erf(x.to(tl.float32) * 0.7071067811865475)).to(x.dtype)


@triton.jit
def compute_gelu_gate_grad(x):
    """Compute the gradient of the GELU gate = 1/sqrt(2pi) * exp(-x^2/2)"""
    return 0.3989422804 * tl.exp(-0.5 * x * x)


@triton.jit
def gelu_wgp_norm_kernel_fwd(
    x_ptr,
    y_ptr,
    output_ptr,
    weights_ptr,
    pnorm_ptr,
    NORMALIZE: tl.constexpr,
    batch_size: tl.constexpr,
    n_features: tl.constexpr,
    BATCH_BLOCK: tl.constexpr,
    FEATURE_BLOCK: tl.constexpr,
    NUM_PRODUCT_WEIGHTS: tl.constexpr,
):
    """
    Apply GELU non-linearity to inputs, compute weighted geometric product,
    and accumulate squared norms for grade-wise RMSNorm.
    """
    batch_block_id = tl.program_id(axis=0)
    thread_block_id = tl.program_id(axis=1)

    batch_ids = batch_block_id*BATCH_BLOCK + tl.arange(0, BATCH_BLOCK)
    feature_ids = thread_block_id*FEATURE_BLOCK + tl.arange(0, FEATURE_BLOCK)

    batch_mask = batch_ids < batch_size
    feature_mask = feature_ids < n_features
    batch_feature_mask = batch_mask[:, None] & feature_mask[None, :]

    stride_component = batch_size * n_features
    base_offset = batch_ids[:, None] * n_features + feature_ids[None, :]

    weight_offset = feature_ids * NUM_PRODUCT_WEIGHTS

    w0 = tl.load(weights_ptr + weight_offset + 0, mask=feature_mask)
    w1 = tl.load(weights_ptr + weight_offset + 1, mask=feature_mask)
    w2 = tl.load(weights_ptr + weight_offset + 2, mask=feature_mask)
    w3 = tl.load(weights_ptr + weight_offset + 3, mask=feature_mask)
    w4 = tl.load(weights_ptr + weight_offset + 4, mask=feature_mask)
    w5 = tl.load(weights_ptr + weight_offset + 5, mask=feature_mask)
    w6 = tl.load(weights_ptr + weight_offset + 6, mask=feature_mask)
    w7 = tl.load(weights_ptr + weight_offset + 7, mask=feature_mask)
    w8 = tl.load(weights_ptr + weight_offset + 8, mask=feature_mask)
    w9 = tl.load(weights_ptr + weight_offset + 9, mask=feature_mask)

    x0 = tl.load(x_ptr + 0 * stride_component + base_offset, mask=batch_feature_mask)
    x1 = tl.load(x_ptr + 1 * stride_component + base_offset, mask=batch_feature_mask)
    x2 = tl.load(x_ptr + 2 * stride_component + base_offset, mask=batch_feature_mask)
    x3 = tl.load(x_ptr + 3 * stride_component + base_offset, mask=batch_feature_mask)

    y0 = tl.load(y_ptr + 0 * stride_component + base_offset, mask=batch_feature_mask)
    y1 = tl.load(y_ptr + 1 * stride_component + base_offset, mask=batch_feature_mask)
    y2 = tl.load(y_ptr + 2 * stride_component + base_offset, mask=batch_feature_mask)
    y3 = tl.load(y_ptr + 3 * stride_component + base_offset, mask=batch_feature_mask)
    
    # Apply GELU gate
    gate_x = compute_gelu_gate(x0)
    gate_y = compute_gelu_gate(y0)

    xvals = gate_kernel((x0,x1,x2,x3), (gate_x,))  # X * GATE_X
    yvals = gate_kernel((y0,y1,y2,y3), (gate_y,))  # Y * GATE_Y
    wvals = (w0, w1, w2, w3, w4, w5, w6, w7, w8, w9)
    o0,o1,o2,o3 = weighted_gp_kernel(xvals, yvals, (wvals,))
    
    if NORMALIZE:
        pn_scalar = tl.sum(o0 * o0, axis=1) / n_features
        pn_vector = tl.sum(o1*o1 + o2*o2, axis=1) / n_features
        pn_pseudo = tl.sum(o3 * o3, axis=1) / n_features

        tl.atomic_add(pnorm_ptr + 0*batch_size + batch_ids, pn_scalar, mask=batch_mask)
        tl.atomic_add(pnorm_ptr + 1*batch_size + batch_ids, pn_vector, mask=batch_mask)
        tl.atomic_add(pnorm_ptr + 2*batch_size + batch_ids, pn_vector, mask=batch_mask)
        tl.atomic_add(pnorm_ptr + 3*batch_size + batch_ids, pn_pseudo, mask=batch_mask)

    tl.store(output_ptr + 0 * stride_component + base_offset, o0, mask=batch_feature_mask)
    tl.store(output_ptr + 1 * stride_component + base_offset, o1, mask=batch_feature_mask)
    tl.store(output_ptr + 2 * stride_component + base_offset, o2, mask=batch_feature_mask)
    tl.store(output_ptr + 3 * stride_component + base_offset, o3, mask=batch_feature_mask)


@triton.jit
def normalize_with_sqrt_kernel(
    output_ptr,
    pnorm_ptr,
    batch_size: tl.constexpr,
    n_features: tl.constexpr,
    BATCH_BLOCK: tl.constexpr,
    FEATURE_BLOCK: tl.constexpr,
    MV_DIM: tl.constexpr,
    EPS: tl.constexpr,
):
    """Normalize the output by dividing each grade with root of its accumulated norm."""
    batch_block_id = tl.program_id(axis=0)
    thread_block_id = tl.program_id(axis=1)

    batch_ids = batch_block_id*BATCH_BLOCK + tl.arange(0, BATCH_BLOCK)
    feature_ids = thread_block_id*FEATURE_BLOCK + tl.arange(0, FEATURE_BLOCK)

    batch_mask = batch_ids < batch_size
    feature_mask = feature_ids < n_features
    batch_feature_mask = batch_mask[:, None, None] & feature_mask[None, :, None]

    component_ids = tl.arange(0, MV_DIM)[None, None, :]

    feature_offset = (component_ids * batch_size * n_features + 
                     batch_ids[:, None, None] * n_features + 
                     feature_ids[None, :, None])
    
    norm_indices = component_ids * batch_size + batch_ids[:, None, None]

    pnorm = tl.load(pnorm_ptr + norm_indices, mask=batch_mask[:, None, None])
    mv = tl.load(output_ptr + feature_offset, mask=batch_feature_mask)

    norm = tl.sqrt(pnorm + EPS)
    mv_normalized = mv / norm

    tl.store(output_ptr + feature_offset, mv_normalized, mask=batch_feature_mask)


def gelu_geometric_product_norm_fwd(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    normalize: bool,
) -> torch.Tensor:
    """Fused operation: GELU non-linearity, weighted geometric product, and grade-wise RMSNorm."""
    assert x.shape == y.shape
    assert x.shape[0] == MV_DIM
    assert x.shape[2] == weight.shape[0]
    assert weight.shape[1] == NUM_PRODUCT_WEIGHTS

    _, B, N = x.shape

    BATCH_BLOCK = min(DEFAULT_BATCH_BLOCK, B)
    FEATURE_BLOCK = min(DEFAULT_FEATURE_BLOCK, N)

    num_blocks_batch = triton.cdiv(B, BATCH_BLOCK)
    num_blocks_features = triton.cdiv(N, FEATURE_BLOCK)

    output = torch.empty_like(x)
    partial_norm = (torch.zeros((MV_DIM, B), device=x.device, dtype=x.dtype) if normalize 
                   else torch.zeros((1,), device=x.device, dtype=x.dtype))

    grid = (num_blocks_batch, num_blocks_features)

    gelu_wgp_norm_kernel_fwd[grid](
        x,
        y,
        output,
        weight,
        partial_norm,
        normalize,
        B,
        N,
        BATCH_BLOCK,
        FEATURE_BLOCK,
        NUM_PRODUCT_WEIGHTS,
        num_warps=DEFAULT_NUM_WARPS,
        num_stages=DEFAULT_NUM_STAGES,
    )

    if normalize:
        normalize_with_sqrt_kernel[grid](
            output,
            partial_norm,
            B,
            N,
            BATCH_BLOCK,
            FEATURE_BLOCK,
            MV_DIM,
            EPS,
            num_warps=DEFAULT_NUM_WARPS,
            num_stages=DEFAULT_NUM_STAGES,
        )

    return output, partial_norm


@triton.jit
def grad_o_dot_o_kernel(
    dot_ptr,
    pnorm_ptr,
    output_ptr,
    grad_output_ptr,
    batch_size: tl.constexpr,
    n_features: tl.constexpr,
    BATCH_BLOCK: tl.constexpr,
    FEATURE_BLOCK: tl.constexpr,
    EPS: tl.constexpr,
):
    """Compute the dot product of grad_output and output for each grade, accumulate across all features."""
    batch_block_id = tl.program_id(axis=0)
    thread_block_id = tl.program_id(axis=1)

    batch_ids = batch_block_id*BATCH_BLOCK + tl.arange(0, BATCH_BLOCK)
    feature_ids = thread_block_id*FEATURE_BLOCK + tl.arange(0, FEATURE_BLOCK)

    batch_mask = batch_ids < batch_size
    feature_mask = feature_ids < n_features
    batch_feature_mask = batch_mask[:, None] & feature_mask[None, :]

    stride_component = batch_size * n_features
    offset = batch_ids[:, None] * n_features + feature_ids[None, :]

    go0 = tl.load(grad_output_ptr + 0 * stride_component + offset, mask=batch_feature_mask)
    go1 = tl.load(grad_output_ptr + 1 * stride_component + offset, mask=batch_feature_mask)
    go2 = tl.load(grad_output_ptr + 2 * stride_component + offset, mask=batch_feature_mask)
    go3 = tl.load(grad_output_ptr + 3 * stride_component + offset, mask=batch_feature_mask)

    pn_scalar = tl.load(pnorm_ptr + 0*batch_size + batch_ids, mask=batch_mask)[:, None]
    pn_vector = tl.load(pnorm_ptr + 1*batch_size + batch_ids, mask=batch_mask)[:, None]
    pn_pseudo = tl.load(pnorm_ptr + 3*batch_size + batch_ids, mask=batch_mask)[:, None]

    o0 = tl.load(output_ptr + 0 * stride_component + offset, mask=batch_feature_mask)
    o1 = tl.load(output_ptr + 1 * stride_component + offset, mask=batch_feature_mask)
    o2 = tl.load(output_ptr + 2 * stride_component + offset, mask=batch_feature_mask)
    o3 = tl.load(output_ptr + 3 * stride_component + offset, mask=batch_feature_mask)

    rms_scalar = tl.sqrt(pn_scalar + EPS)
    rms_vector = tl.sqrt(pn_vector + EPS)
    rms_pseudo = tl.sqrt(pn_pseudo + EPS)

    dot_scalar = tl.sum(rms_scalar * go0 * o0, axis=1)
    dot_vector = tl.sum(rms_vector * (go1*o1 + go2*o2), axis=1)
    dot_pseudo = tl.sum(rms_pseudo * go3 * o3, axis=1)

    tl.atomic_add(dot_ptr + 0*batch_size + batch_ids, dot_scalar, mask=batch_mask)
    tl.atomic_add(dot_ptr + 1*batch_size + batch_ids, dot_vector, mask=batch_mask)
    tl.atomic_add(dot_ptr + 2*batch_size + batch_ids, dot_pseudo, mask=batch_mask)


@triton.jit
def gelu_wgp_norm_kernel_bwd(
    x_ptr,
    y_ptr,
    output_ptr,
    weights_ptr,
    dot_ptr,
    pnorm_ptr,
    grad_output_ptr,
    grad_x_ptr,
    grad_y_ptr,
    grad_weight_ptr,
    NORMALIZE: tl.constexpr,
    batch_size: tl.constexpr,
    n_features: tl.constexpr,
    BATCH_BLOCK: tl.constexpr,
    FEATURE_BLOCK: tl.constexpr,
    NUM_PRODUCT_WEIGHTS: tl.constexpr,
    EPS: tl.constexpr,
):
    """Compute gradients w.r.t. inputs and weights of the fused operation."""
    batch_block_id = tl.program_id(axis=0)
    thread_block_id = tl.program_id(axis=1)

    batch_ids = batch_block_id*BATCH_BLOCK + tl.arange(0, BATCH_BLOCK)
    feature_ids = thread_block_id*FEATURE_BLOCK + tl.arange(0, FEATURE_BLOCK)

    batch_mask = batch_ids < batch_size
    feature_mask = feature_ids < n_features
    batch_feature_mask = batch_mask[:, None] & feature_mask[None, :]

    stride_component = batch_size * n_features
    base_offset = batch_ids[:, None] * n_features + feature_ids[None, :]
    
    weight_offset = feature_ids * NUM_PRODUCT_WEIGHTS
    block_offset = batch_block_id * n_features * NUM_PRODUCT_WEIGHTS

    w0 = tl.load(weights_ptr + weight_offset + 0, mask=feature_mask)
    w1 = tl.load(weights_ptr + weight_offset + 1, mask=feature_mask)
    w2 = tl.load(weights_ptr + weight_offset + 2, mask=feature_mask)
    w3 = tl.load(weights_ptr + weight_offset + 3, mask=feature_mask)
    w4 = tl.load(weights_ptr + weight_offset + 4, mask=feature_mask)
    w5 = tl.load(weights_ptr + weight_offset + 5, mask=feature_mask)
    w6 = tl.load(weights_ptr + weight_offset + 6, mask=feature_mask)
    w7 = tl.load(weights_ptr + weight_offset + 7, mask=feature_mask)
    w8 = tl.load(weights_ptr + weight_offset + 8, mask=feature_mask)
    w9 = tl.load(weights_ptr + weight_offset + 9, mask=feature_mask)

    x0_raw = tl.load(x_ptr + 0 * stride_component + base_offset, mask=batch_feature_mask)
    x1_raw = tl.load(x_ptr + 1 * stride_component + base_offset, mask=batch_feature_mask)
    x2_raw = tl.load(x_ptr + 2 * stride_component + base_offset, mask=batch_feature_mask)
    x3_raw = tl.load(x_ptr + 3 * stride_component + base_offset, mask=batch_feature_mask)

    y0_raw = tl.load(y_ptr + 0 * stride_component + base_offset, mask=batch_feature_mask)
    y1_raw = tl.load(y_ptr + 1 * stride_component + base_offset, mask=batch_feature_mask)
    y2_raw = tl.load(y_ptr + 2 * stride_component + base_offset, mask=batch_feature_mask)
    y3_raw = tl.load(y_ptr + 3 * stride_component + base_offset, mask=batch_feature_mask)

    go0 = tl.load(grad_output_ptr + 0 * stride_component + base_offset, mask=batch_feature_mask)
    go1 = tl.load(grad_output_ptr + 1 * stride_component + base_offset, mask=batch_feature_mask)
    go2 = tl.load(grad_output_ptr + 2 * stride_component + base_offset, mask=batch_feature_mask)
    go3 = tl.load(grad_output_ptr + 3 * stride_component + base_offset, mask=batch_feature_mask)

    if NORMALIZE:
        o0 = tl.load(output_ptr + 0 * stride_component + base_offset, mask=batch_feature_mask)
        o1 = tl.load(output_ptr + 1 * stride_component + base_offset, mask=batch_feature_mask)
        o2 = tl.load(output_ptr + 2 * stride_component + base_offset, mask=batch_feature_mask)
        o3 = tl.load(output_ptr + 3 * stride_component + base_offset, mask=batch_feature_mask)

        pn_scalar = tl.load(pnorm_ptr + 0*batch_size + batch_ids, mask=batch_mask)[:, None]
        pn_vector = tl.load(pnorm_ptr + 1*batch_size + batch_ids, mask=batch_mask)[:, None]
        pn_pseudo = tl.load(pnorm_ptr + 3*batch_size + batch_ids, mask=batch_mask)[:, None]

        dot_scalar = tl.load(dot_ptr + 0*batch_size + batch_ids, mask=batch_mask)[:, None]
        dot_vector = tl.load(dot_ptr + 1*batch_size + batch_ids, mask=batch_mask)[:, None]
        dot_pseudo = tl.load(dot_ptr + 2*batch_size + batch_ids, mask=batch_mask)[:, None]

        rms_scalar = tl.sqrt(pn_scalar + EPS)
        rms_vector = tl.sqrt(pn_vector + EPS)
        rms_pseudo = tl.sqrt(pn_pseudo + EPS)

        go0 = go0/rms_scalar - o0 * dot_scalar / (n_features*rms_scalar*rms_scalar)
        go1 = go1/rms_vector - o1 * dot_vector / (n_features*rms_vector*rms_vector)
        go2 = go2/rms_vector - o2 * dot_vector / (n_features*rms_vector*rms_vector)
        go3 = go3/rms_pseudo - o3 * dot_pseudo / (n_features*rms_pseudo*rms_pseudo)

    # weighted geometric product backward
    gate_x = compute_gelu_gate(x0_raw)
    gate_y = compute_gelu_gate(y0_raw)

    xvals = gate_kernel((x0_raw,x1_raw,x2_raw,x3_raw), (gate_x,))  # X * GATE_X
    yvals = gate_kernel((y0_raw,y1_raw,y2_raw,y3_raw), (gate_y,))  # Y * GATE_Y
    wvals = (w0, w1, w2, w3, w4, w5, w6, w7, w8, w9)
    grads, = weighted_gp_grad_kernel(xvals, yvals, (wvals,), (go0,go1,go2,go3)) # Returns a scalar, which we unpack immidiatelly.

    x_grad_0, x_grad_1, x_grad_2, x_grad_3 = grads[0:4]
    y_grad_0, y_grad_1, y_grad_2, y_grad_3 = grads[4:8]
    _w_grad_0, _w_grad_1, _w_grad_2, _w_grad_3, _w_grad_4, _w_grad_5, _w_grad_6, _w_grad_7, _w_grad_8, _w_grad_9 = grads[8:]

    w_grad_0 = tl.sum(_w_grad_0, axis=0)
    w_grad_1 = tl.sum(_w_grad_1, axis=0)
    w_grad_2 = tl.sum(_w_grad_2, axis=0)
    w_grad_3 = tl.sum(_w_grad_3, axis=0)
    w_grad_4 = tl.sum(_w_grad_4, axis=0)
    w_grad_5 = tl.sum(_w_grad_5, axis=0)
    w_grad_6 = tl.sum(_w_grad_6, axis=0)
    w_grad_7 = tl.sum(_w_grad_7, axis=0)
    w_grad_8 = tl.sum(_w_grad_8, axis=0)
    w_grad_9 = tl.sum(_w_grad_9, axis=0)

    # GELU gate gradients
    dgate_x = compute_gelu_gate_grad(x0_raw)
    dgate_y = compute_gelu_gate_grad(y0_raw)

    x_grad_0 = (gate_x + x0_raw*dgate_x) * x_grad_0 + dgate_x * (x1_raw*x_grad_1 + x2_raw*x_grad_2 + x3_raw*x_grad_3)
    x_grad_1 = gate_x * x_grad_1
    x_grad_2 = gate_x * x_grad_2
    x_grad_3 = gate_x * x_grad_3

    y_grad_0 = (gate_y + y0_raw*dgate_y) * y_grad_0 + dgate_y * (y1_raw*y_grad_1 + y2_raw*y_grad_2 + y3_raw*y_grad_3)
    y_grad_1 = gate_y * y_grad_1
    y_grad_2 = gate_y * y_grad_2
    y_grad_3 = gate_y * y_grad_3

    tl.store(grad_x_ptr + 0 * stride_component + base_offset, x_grad_0, mask=batch_feature_mask)
    tl.store(grad_x_ptr + 1 * stride_component + base_offset, x_grad_1, mask=batch_feature_mask)
    tl.store(grad_x_ptr + 2 * stride_component + base_offset, x_grad_2, mask=batch_feature_mask)
    tl.store(grad_x_ptr + 3 * stride_component + base_offset, x_grad_3, mask=batch_feature_mask)

    tl.store(grad_y_ptr + 0 * stride_component + base_offset, y_grad_0, mask=batch_feature_mask)
    tl.store(grad_y_ptr + 1 * stride_component + base_offset, y_grad_1, mask=batch_feature_mask)
    tl.store(grad_y_ptr + 2 * stride_component + base_offset, y_grad_2, mask=batch_feature_mask)
    tl.store(grad_y_ptr + 3 * stride_component + base_offset, y_grad_3, mask=batch_feature_mask)

    tl.store(grad_weight_ptr + block_offset + weight_offset + 0, w_grad_0, mask=feature_mask)
    tl.store(grad_weight_ptr + block_offset + weight_offset + 1, w_grad_1, mask=feature_mask)
    tl.store(grad_weight_ptr + block_offset + weight_offset + 2, w_grad_2, mask=feature_mask)
    tl.store(grad_weight_ptr + block_offset + weight_offset + 3, w_grad_3, mask=feature_mask)
    tl.store(grad_weight_ptr + block_offset + weight_offset + 4, w_grad_4, mask=feature_mask)
    tl.store(grad_weight_ptr + block_offset + weight_offset + 5, w_grad_5, mask=feature_mask)
    tl.store(grad_weight_ptr + block_offset + weight_offset + 6, w_grad_6, mask=feature_mask)
    tl.store(grad_weight_ptr + block_offset + weight_offset + 7, w_grad_7, mask=feature_mask)
    tl.store(grad_weight_ptr + block_offset + weight_offset + 8, w_grad_8, mask=feature_mask)
    tl.store(grad_weight_ptr + block_offset + weight_offset + 9, w_grad_9, mask=feature_mask)


def gelu_geometric_product_norm_bwd(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    o: torch.Tensor,
    partial_norm: torch.Tensor,
    grad_output: torch.Tensor,
    normalize: bool,
) -> torch.Tensor:
    """Backward pass for the fused operation."""
    _, B, N = x.shape

    BATCH_BLOCK = min(DEFAULT_BATCH_BLOCK, B)
    FEATURE_BLOCK = min(DEFAULT_FEATURE_BLOCK, N)

    num_blocks_batch = triton.cdiv(B, BATCH_BLOCK)
    num_blocks_features = triton.cdiv(N, FEATURE_BLOCK)

    grad_x = torch.zeros_like(x)
    grad_y = torch.zeros_like(y)
    dot = (torch.zeros((NUM_GRADES, B), device=x.device, dtype=x.dtype) if normalize else torch.empty(0))
    grad_weight = torch.zeros((num_blocks_batch, N, NUM_PRODUCT_WEIGHTS), device=x.device, dtype=weight.dtype)

    grid = (num_blocks_batch, num_blocks_features)

    if normalize:
        grad_o_dot_o_kernel[grid](
            dot,
            partial_norm,
            o,
            grad_output,
            B,
            N,
            BATCH_BLOCK,
            FEATURE_BLOCK,
            EPS,
            num_warps=DEFAULT_NUM_WARPS,
            num_stages=DEFAULT_NUM_STAGES,
        )

    gelu_wgp_norm_kernel_bwd[grid](
        x,
        y,
        o,
        weight,
        dot,
        partial_norm,
        grad_output,
        grad_x,
        grad_y,
        grad_weight,
        normalize,
        B,
        N,
        BATCH_BLOCK,
        FEATURE_BLOCK,
        NUM_PRODUCT_WEIGHTS,
        EPS,
        num_warps=DEFAULT_NUM_WARPS,
        num_stages=DEFAULT_NUM_STAGES,
    )

    grad_weight = torch.sum(grad_weight, dim=0)

    return grad_x, grad_y, grad_weight


class WeightedGeluGeometricProductNorm2D(torch.autograd.Function):

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, y, weight, normalize):
        assert x.is_contiguous() and y.is_contiguous() and weight.is_contiguous()

        ctx.dtype = x.dtype
        ctx.normalize = normalize

        o, partial_norm = gelu_geometric_product_norm_fwd(
            x,
            y,
            weight,
            normalize,
        )

        ctx.save_for_backward(x, y, weight, o, partial_norm)

        return o.to(x.dtype)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()

        x, y, weight, o, partial_norm = ctx.saved_tensors

        grad_x, grad_y, grad_weight = gelu_geometric_product_norm_bwd(
            x,
            y,
            weight,
            o,
            partial_norm,
            grad_output,
            ctx.normalize,
        )

        return grad_x, grad_y, grad_weight, None, None, None, None
    

def fused_gelu_sgp_norm_2d(x, y, weight, normalize=True):
    """
    Fused operation that applies GELU non-linearity to two multivector inputs,
    then computes their weighted geometric product, and applies RMSNorm.
    
    Clifford algebra is assumed to be Cl(2,0).

    Args:
        x (torch.Tensor): Input tensor of shape (MV_DIM, B, N).
        y (torch.Tensor): Input tensor of shape (MV_DIM, B, N).
        weight (torch.Tensor): Weight tensor of shape (N, NUM_PRODUCT_WEIGHTS), one weight per geometric product component.
        normalize (bool): Whether to apply RMSNorm after the geometric product.

    Returns:
        torch.Tensor: Output tensor of shape (MV_DIM, B, N) after applying the fused operation.
    """
    return WeightedGeluGeometricProductNorm2D.apply(x, y, weight, normalize)