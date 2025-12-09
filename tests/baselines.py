import torch


### Activations ###

def mv_gelu(x):
    """Apply GELU activation gated by scalar component."""
    scalar = x[[0]]
    gate = 0.5 * (1 + torch.erf(scalar * 0.7071067811865475))
    return x * gate


### Norms ###

def mv_rmsnorm_2d(x, eps=1e-6):
    """RMS normalization for Cl(2,0) (scalar, vector, pseudoscalar)."""
    scalar = x[[0]]
    vector = x[[1, 2]]
    pseudoscalar = x[[3]]

    scalar_rms = (scalar.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
    scalar = scalar / scalar_rms

    vector_norm = vector.norm(dim=0, keepdim=True)
    vector_rms = (vector_norm.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
    vector = vector / vector_rms

    pseudoscalar_rms = (pseudoscalar.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
    pseudoscalar = pseudoscalar / pseudoscalar_rms

    return torch.cat([scalar, vector, pseudoscalar], dim=0)


def mv_rmsnorm_3d(x, eps=1e-6):
    """RMS normalization for Cl(3,0) (scalar, vector, bivector, pseudoscalar)."""
    scalar = x[[0]]
    vector = x[[1, 2, 3]]
    bivector = x[[4, 5, 6]]
    pseudoscalar = x[[7]]

    scalar_rms = (scalar.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
    scalar = scalar / scalar_rms

    vector_norm = vector.norm(dim=0, keepdim=True)
    vector_rms = (vector_norm.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
    vector = vector / vector_rms

    bivector_norm = bivector.norm(dim=0, keepdim=True)
    bivector_rms = (bivector_norm.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
    bivector = bivector / bivector_rms

    pseudoscalar_rms = (pseudoscalar.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
    pseudoscalar = pseudoscalar / pseudoscalar_rms

    return torch.cat([scalar, vector, bivector, pseudoscalar], dim=0)


### Geometric Product Layers ###

def sgp_2d(x, y, weight):
    """Weighted geometric product in Cl(2,0)."""
    x0, x1, x2, x3 = x[0], x[1], x[2], x[3]
    y0, y1, y2, y3 = y[0], y[1], y[2], y[3]
        
    w0, w1, w2, w3, w4, w5, w6, w7, w8, w9 = weight.T
    o0 = w0 * x0 * y0 + w3 * (x1 * y1 + x2 * y2) - w7 * x3 * y3
    o1 = w1 * x0 * y1 + w4 * x1 * y0 - w5 * x2 * y3 + w8 * x3 * y2
    o2 = w1 * x0 * y2 + w5 * x1 * y3 + w4 * x2 * y0 - w8 * x3 * y1
    o3 = w2 * x0 * y3 + w6 * (x1 * y2 - x2 * y1) + w9 * x3 * y0
    
    return torch.stack([o0, o1, o2, o3], dim=0)


def sgp_3d(x, y, weight):
    """Weighted geometric product in Cl(3,0)."""
    x0, x1, x2, x3, x4, x5, x6, x7 = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]
    y0, y1, y2, y3, y4, y5, y6, y7 = y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]
    
    w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18, w19 = weight.T
    
    o0 = (w0*x0*y0 + w4 * (x1*y1 + x2*y2 + x3*y3) - w10 * (x4*y4 + x5*y5 + x6*y6) - w16*x7*y7)
    o1 = (w1*x0*y1 + w5*x1*y0 - w6 * (x2*y4 + x3*y5) + w11 * (x4*y2 + x5*y3) - w12*x6*y7 - w17*x7*y6)
    o2 = (w1*x0*y2 + w6*x1*y4 + w5*x2*y0 - w6*x3*y6 - w11*x4*y1 + w12*x5*y7 + w11*x6*y3 + w17*x7*y5)
    o3 = (w1*x0*y3 + w6 * (x1*y5 + x2*y6) + w5*x3*y0 - w12*x4*y7 - w11 * (x5*y1 + x6*y2) - w17*x7*y4)
    o4 = (w2*x0*y4 + w7*x1*y2 - w7*x2*y1 + w8*x3*y7 + w13*x4*y0 - w14*x5*y6 + w14*x6*y5 + w18*x7*y3)
    o5 = (w2*x0*y5 + w7*x1*y3 - w8*x2*y7 - w7*x3*y1 + w14*x4*y6 + w13*x5*y0 - w14*x6*y4 - w18*x7*y2)
    o6 = (w2*x0*y6 + w8*x1*y7 + w7*x2*y3 - w7*x3*y2 - w14*x4*y5 + w14*x5*y4 + w13*x6*y0 + w18*x7*y1)
    o7 = (w3*x0*y7 + w9*x1*y6 - w9*x2*y5 + w9*x3*y4 + w15*x4*y3 - w15*x5*y2 + w15*x6*y1 + w19*x7*y0)
    
    return torch.stack([o0, o1, o2, o3, o4, o5, o6, o7], dim=0)


def fcgp_2d(x, y, weight):
    """Fully connected geometric product in Cl(2,0)."""
    x0, x1, x2, x3 = x[0], x[1], x[2], x[3]
    y0, y1, y2, y3 = y[0], y[1], y[2], y[3]
        
    w0, w1, w2, w3, w4, w5, w6, w7, w8, w9 = weight
    
    o0 = (x0 * y0) @ w0 + (x1 * y1 + x2 * y2) @ w3 - (x3 * y3) @ w7
    o1 = (x0 * y1) @ w1 + (x1 * y0) @ w4 - (x2 * y3) @ w5 + (x3 * y2) @ w8
    o2 = (x0 * y2) @ w1 + (x1 * y3) @ w5 + (x2 * y0) @ w4 - (x3 * y1) @ w8
    o3 = x0 * y3 @ w2 + (x1 * y2 - x2 * y1) @ w6 + (x3 * y0) @ w9
    
    return torch.stack([o0, o1, o2, o3], dim=0)


def fcgp_3d(x, y, weight):
    """Fully connected geometric product in Cl(3,0)."""
    x0, x1, x2, x3, x4, x5, x6, x7 = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]
    y0, y1, y2, y3, y4, y5, y6, y7 = y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]
    
    w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18, w19 = weight
    
    o0 = (x0 * y0) @ w0 + (x1 * y1 + x2 * y2 + x3 * y3) @ w4 - (x4 * y4 + x5 * y5 + x6 * y6) @ w10 - (x7 * y7) @ w16
    o1 = (x0 * y1) @ w1 + (x1 * y0) @ w5 - (x2 * y4 + x3 * y5) @ w6 + (x4 * y2 + x5 * y3) @ w11 - (x6 * y7) @ w12 - (x7 * y6) @ w17
    o2 = (x0 * y2) @ w1 + (x1 * y4) @ w6 + (x2 * y0) @ w5 - (x3 * y6) @ w6 - (x4 * y1) @ w11 + (x5 * y7) @ w12 + (x6 * y3) @ w11 + (x7 * y5) @ w17
    o3 = (x0 * y3) @ w1 + (x1 * y5 + x2 * y6) @ w6 + (x3 * y0) @ w5 - (x4 * y7) @ w12 - (x5 * y1 + x6 * y2) @ w11 - (x7 * y4) @ w17
    o4 = (x0 * y4) @ w2 + (x1 * y2 - x2 * y1) @ w7 + (x3 * y7) @ w8 + (x4 * y0) @ w13 - (x5 * y6) @ w14 + (x6 * y5) @ w14 + (x7 * y3) @ w18
    o5 = (x0 * y5) @ w2 + (x1 * y3) @ w7 - (x2 * y7) @ w8 - (x3 * y1) @ w7 + (x4 * y6) @ w14 + (x5 * y0) @ w13 - (x6 * y4) @ w14 - (x7 * y2) @ w18
    o6 = (x0 * y6) @ w2 + (x1 * y7) @ w8 + (x2 * y3) @ w7 - (x3 * y2) @ w7 - (x4 * y5) @ w14 + (x5 * y4) @ w14 + (x6 * y0) @ w13 + (x7 * y1) @ w18
    o7 = (x0 * y7) @ w3 + (x1 * y6 - x2 * y5 + x3 * y4) @ w9 + (x4 * y3 - x5 * y2 + x6 * y1) @ w15 + (x7 * y0) @ w19
    
    return torch.stack([o0, o1, o2, o3, o4, o5, o6, o7], dim=0)


### Baseline implementations ###

@torch.compile
def gelu_sgp_norm_2d_torch(x, y, weight, normalize=True):
    """Geometric product layer with GELU activation and RMS normalization in Cl(2,0)."""
    x = mv_gelu(x)
    y = mv_gelu(y)
    o = sgp_2d(x, y, weight)
    if normalize:
        o = mv_rmsnorm_2d(o)
    return o


@torch.compile
def gelu_sgp_norm_3d_torch(x, y, weight, normalize=True):
    """Geometric product layer with GELU activation and RMS normalization in Cl(3,0)."""
    x = mv_gelu(x)
    y = mv_gelu(y)
    o = sgp_3d(x, y, weight)
    if normalize:
        o = mv_rmsnorm_3d(o)
    return o


@torch.compile
def gelu_fcgp_norm_2d_torch(x, y, weight, normalize=True):
    """Fully connected geometric product layer with GELU activation and RMS normalization in Cl(2,0)."""
    x = mv_gelu(x)
    y = mv_gelu(y)
    o = fcgp_2d(x, y, weight)
    if normalize:
        o = mv_rmsnorm_2d(o)
    return o


@torch.compile
def gelu_fcgp_norm_3d_torch(x, y, weight, normalize=True):
    """Fully connected geometric product layer with GELU activation and RMS normalization in Cl(3,0)."""
    x = mv_gelu(x)
    y = mv_gelu(y)
    o = fcgp_3d(x, y, weight)
    if normalize:
        o = mv_rmsnorm_3d(o)
    return o


