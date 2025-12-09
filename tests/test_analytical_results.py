"""
Test if the kingdon generated code is correct by comparing with the hand-optimized results.
"""
from ops.vga2d import weighted_gp, weighted_gp_grad, X, Y, weights, ws, go

def test_weighted_gp():
    wgp_output = weighted_gp(X, Y, weights)
    w0, w1, w2, w3, w4, w5, w6, w7, w8, w9 = ws
    x0,x1,x2,x3 = X.values()
    y0,y1,y2,y3 = Y.values()
    o0 = w0*x0*y0 + w3*(x1*y1 + x2*y2) - w7*x3*y3
    o1 = w1*x0*y1 + w4*x1*y0 - w5*x2*y3 + w8*x3*y2
    o2 = w1*x0*y2 + w5*x1*y3 + w4*x2*y0 - w8*x3*y1
    o3 = w2*x0*y3 + w6*(x1*y2 - x2*y1) + w9*x3*y0
    assert wgp_output.values() == [o0.expand(), o1.expand(), o2.expand(), o3.expand()]

def test_weighted_gp_grad():
    grad_wgp = weighted_gp_grad(X, Y, weights, go)
    w0,w1,w2,w3,w4,w5,w6,w7,w8,w9 = ws
    x0,x1,x2,x3 = X.values()
    y0,y1,y2,y3 = Y.values()
    go0,go1,go2,go3 = go.values()

    (grad_x0, grad_x1, grad_x2, grad_x3, 
     grad_y0, grad_y1, grad_y2, grad_y3, 
     grad_w0, grad_w1, grad_w2, grad_w3, grad_w4, grad_w5, grad_w6, grad_w7, grad_w8, grad_w9) = grad_wgp.e

    assert not grad_w0 - (go0*x0*y0)
    assert not grad_w1 - (go1*x0*y1 + go2*x0*y2)
    assert not grad_w2 - (go3*x0*y3)
    assert not grad_w3 - (go0*(x1*y1 + x2*y2)).expand()
    assert not grad_w4 - (go2*x2*y0 + go1*x1*y0)
    assert not grad_w5 - (go2*x1*y3 - go1*x2*y3)
    assert not grad_w6 - (go3*(x1*y2 - x2*y1)).expand()
    assert not grad_w7 - (-go0*x3*y3)
    assert not grad_w8 - (go1*x3*y2 - go2*x3*y1)
    assert not grad_w9 - (go3*x3*y0)

    assert not grad_x0 - (go3*w2*y3 + go0*w0*y0 + go1*w1*y1 + go2*w1*y2)
    assert not (grad_x1 - (go1*w4*y0 + go2*w5*y3 + go0*w3*y1 + go3*w6*y2))
    assert not (grad_x2 - (-go1*w5*y3 + go2*w4*y0 + go0*w3*y2 - go3*w6*y1))
    assert not (grad_x3 - (go1*w8*y2 + go3*w9*y0 - go0*w7*y3 - go2*w8*y1))

    assert not (grad_y0 - (go3*w9*x3 + go0*w0*x0 + go2*x2*w4 + go1*x1*w4))
    assert not (grad_y1 - (go1*w1*x0 + go0*w3*x1 - go3*w6*x2 - go2*w8*x3))
    assert not (grad_y2 - (go1*w8*x3 + go2*w1*x0 + go0*w3*x2 + go3*w6*x1))
    assert not (grad_y3 - (go2*w5*x1 + go3*w2*x0 - go1*x2*w5 - go0*w7*x3))