"""
Test if the kingdon generated code is correct by comparing with the hand-optimized results.
"""
from ops.vga3d import weighted_gp, weighted_gp_grad, X, Y, weights, go

def test_weighted_gp_3d():
    wgp_output = weighted_gp(X, Y, weights)
    w0,w1,w2,w3,w5,w4,w7,w6,w9,w8,w13,w11,w15,w10,w14,w12,w19,w18,w17,w16 = weights.e
    x0,x1,x2,x3,x4,x5,x6,x7 = X.values()
    y0,y1,y2,y3,y4,y5,y6,y7 = Y.values()
    o0 = (w0*x0*y0 + w4 * (x1*y1 + x2*y2 + x3*y3) - w10 * (x4*y4 + x5*y5 + x6*y6) - w16*x7*y7)
    o1 = (w1*x0*y1 + w5*x1*y0 - w6 * (x2*y4 + x3*y5) + w11 * (x4*y2 + x5*y3) - w12*x6*y7 - w17*x7*y6)
    o2 = (w1*x0*y2 + w6*x1*y4 + w5*x2*y0 - w6*x3*y6 - w11*x4*y1 + w12*x5*y7 + w11*x6*y3 + w17*x7*y5)
    o3 = (w1*x0*y3 + w6 * (x1*y5 + x2*y6) + w5*x3*y0 - w12*x4*y7 - w11 * (x5*y1 + x6*y2) - w17*x7*y4)
    o4 = (w2*x0*y4 + w7*x1*y2 - w7*x2*y1 + w8*x3*y7 + w13*x4*y0 - w14*x5*y6 + w14*x6*y5 + w18*x7*y3)
    o5 = (w2*x0*y5 + w7*x1*y3 - w8*x2*y7 - w7*x3*y1 + w14*x4*y6 + w13*x5*y0 - w14*x6*y4 - w18*x7*y2)
    o6 = (w2*x0*y6 + w8*x1*y7 + w7*x2*y3 - w7*x3*y2 - w14*x4*y5 + w14*x5*y4 + w13*x6*y0 + w18*x7*y1)
    o7 = (w3*x0*y7 + w9*x1*y6 - w9*x2*y5 + w9*x3*y4 + w15*x4*y3 - w15*x5*y2 + w15*x6*y1 + w19*x7*y0)
    assert wgp_output.values() == [o.expand() for o in [o0, o1, o2, o3, o4, o5, o6, o7]]

def test_weighted_gp_grad_3d():
    grad_wgp = weighted_gp_grad(X, Y, weights, go)
    w0,w1,w2,w3,w5,w4,w7,w6,w9,w8,w13,w11,w15,w10,w14,w12,w19,w18,w17,w16 = weights.e
    x0,x1,x2,x3,x4,x5,x6,x7 = X.values()
    y0,y1,y2,y3,y4,y5,y6,y7 = Y.values()
    go0,go1,go2,go3,go4,go5,go6,go7 = go.values()

    (grad_x0, grad_x1, grad_x2, grad_x3, 
     grad_x4, grad_x5, grad_x6, grad_x7, 
     grad_y0, grad_y1, grad_y2, grad_y3, 
     grad_y4, grad_y5, grad_y6, grad_y7, 
     grad_w0, grad_w1, grad_w2, grad_w3, grad_w5, grad_w4, grad_w7, grad_w6, grad_w9, grad_w8,
     grad_w13, grad_w11, grad_w15, grad_w10, grad_w14, grad_w12, grad_w19, grad_w18, grad_w17, grad_w16) = grad_wgp.e

    tmp0 = go0 * w0
    tmp1 = go7 * w3
    tmp2 = go1 * y1
    tmp3 = go2 * y2
    tmp4 = go3 * y3
    tmp5 = go4 * y4
    tmp6 = go5 * y5
    tmp7 = go6 * y6
    tmp8 = go0 * w4
    tmp9 = w5 * y0
    tmp10 = w8 * y7
    tmp11 = go7 * w9
    tmp12 = go1 * w6
    tmp13 = w13 * y0
    tmp14 = go7 * w15
    tmp15 = go0 * w10
    tmp16 = w12 * y7
    tmp17 = go3 * w11
    tmp18 = go7 * w19
    tmp19 = go0 * w16
    tmp20 = go4 * y3
    tmp21 = go6 * y1
    tmp22 = go5 * y2
    tmp23 = go1 * y6
    tmp24 = go3 * y4
    tmp25 = go4 * x4
    tmp26 = go5 * x5
    tmp27 = go6 * x6
    tmp28 = go1 * x1
    tmp29 = go2 * x2
    tmp30 = go3 * x3
    tmp31 = w1 * x0
    tmp32 = w18 * x7
    tmp33 = w2 * x0
    tmp34 = w17 * x7
    tmp35 = go4 * x3
    tmp36 = go6 * x1
    tmp37 = go5 * x2
    tmp38 = go1 * x6
    tmp39 = go3 * x4

    x_grad_0 = (tmp0*y0 + tmp1*y7 + w1 * (tmp2+tmp3+tmp4) + w2 * (tmp5+tmp6+tmp7))
    x_grad_1 = (go1*tmp9 + go6*tmp10 + tmp11*y6 + tmp8*y1 + w6 * (go2*y4 + go3*y5) + w7 * (go4*y2 + go5*y3))
    x_grad_2 = (go2*tmp9 + go3*w6*y6 - go5*tmp10 - tmp11*y5 - tmp12*y4 + tmp8*y2 + w7 * (-go4 * y1 + go6*y3))
    x_grad_3 = (go3*tmp9 + go4*tmp10 + tmp11*y4 + tmp8*y3 - w6 * (go1*y5 + go2*y6) - w7 * (go5*y1 + go6*y2))
    x_grad_4 = (-go3 * tmp16 + go4*tmp13 + tmp14*y3 - tmp15*y4 + w11 * (go1*y2 - go2*y1) + w14 * (go5*y6 - go6*y5))
    x_grad_5 = (go1*w11*y3 + go2*w12*y7 - go4*w14*y6 + go5*w13*y0 + go6*w14*y4 - tmp14*y2 - tmp15*y5 - tmp17*y1)
    x_grad_6 = (-go1 * tmp16 + go6*tmp13 + tmp14*y1 - tmp15*y6 + w11 * (go2*y3 - go3*y2) + w14 * (go4*y5 - go5*y4))
    x_grad_7 = (tmp18*y0 - tmp19*y7 + w17 * (go2*y5 - tmp23 - tmp24) + w18 * (tmp20+tmp21-tmp22))

    y_grad_0 = (tmp0*x0 + tmp18*x7 + w13 * (tmp25+tmp26+tmp27) + w5 * (tmp28+tmp29+tmp30))
    y_grad_1 = (go1*tmp31 + go6*tmp32 + tmp14*x6 + tmp8*x1 - w11 * (go2*x4 + go3*x5) - w7 * (go4*x2 + go5*x3))
    y_grad_2 = (go1*w11*x4 + go2*tmp31 + go4*w7*x1 - go5*tmp32 - go6*w7*x3 - tmp14*x5 - tmp17*x6 + tmp8*x2)
    y_grad_3 = (go3*tmp31 + go4*tmp32 + tmp14*x4 + tmp8*x3 + w11 * (go1*x5 + go2*x6) + w7 * (go5*x1 + go6*x2))
    y_grad_4 = (-go3 * tmp34 + go4*tmp33 + tmp11*x3 - tmp15*x4 + w14 * (-go5 * x6 + go6*x5) + w6 * (-go1 * x2 + go2*x1))
    y_grad_5 = (go2*w17*x7 + go3*w6*x1 + go4*w14*x6 + go5*w2*x0 - go6*w14*x4 - tmp11*x2 - tmp12*x3 - tmp15*x5)
    y_grad_6 = (-go1 * tmp34 + go6*tmp33 + tmp11*x1 - tmp15*x6 + w14 * (-go4 * x5 + go5*x4) + w6 * (-go2 * x3 + go3*x2))
    y_grad_7 = (tmp1*x0 - tmp19*x7 + w12 * (go2*x5 - tmp38 - tmp39) + w8 * (tmp35+tmp36-tmp37))

    w_grad_0 = go0 * x0 * y0
    w_grad_1 = tmp2*x0 + tmp3*x0 + tmp4*x0
    w_grad_2 = tmp5*x0 + tmp6*x0 + tmp7*x0
    w_grad_3 = go7 * x0 * y7
    w_grad_4 = go0 * (x1*y1 + x2*y2 + x3*y3)
    w_grad_5 = tmp28*y0 + tmp29*y0 + tmp30*y0
    w_grad_6 = go1 * (-x2 * y4 - x3*y5) + go2 * (x1*y4 - x3*y6) + go3 * (x1*y5 + x2*y6)
    w_grad_7 = go4 * (x1*y2 - x2*y1) + go5 * (x1*y3 - x3*y1) + go6 * (x2*y3 - x3*y2)
    w_grad_8 = tmp35*y7 + tmp36*y7 - tmp37*y7
    w_grad_9 = go7 * (x1*y6 - x2*y5 + x3*y4)
    w_grad_10 = go0 * (-x4 * y4 - x5*y5 - x6*y6)
    w_grad_11 = go1 * (x4*y2 + x5*y3) + go2 * (-x4 * y1 + x6*y3) + go3 * (-x5 * y1 - x6*y2)
    w_grad_12 = go2*x5*y7 - tmp38*y7 - tmp39*y7
    w_grad_13 = tmp25*y0 + tmp26*y0 + tmp27*y0
    w_grad_14 = go4 * (-x5 * y6 + x6*y5) + go5 * (x4*y6 - x6*y4) + go6 * (-x4 * y5 + x5*y4)
    w_grad_15 = go7 * (x4*y3 - x5*y2 + x6*y1)
    w_grad_16 = -go0 * x7 * y7
    w_grad_17 = go2*x7*y5 - tmp23*x7 - tmp24*x7
    w_grad_18 = tmp20*x7 + tmp21*x7 - tmp22*x7
    w_grad_19 = go7 * x7 * y0
    assert not grad_w0 - w_grad_0.expand()
    assert not grad_w1 - w_grad_1.expand()
    assert not grad_w2 - w_grad_2.expand()
    assert not grad_w3 - w_grad_3.expand()
    assert not grad_w4 - w_grad_4.expand()
    assert not grad_w5 - w_grad_5.expand()
    assert not grad_w6 - w_grad_6.expand()
    assert not grad_w7 - w_grad_7.expand()
    assert not grad_w8 - w_grad_8.expand()
    assert not grad_w9 - w_grad_9.expand()
    assert not grad_w10 - w_grad_10.expand()
    assert not grad_w11 - w_grad_11.expand()
    assert not grad_w12 - w_grad_12.expand()
    assert not grad_w13 - w_grad_13.expand()
    assert not grad_w14 - w_grad_14.expand()
    assert not grad_w15 - w_grad_15.expand()
    assert not grad_w16 - w_grad_16.expand()
    assert not grad_w17 - w_grad_17.expand()
    assert not grad_w18 - w_grad_18.expand()
    assert not grad_w19 - w_grad_19.expand()

    assert not grad_x0 - x_grad_0.expand()
    assert not grad_x1 - x_grad_1.expand()
    assert not grad_x2 - x_grad_2.expand()
    assert not grad_x3 - x_grad_3.expand()
    assert not grad_x4 - x_grad_4.expand()
    assert not grad_x5 - x_grad_5.expand()
    assert not grad_x6 - x_grad_6.expand()
    assert not grad_x7 - x_grad_7.expand()

    assert not grad_y0 - y_grad_0.expand()
    assert not grad_y1 - y_grad_1.expand()
    assert not grad_y2 - y_grad_2.expand()
    assert not grad_y3 - y_grad_3.expand()
    assert not grad_y4 - y_grad_4.expand()
    assert not grad_y5 - y_grad_5.expand()
    assert not grad_y6 - y_grad_6.expand()
    assert not grad_y7 - y_grad_7.expand()
