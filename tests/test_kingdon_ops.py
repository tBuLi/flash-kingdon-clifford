from ops import number_of_wgp_terms

def test_number_of_wgp_terms():
    from ops.vga2d import X, Y
    assert number_of_wgp_terms(X, Y) == 10

    from ops.vga3d import X, Y
    assert number_of_wgp_terms(X, Y) == 20