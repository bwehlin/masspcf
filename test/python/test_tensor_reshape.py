import masspcf as mpcf


def test_tensor2d_flatten():
    X = mpcf.zeros((2, 3), dtype=mpcf.f32)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i, j] = 10 * i + j

    Y = X.flatten()

    assert Y.shape == mpcf.Shape((6,))

    assert Y[0] == X[0, 0]
    assert Y[1] == X[0, 1]
    assert Y[2] == X[0, 2]
    assert Y[3] == X[1, 0]
    assert Y[4] == X[1, 1]
    assert Y[5] == X[1, 2]
