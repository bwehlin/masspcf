import pytest
import masspcf as mpcf

def test_zeros():
    Z = mpcf.zeros((1, 2, 3))

    assert len(Z.shape) == 3
    assert Z.shape[0] == 1
    assert Z.shape[1] == 2
    assert Z.shape[2] == 3

@pytest.mark.parametrize("gpu", [True, False])
def test_pdist_zeros(gpu):
    Z = mpcf.zeros((2,))
    D = mpcf.pdist(Z)

    assert D.shape == (2, 2)
    assert D[0][0] == 0.
    assert D[0][1] == 0.
    assert D[1][0] == 0.
    assert D[1][1] == 0.
