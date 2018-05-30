import numpy as np

def sample_orthogonal_matrix(real_dim, active_dim):
    """
    :return: An orthogonal matrix
    """
    A = np.zeros((real_dim, active_dim), dtype=np.float64)
    for i in range(real_dim):
        for j in range(active_dim):
            A[i, j] = np.random.normal(0, 1)

    Q, R = np.linalg.qr(A)
    assert np.allclose(np.dot(Q.T, Q), np.eye(Q.shape[1]))
    assert Q.shape[0] == real_dim
    assert Q.shape[1] == active_dim
    return Q

