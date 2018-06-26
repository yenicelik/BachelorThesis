import numpy as np

# A = np.zeros((self.real_dim, self.active_dim), dtype=np.float64)
# for i in range(self.real_dim):
#     for j in range(self.active_dim):
#         A[i, j] = np.random.normal(0, 1)
#
# Q, R = np.linalg.qr(A)
# assert np.allclose(np.dot(Q.T, Q), np.eye(Q.shape[1]))
# assert Q.shape[0] == self.real_dim
# assert Q.shape[1] == self.active_dim
# return Q

def sample_orthogonal_matrix(real_dim, active_dim, seed=None):
    """
    :return: An orthogonal matrix
    """
    np.random.seed(seed)
    A = np.zeros((real_dim, active_dim), dtype=np.float64)
    for i in range(real_dim):
        for j in range(active_dim):
            A[i, j] = np.random.normal(0, 1)

    Q, R = np.linalg.qr(A)
    print("Dimensions of Q are: ", Q.shape)
    assert np.allclose(np.dot(Q.T, Q), np.eye(Q.shape[1]))
    assert A.shape[0] == real_dim, ("Something went terribly wrong! ", A.shape, real_dim)
    assert A.shape[1] == active_dim, ("Something went terribly wrong! ", A.shape, active_dim)
    assert Q.shape[0] == real_dim, ("Shapes are: ", Q.shape, real_dim)
    assert Q.shape[1] == active_dim, ("Shapes are: ", Q.shape, active_dim)
    return Q

