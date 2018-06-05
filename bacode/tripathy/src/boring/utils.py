
import numpy as np
from numpy.linalg import lstsq
from scipy.linalg import orth

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

# Procedures to generate a basis orthogonal to the basis of the image of A
def gs_coefficient(v1, v2):
    out = np.dot(v2.T, v1) / np.dot(v1, v1)
    return out

def proj(v1, v2):
    out = gs_coefficient(v1, v2)
    out = np.multiply(out, v1)
    return out

def gs(X):
    Y = []
    for i in range(len(X)):
        temp_vec = X[i]
        for inY in Y :
            proj_vec = proj(inY, X[i])
            temp_vec = temp_vec - proj_vec
        Y.append(temp_vec.reshape(-1, 1))
    Y = np.concatenate(Y, axis=1)
    return Y

def find_orthogonal_kernel_basis(A, additional_dim):
    assert A.shape[1] + additional_dim < A.shape[0], ("Exhausted all dimensions")
    columns_A = [A[:,i].reshape(-1, 1) for i in range(A.shape[1])]
    print(columns_A)

    # Iterate through v_1, ..., v_{q} where q = additional_dim
    for i in range(additional_dim):
        # Sample a random vector which then will be orthonormalized!
        x = np.random.rand(A.shape[0], 1)

        # Iterate through a_1, ..., a_{d_e}
        for a_j in columns_A:

            # Get the projection of v_i on a_j
            coeff = np.dot(a_j.T, x) / np.dot(a_j)






# def get_orthogonal_subspace_matrix(A, real_dim, active_dim):
#     """
#         Given a matrix A and it's column vectors, this function gives another matrix B, where each vector of B is
#         1.) orthonormal to all other vectors in A and B
#     :param A:
#     :return:
#     """
#     assert A.shape == (real_dim, active_dim), ("A is not correctly aligned! ", A.shape, (real_dim, active_dim))


# def find_orth(O):
#     rand_vec = np.random.rand(O.shape[0], 1)
#     A = np.hstack((O, rand_vec))
#     b = np.zeros(O.shape[1] + 1)
#     b[-1] = 1
#     return lstsq(A.T, b)[0]

if __name__ == "__main__":

    A = sample_orthogonal_matrix(3, 2)
    print("A", A.shape)

    B = sample_orthogonal_matrix(3, 3)
    print("B", B.shape)

    # Let's check if this gives us an identity matrix
    V = np.dot(A.T, B).T
    print("V", V.shape)

    # Check if this gives us a orthonormal matrix:
    AV = np.concatenate((A, V), axis=1)
    # print("AV", AV.shape)
    # res = np.dot(AV.T, AV)
    # print(res)

    new_null_columns = AV[:, 0]
    AV = np.concatenate((A, new_null_columns), axis=1)
    res = np.dot(AV.T, AV)
    print(res)