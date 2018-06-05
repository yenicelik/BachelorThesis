"""
This file contains the code to generate a set of basis vectors that are each individually orthogonal
to each vector within a given matrix A
"""
import numpy as np


def create_colwise_normed_A(A):
    normed_A = np.empty_like(A)
    for i in range(A.shape[1]):
        normed_A[:, i] = A[:, i] / np.linalg.norm(A[:, i])
        assert normed_A[:, i].shape == (A.shape[0],), (normed_A.shape, normed_A[:, i].shape, A.shape)

    return normed_A


def generate_orthogonal_matrix_to_A(A, n):
    """
        This is the overarching function that generates a set of basis vectors for a given matrix A
    :param A:
    :param n: Is the number of dimensions that the new orthogonal basis should have
    :return:
    """
    dims = A.shape[0]

    # We will need this to compute the norm between two vectors later
    normed_A = create_colwise_normed_A(A)

    # The basis that we want to output
    Q = np.empty((A.shape[0], 0))

    for i in range(n):
        counter = 0

        while True:  # Do this until we find a vector that is well orthogonalizable! (i.e. the dot product tolerance is below a certain threshold!)
            counter += 1

            # Generate a random vector
            np.random.seed()
            q_i = np.random.rand(dims)
            q_i = q_i / np.linalg.norm(q_i)

            # Apply gram schmidt on that that vector w.r.t. to all column vectors
            tmp_all = np.concatenate((normed_A, Q), axis=1)
            new_basis = apply_gram_schmidt_single_vector(tmp_all, q_i)

            if np.isclose(np.dot(normed_A.T, new_basis), 0).all() and (new_basis > 1e-8).any():
                np.concatenate((Q, new_basis.reshape(-1, 1)), axis=1)
                break
            if counter > 100:
                print("Not orthogonal column found!!", (normed_A, new_basis, np.dot(normed_A.T, new_basis)))
                exit(0)

    return Q


def apply_gram_schmidt_single_vector(A, q):
    """
    :param A: Is the matrix that we want the new vector to be orthogonal to
    :param q: Is the vector that we want to orthogonalize w.r.t. all vectors in A
    :return:
    """
    assert q.shape == (A.shape[0],), ("Something is not right when receiving the vector q!", q.shape)

    # We need to apply the orthogonalization to an orthogonal matrix Q!
    Q, _ = np.linalg.qr(A)

    for i in range(Q.shape[1]):
        v_i = Q[:, i]
        projection = (np.vdot(q, v_i) / np.vdot(v_i, v_i)) * v_i
        q = np.subtract(q, projection)

    assert q.shape == (A.shape[0],), ("Something went wrong when orthogonalizing q!", q.shape)

    return q / np.linalg.norm(q, ord=2)


if __name__ == "__main__":
    print("Starting...")

    np.random.seed()
    A = np.random.rand(5, 4)
    generate_orthogonal_matrix_to_A(A, 1)
