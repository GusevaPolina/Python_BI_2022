import numpy as np

if __name__ == '__main__':
    # my favourite arrays
    a1 = np.arange(20).reshape((4, 5))
    a2 = np.eye(5)
    a3 = np.empty_like(a1, dtype=object)


# true matrix multiplication
def matrix_multiplication(b1, b2):
    return np.matmul(b1, b2)


# ability of matrices to be multiplied
def multiplication_check(c_list):
    c1 = np.vectorize(lambda x: np.array(c_list[x]).shape)(np.arange(len(c_list)))
    return all(c1[0][1:] == c1[1][:-1])


# multiplication of matrices
def multiply_matrices(d_list):
    if multiplication_check(d_list):
        return np.linalg.multi_dot(d_list)
    else:
        return None


# euclidian distance
def compute_2d_distance(e1, e2):
    return np.linalg.norm(e1 - e2)


# euclidian distance in multidimentions
def compute_multidimensional_distance(f1, f2):
    return np.linalg.norm(f1 - f2)


# distance matrix
def compute_pair_distances(g):
    if g.shape[-1] != 2:
        g = g.T
    return np.linalg.norm(g[:, None, :] - g[None, :, :], axis=-1)
