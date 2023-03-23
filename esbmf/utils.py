import numpy as np


def hamming_distance(a, b):
    assert a.shape == b.shape
    return np.count_nonzero(a ^ b)


def weighted_distance(a, b, w):
    assert a.shape == b.shape and len(a.shape) == 2 and w.shape[0] == a.shape[1]
    return np.sum(np.sum(a ^ b, axis=0) * w) * a.shape[1]


def calc_asso_matrix(m, t):
    assert len(m.shape) == 2 and t > 0.0 and t < 1.0
    cols = m.shape[1]
    float_matrix = m.astype(float)
    inner_product = lambda i, j: np.dot(float_matrix[:, i], float_matrix[:, j])
    asso = lambda i, j: inner_product(i, j) / inner_product(i, i) >= t
    asso_matrix = np.vectorize(asso)(*np.indices((cols, cols)))
    return asso_matrix


def calc_cond_prob_matrix(m, t):
    assert len(m.shape) == 2 and t > 0.0 and t < 1.0
    rows = m.shape[0]
    cols = m.shape[1]
    # conditional probability of P( B[x, j] == 1 | B[x, i] == 1 )
    cp_true = lambda i, j: np.count_nonzero(m[:, i] & m[:, j]) / np.count_nonzero(m[:, i])
    # conditional probability of P( B[x, j] == 0 | B[x, i] == 0 ), 2 for division by zero
    cp_false = lambda i, j: 2 if (np.count_nonzero(m[:, i]) == rows) \
        else (rows - np.count_nonzero(m[:, i] | m[:, j])) / (rows - np.count_nonzero(m[:, i]))
    cp_true_matrix = np.vectorize(lambda i, j: cp_true(i, j) > t)(*np.indices((cols, cols)))
    cp_false_matrix = np.vectorize(lambda i, j: cp_false(i, j) <= t)(*np.indices((cols, cols)))
    # drop zero rows
    cp_false_matrix = cp_false_matrix[np.any(cp_false_matrix != 0, axis=1)]
    cond_prob_matrix = np.unique(np.vstack([cp_true_matrix, cp_false_matrix]), axis=0)
    return cond_prob_matrix


def calc_mix_addition(a, b, j):
    assert a.shape == b.shape and len(a.shape) == 2 and j < a.shape[1]
    col_xor = (a[:, j] ^ b[:, j])[:, np.newaxis]
    col_or_left = a[:, :j] | b[:, :j]
    col_or_right = a[:, j + 1:] | b[:, j + 1:]
    result = np.hstack([col_or_left, col_xor, col_or_right])
    return result
