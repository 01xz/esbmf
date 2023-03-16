import numpy as np


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """
    Calculate the hamming distance between two boolean matrices of the same shape.

    Parameters
    ----------
    a: numpy.ndarray
    b: numpy.ndarray

    Returns
    -------
    distance: int, the Hamming distance of a and b
    """
    return np.count_nonzero(a ^ b)


def calc_association_matrix(boolean_matrix: np.ndarray, threshold: int) -> np.ndarray:
    """
    Calculate the association matrix

    Parameters
    ----------
    boolean_matrix: numpy.ndarray
    threshold: int

    Returns
    -------
    association_matrix: numpy.ndarray
    """
    cols = boolean_matrix.shape[1]
    inner_product = lambda i, j: np.dot(boolean_matrix[:, i], boolean_matrix[:, j])
    association = lambda i, j: inner_product(i, j) / inner_product(i, i) >= threshold
    association_matrix = np.vectorize(association)(*np.indices((cols, cols)))
    return association_matrix


def boolean_matrix_error_minimize(orig: np.ndarray, approx: np.ndarray, dc_row: np.ndarray) -> np.ndarray:
    """
    return a compressor column

    Parameters
    ----------
    orig: original boolean matrix
    approx: approximated boolean matrix
    dc_row: a row of the decompressor

    Returns
    -------
    c_col: a column of the compressor
    """
    rows = orig.shape[0]
    diff = lambda i: hamming_distance(approx[i, :], orig[i, :]) - hamming_distance((approx | dc_row)[i, :], orig[i, :])
    diff_col = np.vectorize(diff)(np.arange(rows))
    print("diff_col:")
    print(diff_col)
    num_p, num_n = np.count_nonzero(diff_col > 0), np.count_nonzero(diff_col < 0)
    # print(f"p: {num_p}, n: {num_n}")
    replace_zeros = np.where(diff_col == 0, int(num_p > num_n), diff_col)
    # print("replace zeros:")
    # print(replace_zeros)
    c_col = np.where(replace_zeros < 0, 0, replace_zeros).astype(bool)
    return c_col


def boolean_matrix_column_error_clear(orig: np.ndarray, approx: np.ndarray, c_col: np.ndarray) -> np.ndarray:
    """
    return a decompressor row

    Parameters
    ----------
    orig: original boolean matrix
    approx: approximated boolean matrix
    c_col: a col of the compressor

    Returns
    -------
    dc_row: a row of the decompressor
    """
    cols = orig.shape[1]
    diff = lambda i: hamming_distance(approx[:, i], orig[:, i]) - hamming_distance(
        (approx.transpose() | c_col).transpose()[:, i], orig[:, i])
    diff_row = np.vectorize(diff)(np.arange(cols))
    print("diff_row:")
    print(diff_row)
    c_col = np.where(diff_row < 0, 0, diff_row).astype(bool)
    return c_col
