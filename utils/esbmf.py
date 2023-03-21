import numpy as np
from typing import Tuple


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """
    Calculate the hamming distance between two boolean matrices of the same shape.

    Parameters
    ----------
    a: ndarray
    b: ndarray

    Returns
    -------
    distance: int, the Hamming distance of a and b
    """
    assert a.shape == b.shape
    return np.count_nonzero(a ^ b)


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
    assert orig.shape == approx.shape and dc_row.shape[0] == orig.shape[1] and len(dc_row.shape) == 1
    rows = orig.shape[0]
    diff = lambda i: hamming_distance(approx[i, :], orig[i, :]) - hamming_distance((approx | dc_row)[i, :], orig[i, :])
    diff_col = np.vectorize(diff)(np.arange(rows))
    num_p, num_n = np.count_nonzero(diff_col > 0), np.count_nonzero(diff_col < 0)
    replace_zeros = np.where(diff_col == 0, int(num_p > num_n), diff_col)
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
    assert orig.shape == approx.shape and c_col.shape[1] == 1 and orig.shape[0] == c_col.shape[0]
    cols = orig.shape[1]
    diff = lambda i: hamming_distance(approx[:, i], orig[:, i]) - hamming_distance((approx | c_col)[:, i], orig[:, i])
    diff_row = np.vectorize(diff)(np.arange(cols))
    dc_row = np.where(diff_row < 0, 0, diff_row).astype(bool)
    return dc_row


def calc_association_matrix(boolean_matrix: np.ndarray, threshold: float) -> np.ndarray:
    """
    Calculate the association matrix

    Parameters
    ----------
    boolean_matrix: numpy.ndarray
    threshold: float

    Returns
    -------
    association_matrix: numpy.ndarray
    """
    assert len(boolean_matrix.shape) == 2 and threshold > 0.0 and threshold < 1.0
    cols = boolean_matrix.shape[1]
    float_matrix = boolean_matrix.astype(float)
    inner_product = lambda i, j: np.dot(float_matrix[:, i], float_matrix[:, j])
    association = lambda i, j: inner_product(i, j) / inner_product(i, i) >= threshold
    association_matrix = np.vectorize(association)(*np.indices((cols, cols)))
    return association_matrix


def search_space_under_threshold(boolean_matrix: np.ndarray, threshold: float) -> np.ndarray:
    """
    Search space under a given threshold

    Parameters
    ----------
    boolean_matrix: numpy.ndarray
    threshold: float

    Returns
    -------
    search_space: numpy.ndarray
    """
    assert len(boolean_matrix.shape) == 2 and threshold > 0.0 and threshold < 1.0
    rows = boolean_matrix.shape[0]
    cols = boolean_matrix.shape[1]
    # conditional probability of P( B[x, j] == 1 | B[x, i] == 1 )
    cp_true = lambda i, j: np.count_nonzero(boolean_matrix[:, i] & boolean_matrix[:, j]) / np.count_nonzero(
        boolean_matrix[:, i])
    # conditional probability of P( B[x, j] == 0 | B[x, i] == 0 ), 2 for division by zero
    cp_false = lambda i, j: 2 if (np.count_nonzero(boolean_matrix[:, i]) == rows) else (rows - np.count_nonzero(
        boolean_matrix[:, i] | boolean_matrix[:, j])) / (rows - np.count_nonzero(boolean_matrix[:, i]))
    cp_true_matrix = np.vectorize(lambda i, j: cp_true(i, j) > threshold)(*np.indices((cols, cols)))
    cp_false_matrix = np.vectorize(lambda i, j: cp_false(i, j) <= threshold)(*np.indices((cols, cols)))
    # drop zero rows
    cp_false_matrix = cp_false_matrix[np.any(cp_false_matrix != 0, axis=1)]
    search_space = np.unique(np.vstack([cp_true_matrix, cp_false_matrix]), axis=0)
    return search_space


def search_space_generate(boolean_matrix: np.ndarray, step: float, mode: str) -> np.ndarray:
    """
    Generate the search space of the basic row of decompressor

    Parameters
    ----------
    boolean_matrix: numpy.ndarray
        the boolean matrix used for generation
    step: float
        advance step of the threshold
    mode: str
        'cp': conditional probability
        'asso': association matrix

    Returns
    -------
    search_space: numpy.ndarray
    """
    assert len(boolean_matrix.shape) == 2

    if mode == 'cp':
        search_space_t = lambda t: search_space_under_threshold(boolean_matrix, t)
    elif mode == 'asso':
        search_space_t = lambda t: calc_association_matrix(boolean_matrix, t)
    else:
        raise ValueError("Invalid mode. Must be 'cp' or 'asso'.")

    stack = np.vstack([search_space_t(t) for t in np.arange(step, 1.0, step)])
    search_space = np.unique(stack, axis=0)
    return search_space


def cal_current_error_reduction(orig: np.ndarray, approx_pre: np.ndarray, approx: np.ndarray) -> int:
    """
    Calculate the current error reduction

    Parameters
    ----------
    orig: numpy.ndarray
        original boolean matrix
    approx_pre: numpy.ndarray
        pre-approximated matrix
    approx: numpy.ndarray
        current approximated matrix

    Returns
    -------
    reduction: int
        the current error reduction compared with pre-approximated matrix
    """
    assert orig.shape == approx_pre.shape and orig.shape == approx.shape
    reduction = hamming_distance(approx_pre, orig) - hamming_distance(approx, orig)
    return reduction


def cal_top_n(arr: np.ndarray, n: int, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the top-n max/min elements of the array based on the given mode.

    Parameters
    ----------
    arr: numpy.ndarray
    n: top-n
    mode: str, 'max' or 'min'

    Returns
    -------
    top_n_elems: the top-n max/min elements
    indices: the related indices
    """
    if mode == 'max':
        top_n_indices = np.argpartition(-arr, n)[:n]
        sorted_indices = np.argsort(-arr[top_n_indices])
    elif mode == 'min':
        top_n_indices = np.argpartition(arr, n)[:n]
        sorted_indices = top_n_indices[np.argsort(arr[top_n_indices])]
    else:
        raise ValueError("Invalid mode. Must be 'max' or 'min'.")
    indices = top_n_indices[sorted_indices]
    top_n_elems = arr[indices]
    return top_n_elems, indices


def cal_top_n_col_error(orig: np.ndarray, approx: np.ndarray, n: int, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the top-n max/min errors of each column based on the given mode.

    Parameters
    ----------
    orig: numpy.ndarray
    approx: numpy.ndarray
    n: top-n
    mode: str, 'max' or 'min'

    Returns
    -------
    top_n_errors: the top-n max/min errors
    indices: the related indices
    """
    assert orig.shape == approx.shape
    error = (orig ^ approx)
    error_per_col = np.sum(error, axis=0)
    return cal_top_n(error_per_col, n, mode)


def generate_next_approx(orig: np.ndarray, approx: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate next rank-1 approximate matrix of the original boolean matrix

    Parameters
    ----------
    orig: numpy.ndarray
        the original boolean matrix
    approx: numpy.ndarray
        the (k-1)-th approximate matrix of the orig
    k: int
        the k-th

    Returns
    -------
    c_col: numpy.ndarray
    dc_row: numpy.ndarray
    resid_error:
    """
    assert orig.shape == approx.shape
    # using cp - conditional probability to generate search space
    search_space = search_space_generate(orig | approx, 0.1, 'cp')
    rows, cols = orig.shape
    search_space_rows = search_space.shape[0]
    # choose i-th row of search_space as a row of decompressor
    mem_dc_row = lambda i: search_space[i, :]
    # matrix error minimize scheme
    mem_c_col = lambda i: boolean_matrix_error_minimize(orig, approx, mem_dc_row(i))[:, np.newaxis]
    # all candidates from mem
    mem_approx = lambda i: approx | (mem_c_col(i) & mem_dc_row(i))
    # error reduction of all candidates of mem
    mem_err_reduct = lambda i: cal_current_error_reduction(orig, approx, mem_approx(i))
    mem_err_reducts = np.array([mem_err_reduct(i) for i in np.arange(search_space_rows)], dtype=int)
    # TODO: residual errors of all candidate of mem
    mem_resid_errs = lambda i: np.sum(cal_top_n_col_error(orig, mem_approx(i), cols - k, 'min')[0])
    # error matrix for cec scheme
    cec_error = orig ^ approx
    # choose j-th col of error matrix as a col of compressor
    cec_c_col = lambda j: (cec_error[:, j])[:, np.newaxis]
    # column error clean scheme
    cec_dc_row = lambda j: boolean_matrix_column_error_clear(orig, approx, cec_c_col(j))
    # all candidates from cec
    # TODO: addition type
    cec_approx = lambda j: approx | (cec_c_col(j) & cec_dc_row(j))
    # error reduction of all candidates of cec
    cec_err_reduct = lambda j: cal_current_error_reduction(orig, approx, cec_approx(j))
    cec_err_reducts = np.array([cec_err_reduct(j) for j in np.arange(cols)], dtype=int)
    # greedy scheme
    mem_max_err_reduct, mem_index = cal_top_n(mem_err_reducts, 1, 'max')
    cec_max_err_reduct, cec_index = cal_top_n(cec_err_reducts, 1, 'max')
    # capture element
    mem_max_err_reduct, mem_index = mem_max_err_reduct[0], mem_index[0]
    cec_max_err_reduct, cec_index = cec_max_err_reduct[0], cec_index[0]

    # TODO: remove debug info
    print(f"====================\nthe search space when k ={k}")
    print(search_space.astype(int))
    print("cec err reduction:")
    print(cec_err_reducts)
    print("mem err reduction:")
    print(mem_err_reducts)
    print(f"cec error: {cec_max_err_reduct}, mem error: {mem_max_err_reduct}")
    if (cec_max_err_reduct > mem_max_err_reduct):
        print(f"using cec when k = {k}")
    else:
        print(f"using mem when k = {k}")

    # optimum result
    opt_c_col, opt_dc_row = (cec_c_col(cec_index), cec_dc_row(cec_index)) if (
        cec_max_err_reduct > mem_max_err_reduct) else (mem_c_col(mem_index), mem_dc_row(mem_index))
    return opt_c_col, opt_dc_row


def generate_approx(orig: np.ndarray, k: int, f: int) -> Tuple[list, list]:
    """
    Generate approximate boolean matrix in a recursion way

    Parameters
    ----------
    orig: numpy.ndarray
        the original boolean matrix
    k: int
        the k-th generation step
    f: int
        the factorization degree

    Returns
    -------
    c_matrix: list
    dc_matrix: list
    """
    if k == 1:
        approx_init = np.zeros_like(orig, dtype=bool)
        c, dc = generate_next_approx(orig, approx_init, k)
        c_matrix, dc_matrix = np.hstack([c]), np.vstack([dc])
    else:
        prev_c_matrix, prev_dc_matrix = generate_approx(orig, k - 1, f)
        prev_approx = np.dot(prev_c_matrix, prev_dc_matrix)
        c, dc = generate_next_approx(orig, prev_approx, k)
        c_matrix, dc_matrix = np.hstack([prev_c_matrix, c]), np.vstack([prev_dc_matrix, dc])

    # TODO: remove debug info
    print(f"the approx when k = {k}:")
    print(np.dot(c_matrix, dc_matrix).astype(int))

    return c_matrix, dc_matrix
