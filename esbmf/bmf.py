import numpy as np
from esbmf import utils


class BMF:

    def __init__(self, orig, f, w=None):
        self.orig = orig
        self.rows, self.cols = self.orig.shape
        self.f = f
        self.w = w or (np.ones(self.cols, dtype=float) / self.cols)
        self.c = None
        self.dc = None
        self.add_t = None
        self.approx = None
        self.__search_space = None
        self.__p_c = None
        self.__p_dc = None
        self.__res = None
        self.__greedy_scheme_err = None
        self.__enable_err_shape = False

    def __greedy_scheme_init(self):
        self.c = None
        self.dc = None
        self.add_t = None
        self.approx = np.zeros_like(self.orig, dtype=bool)
        self.__search_space = None
        self.__p_c = None
        self.__p_dc = None
        self.__res = None

    def __err_shape_scheme_init(self):
        self.__greedy_scheme_err = utils.weighted_distance(self.orig, self.approx, self.w)
        self.__enable_err_shape = False
        self.approx = np.zeros_like(self.orig, dtype=bool)

    def __generate_search_space(self, step: float, mode: str):
        assert len(self.approx.shape) == 2
        m = self.orig | self.approx
        # TODO: cp or asso, remove one of them later
        if mode == 'cp':
            search_space_t = lambda t: utils.calc_cond_prob_matrix(m, t)
        elif mode == 'asso':
            search_space_t = lambda t: utils.calc_asso_matrix(m, t)
        else:
            raise ValueError("Invalid mode. Must be 'cp' or 'asso'.")
        stack = np.vstack([search_space_t(t) for t in np.arange(step, 1.0, step)])
        self.__search_space = np.unique(stack, axis=0)

    def __matrix_error_minimize(self, dc_row):
        assert dc_row.shape[0] == self.orig.shape[1] and len(dc_row.shape) == 1
        e = 1e-10
        diff = lambda i: np.sum((self.approx[i, :] ^ self.orig[i, :]) * self.w) - np.sum(
            ((self.approx | dc_row)[i, :] ^ self.orig[i, :]) * self.w)
        diff_col = np.vectorize(diff)(np.arange(self.rows))
        num_p, num_n = np.count_nonzero(diff_col > e), np.count_nonzero(diff_col < -e)
        replace_zeros = np.where(np.abs(diff_col) <= e, int(num_p > num_n), diff_col)
        c_col = np.where(replace_zeros < 0, 0, replace_zeros).astype(bool)
        c_col = c_col[:, np.newaxis]
        return c_col

    def __column_error_clear(self, c_col, j_xor):
        assert c_col.shape[1] == 1 and self.orig.shape[0] == c_col.shape[0]
        diff = lambda j: utils.hamming_distance(self.approx[:, j], self.orig[:, j]) - utils.hamming_distance(
            utils.calc_mix_addition(self.approx, np.tile(c_col, (1, self.cols)), j_xor)[:, j], self.orig[:, j])
        diff_row = np.vectorize(diff)(np.arange(self.cols))
        dc_row = np.where(diff_row < 0, 0, diff_row).astype(bool)
        return dc_row

    def __get_cur_err_reduct(self, cur_approx):
        assert self.orig.shape == cur_approx.shape
        return utils.weighted_distance(self.approx, self.orig, self.w) - utils.weighted_distance(
            cur_approx, self.orig, self.w)

    def __get_cur_resid_err(self, cur_approx, k):
        assert self.orig.shape == cur_approx.shape
        n = self.cols - (self.f - k - 1)
        err = np.sum(self.orig ^ cur_approx, axis=0) * self.w * self.cols
        resid_err = np.sum(err) if (n == self.cols) else np.sum(np.partition(err, n)[:n])
        return resid_err

    def __update_approx(self, k):
        enable_cec = not np.all(self.add_t[k, :] == False)
        c_col, dc_row = self.c[:, k][:, np.newaxis], self.dc[k, :]
        add_t = self.add_t[k, :]
        approx = utils.calc_mix_addition(self.approx, c_col & dc_row, np.where(add_t)[0][0]) if (enable_cec) \
            else self.approx | (c_col & dc_row)
        self.approx = approx

    def __greedy_scheme(self, k):
        assert self.orig.shape == self.approx.shape
        # using cp - conditional probability to generate search space
        self.__generate_search_space(0.1, 'cp')
        ss_rows = self.__search_space.shape[0]
        # choose i-th row of search_space as a row of decompressor
        mem_dc_row = lambda i: self.__search_space[i, :]
        # matrix error minimize scheme
        mem_c_col = lambda i: self.__matrix_error_minimize(mem_dc_row(i))
        # all candidates from mem
        mem_approx = lambda i: self.approx | (mem_c_col(i) & mem_dc_row(i))
        # error reduction of all candidates of mem
        mem_err_reduct = lambda i: self.__get_cur_err_reduct(mem_approx(i))
        # one/zero proportion of dc_row
        mem_one_prop = lambda i: np.count_nonzero(mem_dc_row(i)) / len(mem_dc_row(i))
        mem_zero_prop = lambda i: 1.0 - mem_one_prop(i)
        # mem score = err_reduct + zero_prop
        mem_scores = np.array([mem_err_reduct(i) + mem_zero_prop(i) for i in np.arange(ss_rows)], dtype=float)
        # residual error of all candidates of mem
        mem_resid_err = lambda i: self.__get_cur_resid_err(mem_approx(i), k)
        # mem residual error score = resid_err + one_prop
        mem_re_scores = np.array([mem_resid_err(i) + mem_one_prop(i) for i in np.arange(ss_rows)], dtype=float)
        # error matrix for cec scheme
        cec_error = self.orig ^ self.approx
        # choose j-th col of error matrix as a col of compressor
        cec_c_col = lambda j: (cec_error[:, j])[:, np.newaxis]
        # column error clean scheme
        cec_dc_row = lambda j: self.__column_error_clear(cec_c_col(j), j)
        # all candidates from cec
        cec_approx = lambda j: utils.calc_mix_addition(self.approx, cec_c_col(j) & cec_dc_row(j), j)
        # error reduction of all candidates of cec
        cec_err_reduct = lambda j: self.__get_cur_err_reduct(cec_approx(j))
        # zero proportion of dc_row
        cec_zero_prop = lambda j: 1.0 - np.count_nonzero(cec_dc_row(j)) / len(cec_dc_row(j))
        # cec score = err_reduct + zero_prop
        cec_scores = np.array([cec_err_reduct(j) + cec_zero_prop(j) for j in np.arange(self.cols)], dtype=float)
        # greedy scheme
        mem_max_score, mem_index = np.max(mem_scores), np.argmax(mem_scores)
        cec_max_score, cec_index = np.max(cec_scores), np.argmax(cec_scores)
        enable_cec = cec_max_score > mem_max_score
        mem_re_index = np.argmin(mem_re_scores)
        mem_min_re = mem_resid_err(mem_re_index)
        # optimum result
        c_col, dc_row = (cec_c_col(cec_index), cec_dc_row(cec_index)) if (enable_cec) \
            else (mem_c_col(mem_index), mem_dc_row(mem_index))
        # cec add type
        add_t = np.array([True if j == cec_index else False for j in range(self.cols)], dtype=bool) if (enable_cec) \
            else np.zeros(self.cols, dtype=bool)
        # potential result
        p_c_col, p_dc_row = mem_c_col(mem_re_index), mem_dc_row(mem_re_index)
        # update
        self.c = np.hstack([c_col]) if (k == 0) else np.hstack([self.c, c_col])
        self.dc = np.vstack([dc_row]) if (k == 0) else np.vstack([self.dc, dc_row])
        self.add_t = np.vstack([add_t]) if (k == 0) else np.vstack([self.add_t, add_t])
        self.__p_c = np.hstack([p_c_col]) if (k == 0) else np.hstack([self.__p_c, p_c_col])
        self.__p_dc = np.vstack([p_dc_row]) if (k == 0) else np.vstack([self.__p_dc, p_dc_row])
        self.__res = np.hstack([mem_min_re]) if (k == 0) else np.hstack([self.__res, mem_min_re])
        self.__update_approx(k)

    def __err_shape_scheme(self, k):
        assert self.orig.shape == self.approx.shape
        if (self.__enable_err_shape):
            err = self.orig ^ self.approx
            idx = np.argmax(np.sum(err, axis=0) * self.w * self.cols)
            c_col = err[:, idx]
            dc_row = self.__column_error_clear(c_col[:, np.newaxis], idx)
            add_t = np.array([True if j == idx else False for j in range(self.cols)], dtype=bool)
            self.c[:, k], self.dc[k, :] = c_col, dc_row
            self.add_t[k, :] = add_t
        else:
            if (self.__res[k] < self.__greedy_scheme_err and k < self.f - 1):
                self.c[:, k], self.dc[k, :] = self.__p_c[:, k], self.__p_dc[k, :]
                self.__enable_err_shape = True
        self.__update_approx(k)

    def __run_greedy_scheme(self):
        self.__greedy_scheme_init()
        for k in range(0, self.f):
            self.__greedy_scheme(k)

    def __run_err_shape_scheme(self):
        self.__err_shape_scheme_init()
        for k in range(0, self.f):
            self.__err_shape_scheme(k)

    def run_esbmf(self):
        self.__run_greedy_scheme()
        self.__run_err_shape_scheme()
