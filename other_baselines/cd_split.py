import numpy as np
from tqdm.autonotebook import tqdm

from sklearn.cluster import KMeans
from cde.density_estimator import KernelMixtureNetwork, MixtureDensityNetwork
from sympy import Interval, Union


def union(data):
    """ Union of a list of intervals e.g. [(1,2),(3,4)] """
    intervals = [Interval(begin, end) for (begin, end) in data]
    u = Union(*intervals)
    if isinstance(u, Interval):
        return [list(u.args[:2])]
    else:
        result = [list(v.args[:2]) for v in list(u.args)]
        return result


def point_wise_interval(y, epsilon):
    bands = []
    for m in y:
        bands.append((m - epsilon, m + epsilon))
    return union(bands)


def profile_density(t_grid, y_grid, X_train, model):
    y_num, n_levels = len(y_grid), len(t_grid)
    delta_y = y_grid[2] - y_grid[1]

    n, n_feats = X_train.shape
    X_rep = np.tile(X_train, y_num).reshape(n * y_num, n_feats)
    y_grid_rep = np.tile(y_grid, n).reshape(n * y_num, 1)
    pdf_grid = model.pdf(X_rep, y_grid_rep).reshape(n, y_num)
    cde = []
    for t in t_grid:
        cde_t = np.where(pdf_grid >= t, pdf_grid, 0.0).sum(axis=1, keepdims=True) * delta_y
        cde.append(cde_t)
    cde = np.concatenate(cde, axis=1)
    return cde


class CDSplit:
    """
    Our implementation of CD_split in python, since the R version is too slow...
    """
    def __init__(self, X_train, Y_train, y_num=100, ymin=-1, ymax=1, seed=0, alpha=0.1, name = "y0", model = None):
        # name argument is used for md, please don't delete
        self.seed = seed
        self.alpha = alpha
        x_dim, y_dim = X_train.shape[1], 1
        split = X_train.shape[0] // 2
        self.X_train, self.Y_train = X_train.reshape(-1, x_dim), Y_train.reshape(-1, 1)

        self.y_num = y_num
        self.ymin, self.ymax = ymin, ymax
        self.y_grid = np.linspace(self.ymin, self.ymax, self.y_num)

        if model:
            self.regr = model
        else:
            kmn_model = KernelMixtureNetwork(name=f"KMN-CD{seed}_{name}", ndim_x=x_dim, ndim_y=y_dim,
                                             hidden_sizes=(50, 50), random_seed=seed)
            self.regr = kmn_model
            self.regr.fit(self.X_train, self.Y_train)

    def calibrate(self, X_calib, Y_calib, alpha):
        self.X_calib, self.Y_calib = X_calib.reshape(-1, X_calib.shape[1]), Y_calib.reshape(-1, 1)

        n_levels = self.n_levels = 100
        n_calib = self.X_calib.shape[0]
        n_clusters = max(round(n_calib / 100), 1)

        self.alpha = alpha
        # Get Kmeans Centers
        cde = self.regr.pdf(self.X_train, self.Y_train)
        t_grid = self.t_grid = np.linspace(0, max(cde), n_levels)

        cde_calib = self.cde_calib = profile_density(t_grid, self.y_grid, self.X_calib, self.regr)

        self.kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed).fit(cde_calib)
        self.calib_scores = self.regr.pdf(self.X_calib, self.Y_calib)

    def predict(self, X_test):
        n_test, n_feats = X_test.shape
        n_levels = self.n_levels

        cde_test = profile_density(self.t_grid, self.y_grid, X_test, self.regr)

        calib_class = self.kmeans.predict(self.cde_calib)
        test_class = self.kmeans.predict(cde_test)

        y_num = self.y_num
        y_grid = self.y_grid
        delta_y = y_grid[1] - y_grid[0]
        bands = []
        for i, tc in tqdm(enumerate(test_class)):
            scores = self.calib_scores[calib_class == tc]
            ths = np.quantile(scores, self.alpha)
            yi_mask = (self.regr.pdf(np.tile(X_test[i], y_num).reshape(y_num, n_feats), y_grid.reshape(-1, 1)) >= ths)
            yi_interval = point_wise_interval(y_grid[yi_mask], delta_y)
            bands.append(yi_interval)

        #del self.regr  # for saving memory
        return bands

