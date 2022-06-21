import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import collections

from chr import coverage

import pdb

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class RegressionDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = torch.from_numpy(X_data).float()
        self.y_data = torch.from_numpy(y_data).float()

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__ (self):
        return len(self.X_data)

def evaluate_predictions(pred, Y, X=None):
    # Extract lower and upper prediction bands
    pred_l = np.min(pred,1)
    pred_h = np.max(pred,1)
    # Marginal coverage
    cover = (Y>=pred_l)*(Y<=pred_h)
    marg_coverage = np.mean(cover)
    if X is None:
        wsc_coverage = None
    else:
        # Estimated conditional coverage (worse-case slab)
        wsc_coverage = coverage.wsc_unbiased(X, Y, pred, M=100)

    # Marginal length
    lengths = pred_h-pred_l
    length = np.mean(lengths)
    # Length conditional on coverage
    idx_cover = np.where(cover)[0]
    length_cover = np.mean([lengths for i in idx_cover])

    # Combine results
    out = pd.DataFrame({'Coverage': [marg_coverage], 'Conditional coverage': [wsc_coverage],
                        'Length': [length], 'Length cover': [length_cover]})
    return out


def evaluate_predictions_pcp(pred, Y, X=None):
    # Extract lower and upper prediction bands
    # evaluations
    coverages = []
    lengths = []
    n_interval = []
    bands = pred 
    y_te = Y.reshape(-1,1)
    for i_test in range(len(y_te)):
        band_i = bands[i_test]
        y_i = y_te[i_test]
        n_interval.append(len(band_i))
        coverage = 0
        length = 0
        for interval_i in band_i:
            length += (interval_i[1] - interval_i[0])
            if ((y_i>interval_i[0]) & (y_i<interval_i[1])):
                coverage = 1
        coverages.append(coverage)
        lengths.append(length)

    marg_coverage = np.mean(coverages)
    # to do 
    wsc_coverage = 1
    length = np.mean(lengths)
    # to do 
    idx_cover = np.where(coverages)[0]
    length_cover = np.mean([lengths for i in idx_cover])
    # Combine results
    out = pd.DataFrame({'Coverage': [marg_coverage], 'Conditional coverage': [wsc_coverage],
                        'Length': [length], 'Length cover': [length_cover]})
    return out


def plot_histogram(breaks, weights, S=None, fig=None, limits=None, i=0, colors=None, linestyles=None, xlim=None, filename=None):
    if colors is None:
        if limits is not None:
            colors = ['tab:blue'] * len(limits)
    if linestyles is None:
        if limits is not None:
            linestyles = ['-'] * len(limits)

    if fig is None:
        fig = plt.figure()
    plt.step(breaks, weights[i], where='pre', color='black')
    if S is not None:
        idx = S[i]
        z = np.zeros(len(breaks),)
        z[idx] = weights[i,idx]
        plt.fill_between(breaks, z, step="pre", alpha=0.4, color='gray')
    if limits is not None:
        for q_idx in range(len(limits[i])):
            q = limits[i][q_idx]
            plt.axvline(q, 0, 1, linestyle=linestyles[q_idx], color=colors[q_idx])

    plt.xlabel('$Y$')
    plt.ylabel('Density')

    if xlim is not None:
        plt.xlim(xlim)

    if filename is not None:
        fig.set_size_inches(4.5, 3)
        plt.savefig(filename, bbox_inches='tight', dpi=300)

    plt.show()
