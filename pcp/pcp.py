import torch
import numpy as np
from sympy import Interval, Union
from tqdm.autonotebook import tqdm


def union(data):
    """ Union of a list of intervals e.g. [(1,2),(3,4)] """
    intervals = [Interval(begin, end) for (begin, end) in data]
    u = Union(*intervals)
    if isinstance(u, Interval):
        return [list(u.args[:2])]
    else:
        result = [list(v.args[:2]) for v in list(u.args)]
        return result


class PCP(object):
    def __init__(self, model, base='torch', device="cpu", alpha=0.1, sample_K=40, cal_type='uniform', fr=0.2):
        self.model = model
        self.base = base
        self.device = device
        self.alpha = alpha
        self.sample_K = sample_K
        self.cal_type = cal_type
        self.fr = fr

    def calibrate(self, X, Y, alpha=None):
        X_numpy, X, Y = X, torch.from_numpy(X), torch.from_numpy(Y.reshape(-1, 1))
        if self.cal_type == 'uniform':
            preds = []
            for j in range(self.sample_K):
                if self.base == 'torch':
                    y_sample = self.model.sample(X.to(self.device)).detach().cpu()
                else:
                    y_sample = torch.from_numpy(self.model.sample(X_numpy)[1].reshape(-1, 1))
                preds.append(y_sample)
            preds = torch.hstack(preds)  # preds shape: [nsamples, sample_K]
            score = torch.min(torch.abs(preds - Y), dim=1)[0]
        elif self.cal_type == 'filtered':
            preds = []
            densities = []
            for j in range(self.sample_K):
                y_sample = torch.from_numpy(self.model.sample(X_numpy)[1].reshape(-1, 1))
                preds.append(y_sample)
                densities.append(torch.from_numpy(self.model.pdf(X_numpy, y_sample).reshape(-1, 1)))
            preds = torch.hstack(preds)  # preds shape: [nsamples, sample_K]
            densities = torch.hstack(densities)  # densities shape: [nsamples, sample_K]
            densities_ind = densities.argsort()
            preds = torch.gather(preds, 1, densities_ind)[:, int(self.sample_K * self.fr):]

            score = torch.min(torch.abs(preds - Y), dim=1)[0]

        self.qt = qt = torch.quantile(score, 1 - self.alpha).numpy()
        return qt

    def predict(self, X):
        x_te = torch.from_numpy(X)
        if self.cal_type == 'uniform':
            preds_te = []
            for j in range(self.sample_K):
                if self.base == 'torch':
                    y_sample = self.model.sample(x_te.to(self.device)).detach().cpu()
                else:
                    y_sample = torch.from_numpy(self.model.sample(X)[1].reshape(-1, 1))
                preds_te.append(y_sample)
            preds_te = torch.hstack(preds_te).numpy()

            bands = []
            for pred_te in tqdm(preds_te):
                interval = union([(s - self.qt, s + self.qt) for s in pred_te])
                bands.append(interval)
        elif self.cal_type == 'filtered':  # only valid for density available models.
            preds_te = []
            densities = []
            for j in range(self.sample_K):
                y_sample = torch.from_numpy(self.model.sample(X)[1].reshape(-1, 1))
                preds_te.append(y_sample)
                densities.append(torch.from_numpy(self.model.pdf(X, y_sample)).reshape(-1, 1))

            preds_te = torch.hstack(preds_te)
            densities = torch.hstack(densities)
            densities_ind = densities.argsort()
            preds_te = torch.gather(preds_te, 1, densities_ind)[:, int(self.sample_K * self.fr):].numpy()

            bands = []
            for pred_te in tqdm(preds_te):
                interval = union([(s - self.qt, s + self.qt) for s in pred_te])
                bands.append(interval)

        return bands

    def calibrate_md(self, X, Y, caltype = 'uniform', fr=0):
        X_numpy, X, Y = X, torch.from_numpy(X), torch.from_numpy(Y)
        preds = []
        density = []
        if caltype == 'filtered':
            sample_K = int(self.sample_K / (1-fr))
        else:
            sample_K = self.sample_K
        for j in range(sample_K):
            if self.base == 'torch':
                y_sample = self.model.sample(X.to(self.device)).detach().cpu()
            else:
                y_sample = torch.from_numpy(self.model.sample(X_numpy)[1])
                if caltype == 'density' or caltype == 'filtered':
                    density_sample = torch.from_numpy(self.model.pdf(X_numpy, y_sample.numpy()))
                    density.append(density_sample)
            preds.append(y_sample)
        preds = torch.stack(preds, dim = 1)
        # change to L2
        from scipy.spatial.distance import mahalanobis
        s = preds - Y.reshape(Y.shape[0], 1, Y.shape[1])

        if caltype == 'uniform':
            score = torch.min(torch.sum(s ** 2, 2) , 1)[0]
        elif caltype == 'filtered': # HD-PCP
            density = torch.stack(density, dim = 1)
            _, indices = torch.topk(density, int((1-fr)*preds.shape[1]),dim = 1)
            indices = indices.unsqueeze(-1).expand(*(-1,)*density.ndim, 2)
            preds = torch.gather(preds, 1, indices)
            s = preds - Y.reshape(Y.shape[0], 1, Y.shape[1])
            score = torch.min(torch.sum(s**2, 2), 1)[0]
            print(s.shape)
        self.qt = qt = torch.quantile(score, 1 - self.alpha).numpy()
        return qt

    def predict_md(self, X, caltype = 'uniform', fr=0):
        import tensorflow as tf
        x_te = torch.from_numpy(X)
        preds_te = []
        density_te = []
        if caltype == 'filtered':
            sample_K = int(self.sample_K / (1-fr))
        else:
            sample_K = self.sample_K
        for j in range(sample_K):
            if self.base == 'torch':
                y_sample = self.model.sample(x_te.to(self.device)).detach().cpu()
            else:
                y_sample = torch.from_numpy(self.model.sample(X)[1])
                if caltype == 'filtered':
                    density_sample = torch.from_numpy(self.model.pdf(x_te, y_sample.numpy()))
                    density_te.append(density_sample)
            preds_te.append(y_sample)
        preds_te = torch.stack(preds_te, dim = 1)
        if caltype == 'uniform':
            return preds_te, None
        elif caltype == 'filtered': # HD-PCP
            density_te = torch.stack(density_te, dim = 1)
            _, indices = torch.topk(density_te, int((1-fr)*preds_te.shape[1]),dim = 1)
            indices = indices.unsqueeze(-1).expand(*(-1,)*density_te.ndim, 2)
            preds_te = torch.gather(preds_te, 1, indices)
            return preds_te, None




