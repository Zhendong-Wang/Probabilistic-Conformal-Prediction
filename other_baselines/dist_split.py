import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats.mstats import mquantiles

from other_baselines.utils import _estim_dist


class DistSplit:
  """
  Method from "Flexible distribution-free conditional predictive bandsusing density estimators"
  """

  def __init__(self, bbox=None, ymin=-1, ymax=1):
    if bbox is not None:
      self.init_bbox(bbox)
    self.ymin = ymin
    self.ymax = ymax

  def init_bbox(self, bbox):
    self.bbox = bbox

  def fit(self, X, Y):
    # Fit black-box model
    self.bbox.fit(X, Y)

  def calibrate(self, X, Y, alpha, bbox=None, return_scores=False):
    self.alpha = alpha

    if bbox is not None:
      # Store the pre-trained black-box
      self.init_bbox(bbox)

    # Compute predictions on calibration data
    n2 = X.shape[0]
    quantiles = self.bbox.predict(X)
    percentiles = self.bbox.get_quantiles()

    # Compute conformity scores
    scores = np.array([0.0] * n2)
    for i in range(n2):
        cdf, inv_cdf = _estim_dist(quantiles[i], percentiles, y_min=self.ymin, y_max=self.ymax, smooth_tails=True, tau=0.01)
        scores[i] = cdf(Y[i])

    # Compute upper and lower quantiles of conformity scores
    alpha_adjusted = 1-(1-alpha)*(1.0+1.0/float(n2))
    self.t_lo = mquantiles(scores, prob=alpha_adjusted/2)[0]
    self.t_up = mquantiles(scores, prob=1.0-alpha_adjusted/2)[0]

    # def preview_coverage(t_lo, t_up):
    #     quantiles = self.bbox.predict(X)
    #     percentiles = self.bbox.get_quantiles()

    #     n = X.shape[0]
    #     pred = np.zeros((n,2))
    #     for i in range(n):
    #         cdf, inv_cdf = _estim_dist(quantiles[i], percentiles, y_min=self.ymin, y_max=self.ymax, smooth_tails=False, tau=0.01)
    #         pred[i,0] = inv_cdf(t_lo)
    #         pred[i,1] = inv_cdf(t_up)

    #     return np.mean((Y>=pred[:,0])*(Y<=pred[:,1]))

  def fit_calibrate(self, X, Y, alpha, bbox=None, random_state=2020, verbose=False):
    self.alpha = alpha
    if bbox is not None:
      # Store the pre-trained black-box
      self.init_bbox(bbox)

    # Split data into training/calibration sets
    X_train, X_calib, Y_train, Y_calib = train_test_split(X, Y, test_size=0.5, random_state=random_state)

    # Fit black-box model
    self.fit(X_train, Y_train)

    # Calibrate
    self.calibrate(X_calib, Y_calib, alpha)

  def predict(self, X):

    quantiles = self.bbox.predict(X)
    percentiles = self.bbox.get_quantiles()

    n = X.shape[0]
    pred = np.zeros((n,2))
    for i in range(n):
        cdf, inv_cdf = _estim_dist(quantiles[i], percentiles, y_min=self.ymin, y_max=self.ymax, smooth_tails=True, tau=0.01)
        pred[i,0] = inv_cdf(self.t_lo)
        pred[i,1] = inv_cdf(self.t_up)

    return pred