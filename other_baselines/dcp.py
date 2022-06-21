import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats.mstats import mquantiles

from other_baselines.utils import _estim_dist


class DCP:
  """
  Method from "Distributional conformal prediction"
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
    cdf_values = np.array([0.0] * n2)
    for i in range(n2):
        cdf, inv_cdf = _estim_dist(quantiles[i], percentiles, y_min=self.ymin, y_max=self.ymax, smooth_tails=True, tau=0.01)
        cdf_values[i] = cdf(Y[i])

    scores = np.abs(np.clip(cdf_values, 0, 1)-1/2)
    #noise = np.random.uniform(low=-1e-6, high=1e-6, size=n2)
    #scores = np.abs(np.clip(scores+noise, 0, 1)-1/2)

    level_adjusted = (1.0-alpha)*(1.0+1.0/float(n2))
    self.alpha_calibrated = 0.5 - mquantiles(scores, prob=level_adjusted)[0]

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
        pred[i,0] = inv_cdf(self.alpha_calibrated)
        pred[i,1] = inv_cdf(1-self.alpha_calibrated)

    return pred