import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats.mstats import mquantiles


def find_nearest(a, a0):
    "Index of element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return idx


# CQR error function
class QR_errfun():
  """Calculates conformalized quantile regression error.
  Conformity scores:
  .. math::
  max{\hat{q}_low - y, y - \hat{q}_high}
  """
  def __init__(self):
    super(QR_errfun, self).__init__()

  def apply(self, prediction, y):
    y_lower = prediction[:,0]
    y_upper = prediction[:,-1]
    error_low = y_lower - y
    error_high = y - y_upper
    err = np.maximum(error_high,error_low)
    return err

  def apply_inverse(self, nc, alpha):
    q = np.quantile(nc, np.minimum(1.0, (1.0-alpha)*(nc.shape[0]+1.0)/nc.shape[0]))
    return np.vstack([q, q])


class CQR:
  """
  Classical CQR
  """
  def __init__(self, bbox=None):
    if bbox is not None:
      self.init_bbox(bbox)

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
    pred = self.bbox.predict(X)

    # Predict using (alpha/2, 1-alpha/2) quantiles
    quantiles = self.bbox.get_quantiles()
    idx_lower = find_nearest(quantiles, self.alpha/2.0)
    idx_upper = find_nearest(quantiles, 1.0-self.alpha/2.0)
    pred = pred[:,[idx_lower,idx_upper]]

    # Choose conformity score
    scorer = QR_errfun()
    scores = scorer.apply(pred, Y)

    # Compute correction factor based on scores
    self.score_correction = scorer.apply_inverse(scores, alpha)

    # Print message
    print("Calibrated score corrections: {:.3f}, {:.3f}". \
          format(-self.score_correction[0,0], self.score_correction[1,0]))

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
    # Predict using (alpha/2, 1-alpha/2) quantiles
    quantiles = self.bbox.get_quantiles()
    idx_lower = find_nearest(quantiles, self.alpha/2.0)
    idx_upper = find_nearest(quantiles, 1.0-self.alpha/2.0)
    pred = self.bbox.predict(X)
    pred = pred[:,[idx_lower,idx_upper]]

    # Apply correction
    pred[:,0] -= (self.score_correction[0,0])
    pred[:,1] += (self.score_correction[1,0])
    return pred

  def predict_all(self, X):
    pred = self.bbox.predict(X)
    return pred

class CQR2:
    """
    CQR with inverse quantile scores
    """
    def __init__(self, bbox=None):

        if bbox is not None:
            self.init_bbox(bbox)

    def init_bbox(self, bbox):

        self.bbox = bbox

        # Define sequence of prediction intervals for the black-box
        # e.g.
        # [0.05,0.1,0.5,0.9,0.95] -> [[0.05,0.95], [0.1,0.9]]
        # This assumes that the input quantiles are sorted
        quantiles = bbox.get_quantiles()
        assert((np.diff(quantiles)>=0).all())
        # Number of prediction intervals = 1/2 number of black-box quantiles
        num_quantiles = len(quantiles)
        num_alpha = int(np.floor(num_quantiles/2))
        assert(num_alpha>1)
        # Make list of lower and upper ends
        quantiles_idx = np.arange(num_quantiles)
        qidx_low = quantiles_idx[0:num_alpha]
        self.qidx_low = -np.sort(-qidx_low)
        self.qidx_high = quantiles_idx[(len(quantiles)-num_alpha):len(quantiles)]

    def fit(self, X, Y, bbox=None):
        # Store the black-box
        if bbox is not None:
            self.init_bbox(bbox)

        # Fit black-box model
        self.bbox.fit(X.astype(np.float32), Y.astype(np.float32))

    def calibrate(self, X_calib, Y_calib, alpha, bbox=None, return_scores=False):
        if bbox is not None:
            self.init_bbox(bbox)

        # Compute predictions on calibration data
        pred = self.bbox.predict(X_calib)

        # Extract black-box quantiles
        quantiles = self.bbox.get_quantiles()
        num_quantiles = len(quantiles)

        # Check coverage for all intervals on calibration data
        pred_low = pred[:,self.qidx_low]
        pred_high = pred[:,self.qidx_high]
        Y_c_mat = Y_calib.reshape((len(Y_calib),1))
        covered = (Y_c_mat>=pred_low) * (Y_c_mat<=pred_high)
        # Add padding to make sure coverage is always possible
        covered = np.pad(covered, ((0, 0), (1, 1)), 'constant', constant_values=(False, True))
        # Compute conformity scores on calibration data
        scores = np.argmax(covered==True, axis=1)
        # Remove padding
        scores = scores - 1
        scores[np.where(scores<0)] = 0

        # Compute upper quantile of scores
        n2 = X_calib.shape[0]
        level_adjusted = (1.0-alpha)*(1.0+1.0/float(n2))
        calibrated_idx = int(mquantiles(scores, prob=level_adjusted)[0])

        # Use the most conservative bands if everything else failed
        if calibrated_idx >= len(self.qidx_low):
            calibrated_idx = len(self.qidx_low)-1

        # Apply CQR on top of the selected bands
        self.calibrated_qidx_low = self.qidx_low[calibrated_idx]
        self.calibrated_qidx_high = self.qidx_high[calibrated_idx]
        pred_cqr = np.zeros((n2,2))
        pred_cqr[:,0] = pred[:,self.calibrated_qidx_low]
        pred_cqr[:,1] = pred[:,self.calibrated_qidx_high]

        # Choose conformity score for CQR correction
        scorer_cqr = QR_errfun()
        scores_cqr = scorer_cqr.apply(pred_cqr, Y_calib)

        # Compute correction factor based on scores
        self.cqr_correction = scorer_cqr.apply_inverse(scores_cqr, alpha).flatten()

        # Print message
        q_star_low = quantiles[self.calibrated_qidx_low]
        q_star_high = quantiles[self.calibrated_qidx_high]
        print("Calibrated quantiles (nominal level: {}): {:.3f},{:.3f}; CQR correction: {:.3f}".format(alpha, q_star_low, q_star_high, self.cqr_correction[0]))

        # Return p-value scores
        scores_out = scores
        scores_out[scores_out==len(self.qidx_low)] = len(self.qidx_low)-1
        scores_out = 1.0-2*quantiles[self.qidx_low[scores_out]]

        # Return conformity scores
        if return_scores:
            return scores_out

    def fit_calibrate(self, X, Y, alpha, random_state=2020, bbox=None,
                                        verbose=False, return_scores=False):
        # Store the black-box
        if bbox is not None:
            self.init_bbox(bbox)

        # Split data into training/calibration sets
        X_train, X_calib, Y_train, Y_calib = train_test_split(X, Y, test_size=0.5, random_state=random_state)

        # Fit black-box model
        self.fit(X_train, Y_train)

        # Calibrate
        scores = self.calibrate(X_calib, Y_calib, alpha)

        # Return conformity scores
        if return_scores:
            return scores

    def predict(self, X):
        pred = self.bbox.predict(X)
        pred_low = pred[:,self.calibrated_qidx_low] - (self.cqr_correction[0])
        pred_high = pred[:,self.calibrated_qidx_high] + (self.cqr_correction[1])

        return np.concatenate((pred_low[:,np.newaxis],pred_high[:,np.newaxis]), axis=1).squeeze()

    def predict_all(self, X):
        pred = self.bbox.predict(X)
        return pred