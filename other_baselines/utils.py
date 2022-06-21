import numpy as np
from scipy import interpolate


def _estim_dist(quantiles, percentiles, y_min, y_max, smooth_tails, tau):
    """ Estimate CDF from list of quantiles, with smoothing """

    noise = np.random.uniform(low=0.0, high=1e-5, size=((len(quantiles),)))
    noise_monotone = np.sort(noise)
    quantiles = quantiles + noise_monotone

    # Smooth tails
    def interp1d(x, y, a, b):
        return interpolate.interp1d(x, y, bounds_error=False, fill_value=(a, b), assume_sorted=True)

    cdf = interp1d(quantiles, percentiles, 0.0, 1.0)
    inv_cdf = interp1d(percentiles, quantiles, y_min, y_max)

    if smooth_tails:
        # Uniform smoothing of tails
        quantiles_smooth = quantiles
        tau_lo = tau
        tau_hi = 1-tau
        q_lo = inv_cdf(tau_lo)
        q_hi = inv_cdf(tau_hi)
        idx_lo = np.where(percentiles < tau_lo)[0]
        idx_hi = np.where(percentiles > tau_hi)[0]
        if len(idx_lo) > 0:
            quantiles_smooth[idx_lo] = np.linspace(quantiles[0], q_lo, num=len(idx_lo))
        if len(idx_hi) > 0:
            quantiles_smooth[idx_hi] = np.linspace(q_hi, quantiles[-1], num=len(idx_hi))

        cdf = interp1d(quantiles_smooth, percentiles, 0.0, 1.0)
        inv_cdf = interp1d(percentiles, quantiles_smooth, y_min, y_max)

    # Standardize
    breaks = np.linspace(y_min, y_max, num=1000, endpoint=True)
    cdf_hat = cdf(breaks)
    f_hat = np.diff(cdf_hat)
    f_hat = (f_hat+1e-6) / (np.sum(f_hat+1e-6))
    cdf_hat = np.concatenate([[0],np.cumsum(f_hat)])
    cdf = interp1d(breaks, cdf_hat, 0.0, 1.0)
    inv_cdf = interp1d(cdf_hat, breaks, y_min, y_max)

    return cdf, inv_cdf