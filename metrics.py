# reference : https://github.com/TUW-GEO/pytesmo/blob/master/src/pytesmo/metrics/pairwise.py

import numpy as np
from scipy import stats

def bias(x, y):
    """
    Difference of the mean values.
    Sign of output depends on argument order.
    We calculate mean(x) - mean(y).

    Parameters
    ----------
    x : numpy.ndarray
        Test input vector.
    y : numpy.ndarray
        Reference input vector.

    Returns
    -------
    bias : float
        Bias between x and y.
    """
    return np.mean(x) - np.mean(y)

def bias_ci(x, y, b, alpha=0.05):
    """
    Confidence interval for bias.

    The confidence interval is the same as the confidence interval for a mean.

    Parameters
    ----------
    x : numpy.ndarray
        Test input vector.
    y : numpy.ndarray
        Reference input vector.
    b : float
        bias
    alpha : float, optional
        1 - confidence level, default is 0.05

    Returns
    -------
    lower, upper : float
        Lower and upper confidence interval bounds.
    """
    n = len(x)
    delta = np.std(x - y, ddof=1) / np.sqrt(n) * stats.t.ppf(1 - alpha / 2, n - 1)
    return b - delta, b + delta

def aad(x, y):
    """
    Average (=mean) absolute deviation (AAD).

    Parameters
    ----------
    x : numpy.ndarray
        Test input vector.
    y : numpy.ndarray
        Reference input vector.

    Returns
    -------
    d : float
        Mean absolute deviation.
    """
    return np.mean(np.abs(x - y))


def mad(x, y):
    """
    Median absolute deviation (MAD).

    Parameters
    ----------
    x : numpy.ndarray
        Test input vector.
    y : numpy.ndarray
        Reference input vector.

    Returns
    -------
    d : float
        Median absolute deviation.
    """
    return np.median(np.abs(x - y))

def RSS(x, y):
    """
    Residual sum of squares.

    Parameters
    ----------
    x : numpy.ndarray
        Test input vector.
    y : numpy.ndarray
        Reference input vector.

    Returns
    -------
    res : float
        Residual sum of squares.
    """
    return np.sum((x - y) ** 2)

def rmsd(x, y, ddof=0):
    """
    Root-mean-square deviation (RMSD). It is implemented for an unbiased
    estimator, which means the RMSD is the square root of the variance, also
    known as the standard error. The delta degree of freedom keyword (ddof) can
    be used to correct for the case the true variance is unknown and estimated
    from the population. Concretely, the naive sample variance estimator sums
    the squared deviations and divides by n, which is biased. Dividing instead
    by n -1 yields an unbiased estimator

    Parameters
    ----------
    x : numpy.ndarray
        Test input vector.
    y : numpy.ndarray
        Reference input vector.
    ddof : int, optional
        Delta degree of freedom.The divisor used in calculations is N - ddof,
        where N represents the number of elements. By default ddof is zero.

    Returns
    -------
    rmsd : float
        Root-mean-square deviation.
    """
    return np.sqrt(RSS(x, y) / (len(x) - ddof))


def nrmsd(x, y, ddof=0):
    """
    Normalized root-mean-square deviation (nRMSD).

    Parameters
    ----------
    x : numpy.ndarray
        Test input vector.
    y : numpy.ndarray
        Reference input vector.
    ddof : int, optional
        Delta degree of freedom.The divisor used in calculations is N - ddof,
        where N represents the number of elements. By default ddof is zero.

    Returns
    -------
    nrmsd : float
        Normalized root-mean-square deviation (nRMSD).
    """
    return rmsd(x, y, ddof) / (np.max([x, y]) - np.min([x, y]))


def ubrmsd(x, y, ddof=0):
    """
    Unbiased root-mean-square deviation (uRMSD).

    Parameters
    ----------
    x : numpy.ndarray
        Test input vector.
    y : numpy.ndarray
        Reference input vector.
    ddof : int, optional
        Delta degree of freedom.The divisor used in calculations is N - ddof,
        where N represents the number of elements. By default ddof is zero.

    Returns
    -------
    urmsd : float
        Unbiased root-mean-square deviation (uRMSD).
    """
    return np.sqrt(np.sum(((x - np.mean(x)) -
                           (y - np.mean(y))) ** 2) / (len(x) - ddof))

def ubrmsd_ci(x, y, ubrmsd, alpha=0.05):
    """
    Confidende interval for unbiased root-mean-square deviation (uRMSD).

    Parameters
    ----------
    x : numpy.ndarray
        Test input vector
    y : numpy.ndarray
        Reference input vector
    ubrmsd : float
        ubRMSD for this data
    alpha : float, optional
        1 - confidence level, default is 0.05

    Returns
    -------
    lower, upper : float
        Lower and upper confidence interval bounds.
    """
    n = len(x)
    ubMSD = ubrmsd ** 2
    lb_ubMSD = n * ubMSD / stats.chi2.ppf(1 - alpha / 2, n - 1)
    ub_ubMSD = n * ubMSD / stats.chi2.ppf(alpha / 2, n - 1)
    return np.sqrt(lb_ubMSD), np.sqrt(ub_ubMSD)

def mse(x, y, ddof=0):
    """
    Mean square error (MSE) as a decomposition of the RMSD into individual
    error components. The MSE is the Reference moment (about the origin) of the
    error, and thus incorporates both the variance of the estimator and
    its bias. For an unbiased estimator, the MSE is the variance of the
    estimator. Like the variance, MSE has the same units of measurement as
    the square of the quantity being estimated.
    The delta degree of freedom keyword (ddof) can be used to correct for
    the case the true variance is unknown and estimated from the population.
    Concretely, the naive sample variance estimator sums the squared deviations
    and divides by n, which is biased. Dividing instead by n - 1 yields an
    unbiased estimator.

    Parameters
    ----------
    x : numpy.ndarray
        Test input vector.
    y : numpy.ndarray
        Reference input vector.
    ddof : int, optional
        Delta degree of freedom.The divisor used in calculations is N - ddof,
        where N represents the number of elements. By default ddof is zero.

    Returns
    -------
    mse : float
        Mean square error (MSE).
    mse_corr : float
        Correlation component of MSE.
    mse_bias : float
        Bias component of the MSE.
    mse_var : float
        Variance component of the MSE.
    """
    mse_corr = 2 * np.std(x, ddof=ddof) * \
        np.std(y, ddof=ddof) * (1 - pearson_r(x, y))
    mse_bias = bias(x, y) ** 2
    mse_var = (np.std(x, ddof=ddof) - np.std(y, ddof=ddof)) ** 2
    mse = mse_corr + mse_bias + mse_var

    return mse, mse_corr, mse_bias, mse_var

def pearson_r(x, y):
    """
    Pearson's linear correlation coefficient.

    Parameters
    ----------
    x : numpy.ndarray
        Test input vector.
    y : numpy.ndarray
        Reference input vector.

    Returns
    -------
    r : float
        Pearson's correlation coefficent.

    See Also
    --------
    scipy.stats.pearsonr
    """
    return stats.pearsonr(x, y)[0]

def pearson_r_ci(x, y, r, alpha=0.05):
    """
    Confidence interval for Pearson correlation coefficient.

    Parameters
    ----------
    x : numpy.ndarray
        Test input vector
    y : numpy.ndarray
        Reference input vector
    r : float
        Pearson r for this data
    alpha : float, optional
        1 - confidence level, default is 0.05

    Returns
    -------
    lower, upper : float
        Lower and upper confidence interval bounds.

    References
    ----------
    Bonett, D. G., & Wright, T. A. (2000). Sample size requirements for
    estimating Pearson, Kendall and Spearman correlations. Psychometrika,
    65(1), 23-28.
    """
    n = len(x)
    v = np.arctanh(r)
    z = stats.norm.ppf(1 - alpha / 2)
    cl = v - z / np.sqrt(n - 3)
    cu = v + z / np.sqrt(n - 3)
    return np.tanh(cl), np.tanh(cu)

def pearsonr(x, y):
    """
    Calculate the Pearson correlation coefficient between two arrays.

    The Pearson correlation coefficient measures the linear relationship
    between two datasets. Strictly speaking, Pearson's correlation requires
    that each dataset be normally distributed. Like other correlation coefficients,
    this one varies between -1 and +1 with 0 implying no correlation. Correlations of
    -1 or +1 imply an exact linear relationship. Positive correlations imply that
    as x increases, so does y. Negative correlations imply that as x increases,
    y decreases.

    Parameters
    ----------
    x : numpy.ndarray
        Test input vector.
    y : numpy.ndarray
        Reference input vector.

    Returns
    -------
    r : float
        Pearson correlation coefficient. If the calculation cannot be performed
        (e.g., due to division by zero or constant vectors), the function returns 0.

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([5, 6, 7, 8, 7])
    >>> pearson_r(x, y)
    0.9

    Notes
    -----
    The Pearson correlation coefficient is calculated as:
        r = sum((x - mean(x)) * (y - mean(y))) / sqrt(sum((x - mean(x))**2) * sum((y - mean(y))**2))
    """
    # Ensure x and y are numpy arrays
    x = np.array(x)
    y = np.array(y)
    
    # Compute the mean of the arrays
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # Subtract means
    x_diff = x - mean_x
    y_diff = y - mean_y
    
    # Compute Pearson correlation
    numerator = np.sum(x_diff * y_diff)
    denominator = np.sqrt(np.sum(x_diff**2) * np.sum(y_diff**2))
    
    # Handle division by zero
    if denominator == 0:
        return 0  # or np.nan to indicate an undefined result
    
    r = numerator / denominator
    return r

import numpy as np

def rank_simple(vector):
    """
    Assign ranks to the elements of an array, handling ties by assigning the average of the ranks.
    This method works on a sorted copy of the original vector to determine ranks, then applies these
    ranks back to the original order of the vector.
    """
    sorted_indices = np.argsort(vector)
    ranks = np.zeros(len(vector))
    current_rank = 1
    for i in range(len(vector)):
        ranks[sorted_indices[i]] = current_rank
        current_rank += 1

    # Handle ties: average the ranks
    for i in range(len(vector)):
        tie_indices = np.where(vector == vector[i])[0]
        if len(tie_indices) > 1:
            average_rank = np.mean([ranks[index] for index in tie_indices])
            for index in tie_indices:
                ranks[index] = average_rank

    return ranks

def spearmanr(x, y):
    """
    Calculate the Spearman rank correlation coefficient for two arrays.

    Spearman's rho measures the strength and direction of association between two ranked variables.
    It assesses how well the relationship between two variables can be described using a monotonic function.

    Parameters
    ----------
    x : numpy.ndarray
        A 1D array containing data for the first variable.
    y : numpy.ndarray
        A 1D array containing data for the Reference variable.

    Returns
    -------
    rho : float
        The Spearman rank correlation coefficient.

    Notes
    -----
    Spearman's rho is calculated by ranking the data and then applying the Pearson correlation coefficient
    formula to these ranks. This method directly calculates ranks and accounts for ties by averaging the ranks
    of tied values. It evaluates the correlation of the ranks and thus differs from Pearson's, which evaluates
    the correlation of actual values.
    """
    # Rank the data
    rank_x = rank_simple(x)
    rank_y = rank_simple(y)
    
    # Calculate Pearson correlation on ranks
    rho = pearsonr(rank_x, rank_y)
    return rho


def spearman_r(x, y):
    """
    Spearman's rank correlation coefficient.

    Parameters
    ----------
    x : numpy.array
        Test input vector.
    y : numpy.array
        Reference input vector.

    Returns
    -------
    rho : float
        Spearman correlation coefficient

    See Also
    --------
    scipy.stats.spearmenr
    """
    return stats.spearmanr(x, y)[0]


def spearman_r_ci(x, y, r, alpha=0.05):
    """
    Confidence interval for Spearman rank correlation coefficient.

    Parameters
    ----------
    x : numpy.ndarray
        Test input vector
    y : numpy.ndarray
        Reference input vector
    r : float
        Spearman's r for this data
    alpha : float, optional
        1 - confidence level, default is 0.05

    Returns
    -------
    lower, upper : float
        Lower and upper confidence interval bounds.

    References
    ----------
    Bonett, D. G., & Wright, T. A. (2000). Sample size requirements for
    estimating Pearson, Kendall and Spearman correlations. Psychometrika,
    65(1), 23-28.
    """
    n = len(x)
    v = np.arctanh(r)
    z = stats.norm.ppf(1 - alpha / 2)
    # see reference for this formula
    cl = v - z * np.sqrt(1 + r ** 2 / 2) / np.sqrt(n - 3)
    cu = v + z * np.sqrt(1 + r ** 2 / 2) / np.sqrt(n - 3)
    return np.tanh(cl), np.tanh(cu)


def calculate_iqr(data):
    """
    Calculate the Interquartile Range (IQR) of the given data.

    Parameters
    ----------
    data : array-like
        Input data from which to calculate the IQR.

    Returns
    -------
    float
        The interquartile range of the data.
    """
    y = [x for x in data if not np.isnan(x)]
    return stats.iqr(y)

def fisher_z(r):
    """
    Convert an array of Pearson correlation coefficients to Fisher z-values.

    Parameters
    ----------
    r : array-like
        List or array of Pearson correlation coefficients. Values should be between -1 and 1, exclusive.

    Returns
    -------
    array-like
        The Fisher z-transformed values of the input correlation coefficients.

    Notes
    -----
    This function adjusts correlation coefficients exactly equal to 1 or -1 to 0.999999 and -0.999999 respectively
    to avoid mathematical errors in transformations.
    """
    # r = np.clip(r, -0.999999, 0.999999)
    # Adjust values exactly equal to 1.0 or -1.0
    r = [0.999999 if x == 1.0 else -0.999999 if x == -1.0 else x for x in r]
    return np.arctanh(r)

def inverse_fisher_z(z):
    """
    Convert an array of Fisher z-values back to Pearson correlation coefficients.

    Parameters
    ----------
    z : array-like
        List or array of Fisher z-values.

    Returns
    -------
    array-like
        The Pearson correlation coefficients corresponding to the input Fisher z-values.
    """
    return np.tanh(z)

def mean_r(correlations):
    """
    Calculate the mean of an array of Pearson correlation coefficients using Fisher z transformation.

    Parameters
    ----------
    correlations : array-like
        List or array of Pearson correlation coefficients.

    Returns
    -------
    float
        The mean Pearson correlation coefficient.

    Notes
    -----
    This function uses the Fisher z transformation to compute the mean of the correlation coefficients
    to improve the accuracy of the mean calculation for normally distributed values.
    """

    # Convert Pearson r values to Fisher z values
    z_values = fisher_z(np.array(correlations))
    
    # Calculate the mean of the Fisher z values
    mean_z = np.nanmean(z_values)
    
    # Convert the mean Fisher z value back to a Pearson r value
    mean_r = inverse_fisher_z(mean_z)
    
    return mean_r
