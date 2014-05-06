"""Statistics routines."""

import numpy as np
import scipy.signal as signal
import scipy.stats as stats

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  angle_wrap
#--------------------------------------------------------------------------------------------------

def angle_wrap(angle, period=2.0 * np.pi):
    """Wrap angle into interval centred on zero.

    This wraps the *angle* into the interval -*period* / 2 ... *period* / 2.

    """
    return (angle + 0.5 * period) % period - 0.5 * period

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  minimise_angle_wrap
#--------------------------------------------------------------------------------------------------

def minimise_angle_wrap(angles, axis=0):
    """Minimise wrapping of angles to improve interpretation.

    Move wrapping point as far away as possible from mean angle on given axis.
    The main use of this function is to improve the appearance of angle plots.

    Parameters
    ----------
    angles : array-like
        Array of angles to unwrap, in radians
    axis : int, optional
        Axis along which angle wrap is evaluated. Plots along this axis will
        typically improve in appearance.

    Returns
    -------
    angles : array
        Array of same shape as input array, with angles wrapped around new point

    """
    angles = np.asarray(angles)
    # Calculate a "safe" mean on the unit circle
    mu = np.arctan2(np.sin(angles).mean(axis=axis), np.cos(angles).mean(axis=axis))
    # Wrap angle differences into interval -pi ... pi
    delta_ang = angle_wrap(angles - np.expand_dims(mu, axis))
    return delta_ang + np.expand_dims(mu, axis)

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  mu_sigma
#--------------------------------------------------------------------------------------------------

def mu_sigma(data, axis=0):
    """Determine second-order statistics from data.

    Convenience function to return second-order statistics of data along given
    axis.

    Parameters
    ----------
    data : array-like
        Numpy data array (or equivalent) of arbitrary shape
    axis : int, optional
        Index of axis along which stats are calculated (will be averaged away
        in the process)

    Returns
    -------
    mu, sigma : array
        Mean and standard deviation as arrays of same shape as *data*, but
        without given axis

    """
    return np.mean(data, axis=axis), np.std(data, axis=axis)

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  robust_mu_sigma
#--------------------------------------------------------------------------------------------------

def robust_mu_sigma(data, axis=0):
    """Determine second-order statistics from data, using robust statistics.

    Convenience function to return second-order statistics of data along given
    axis. These are determined via the median and interquartile range.

    Parameters
    ----------
    data : array-like
        Numpy data array (or equivalent) of arbitrary shape
    axis : int, optional
        Index of axis along which stats are calculated (will be averaged away
        in the process)

    Returns
    -------
    mu, sigma : array
        Mean and standard deviation as arrays of same shape as *data*, but
        without given axis

    """
    data = np.asarray(data)
    # Create sequence of axis indices with specified axis at the front, and the rest following it
    move_axis_to_front = range(len(data.shape))
    move_axis_to_front.remove(axis)
    move_axis_to_front = [axis] + move_axis_to_front
    # Create copy of data sorted along specified axis, and reshape so that the specified axis becomes the first one
    sorted_data = np.sort(data, axis=axis).transpose(move_axis_to_front)
    # Obtain quartiles
    perc25 = sorted_data[int(0.25 * len(sorted_data))]
    perc50 = sorted_data[int(0.50 * len(sorted_data))]
    perc75 = sorted_data[int(0.75 * len(sorted_data))]
    # Conversion factor from interquartile range to standard deviation (based on normal pdf)
    iqr_to_std = 0.741301109253
    return perc50, iqr_to_std * (perc75 - perc25)

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  periodic_mu_sigma
#--------------------------------------------------------------------------------------------------

def periodic_mu_sigma(data, axis=0, period=2.0 * np.pi):
    """Determine second-order statistics of periodic (angular, directional) data.

    Convenience function to return second-order statistics of data along given
    axis. This handles periodic variables, which exhibit the problem of
    wrap-around and therefore are unsuited for the normal mu_sigma function. The
    period with which the values repeat can be explicitly specified, otherwise
    the data is assumed to be radians. The mean is in the range -period/2 ...
    period/2, and the maximum standard deviation is about period/4.

    Parameters
    ----------
    data : array-like
        Numpy array (or equivalent) of arbitrary shape, containing angles
    axis : int, optional
        Index of axis along which stats are calculated (will be averaged away
        in the process)
    period : float, optional
        Period with which data values repeat

    Returns
    -------
    mu, sigma : array
        Mean and standard deviation as arrays of same shape as *data*, but
        without given axis

    Notes
    -----
    The approach in [1]_ is used.

    .. [1] R. J. Yamartino, "A Comparison of Several 'Single-Pass' Estimators
       of the Standard Deviation of Wind Direction," Journal of Climate and
       Applied Meteorology, vol. 23, pp. 1362-1366, 1984.

    """
    data = np.asarray(data, dtype='double')
    # Create sequence of axis indices with specified axis at the front, and the rest following it
    move_axis_to_front = range(len(data.shape))
    move_axis_to_front.remove(axis)
    move_axis_to_front = [axis] + move_axis_to_front
    # Create copy of data, and reshape so that the specified axis becomes the first one
    data = data.copy().transpose(move_axis_to_front)
    # Scale data so that one period becomes 2*pi, the natural period for angles
    scale = 2.0 * np.pi / period
    data *= scale
    # Calculate a "safe" mean on the unit circle
    mu = np.arctan2(np.sin(data).mean(axis=0), np.cos(data).mean(axis=0))
    # Wrap angle differences into interval -pi ... pi
    delta_ang = angle_wrap(data - mu)
    # Calculate variance using standard formula with a second correction term
    sigma2 = (delta_ang ** 2.0).mean(axis=0) - (delta_ang.mean(axis=0) ** 2.0)
    # Scale answers back to original data range
    return mu / scale, np.sqrt(sigma2) / scale

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  remove_spikes
#--------------------------------------------------------------------------------------------------

def remove_spikes(data, axis=0, spike_width=3, outlier_sigma=5.0):
    """Remove outliers from data, replacing them with a local median value.

    The data is median-filtered along the specified axis, and any data values
    that deviate significantly from the local median is replaced with the median.

    Parameters
    ----------
    data : array-like
        N-dimensional numpy array containing data to clean
    axis : int, optional
        Axis along which to perform median, between 0 and N-1
    spike_width : int, optional
        Spikes with widths up to this limit (in samples) will be removed. A size
        of <= 0 implies no spike removal. The kernel size for the median filter
        will be 2 * spike_width + 1.
    outlier_sigma : float, optional
        Multiple of standard deviation that indicates an outlier

    Returns
    -------
    cleaned_data : array
        N-dimensional numpy array of same shape as original data, with outliers
        removed

    Notes
    -----
    This is very similar to a *Hampel filter*, also known as a *decision-based
    filter* or three-sigma edit rule combined with a Hampel outlier identifier.

    .. todo::

       TODO: Make this more like a Hampel filter by making MAD time-variable too.

    """
    data = np.atleast_1d(data)
    kernel_size = 2 * max(int(spike_width), 0) + 1
    if kernel_size == 1:
        return data
    # Median filter data along the desired axis, with given kernel size
    kernel = np.ones(data.ndim, dtype='int32')
    kernel[axis] = kernel_size
    # Medfilt now seems to upcast 32-bit floats to doubles - convert it back to floats...
    filtered_data = np.asarray(signal.medfilt(data, kernel), data.dtype)
    # The deviation is measured relative to the local median in the signal
    abs_dev = np.abs(data - filtered_data)
    # Calculate median absolute deviation (MAD)
    med_abs_dev = np.expand_dims(np.median(abs_dev, axis), axis)
#    med_abs_dev = signal.medfilt(abs_dev, kernel)
    # Assuming normally distributed deviations, this is a robust estimator of the standard deviation
    estm_stdev = 1.4826 * med_abs_dev
    # Identify outliers (again based on normal assumption), and replace them with local median
    outliers = (abs_dev > outlier_sigma * estm_stdev)
    cleaned_data = data.copy()
    cleaned_data[outliers] = filtered_data[outliers]
    return cleaned_data

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  chi2_conf_interval
#--------------------------------------------------------------------------------------------------

def chi2_conf_interval(dof, mean=1.0, sigma=3.0):
    """Confidence interval for chi-square distribution.

    Return lower and upper limit of confidence interval of chi-square
    distribution, defined in terms of a normal confidence interval. That is,
    given *sigma*, which is a multiple of the standard deviation, calculate the
    probability mass within the interval [-sigma, sigma] for a standard normal
    distribution, and return the interval with the same probability mass for the
    chi-square distribution with *dof* degrees of freedom. The interval is
    scaled by ``(mean/dof)``, which enforces the given mean and implies a
    standard deviation of ``mean*sqrt(2/dof)``. This represents the distribution
    of the power estimator $P = 1/N \sum_{i=1}^{N} x_i^2$, with N = *dof* and
    zero-mean Gaussian voltages $x_i$ with variance *mean*.

    Parameters
    ----------
    dof : array-like or float
        Degrees of freedom (number of independent samples summed to form chi^2
        variable)
    mean : array-like or float, optional
        Desired mean of chi^2 distribution
    sigma : array-like or float, optional
        Multiple of standard deviation, used to specify size of required
        confidence interval

    Returns
    -------
    lower : array or float
        Lower limit of confidence interval (numpy array if any input is one)
    upper : array or float
        Upper limit of confidence interval (numpy array if any input is one)

    Notes
    -----
    The advantage of this approach is that it uses a well-known concept to
    specify the interval (multiples of standard deviation), while returning
    valid intervals for all values of *dof*. For (very) large values of *dof*,
    (lower, upper) will be close to

    (mean - sigma * mean*sqrt(2/dof), mean + sigma * mean*sqrt(2/dof)),

    as the chi-square distribution will be approximately normal. For small *dof*
    or large *sigma*, however, this approximation breaks down and may lead to
    negative lower values, for example.

    """
    if not np.isscalar(dof):
        dof = np.atleast_1d(np.asarray(dof))
    if not np.isscalar(mean):
        mean = np.atleast_1d(np.asarray(mean))
    if not np.isscalar(sigma):
        sigma = np.atleast_1d(np.asarray(sigma))
    # Ensure degrees of freedom is positive integer >= 1
    dof = np.array(np.clip(np.floor(dof), 1.0, np.inf), dtype=np.int)
    chi2_rv = stats.chi2(dof)
    normal_rv = stats.norm()
    # Translate normal conf interval to chi^2 distribution, maintaining the probability inside interval
    lower = chi2_rv.ppf(normal_rv.cdf(-sigma)) * mean / dof
    upper = chi2_rv.ppf(normal_rv.cdf(sigma)) * mean / dof
    return lower, upper

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  ratio_stats
#--------------------------------------------------------------------------------------------------

def ratio_stats(mean_num, std_num, mean_den, std_den, corrcoef=0, method='F'):
    """Approximate 2nd-order statistics of ratio of correlated normal variables.

    Given a normally distributed numerator *num* and denominator *den* with the
    specified means, standard deviations and correlation coefficient, this
    approximates the mean and standard deviation of the ratio *num* / *den*.
    The function is vectorised, accepting arrays of matching size. Corner cases
    are not checked yet, leading to division by zero...

    Parameters
    ----------
    mean_num, std_num: float or array
        Mean and standard deviation of numerator (std_num must be > 0)
    mean_den, std_den: float or array
        Mean and standard deviation of denominator (std_den must be > 0)
    corrcoef : float or array, optional
        Correlation coefficient of numerator and denominator (must be between
        -1 and 1, but not exactly 1 or -1)
    method : {'F', 'Marsaglia'}, optional
        Approximation technique to use ('Marsaglia' is taken from [1]_, while
        'F' seems to work better on a larger parameter space)
    Returns
    -------
    mean_ratio, std_ratio : float or array
        Approximate mean and standard deviation of ratio *num* / *den*

    Notes
    -----
    This is inspired by the "practical" mean and standard deviation of a ratio
    of correlated normal variables, as proposed in [1]_. The reference also
    discusses the assumptions behind this approximation. Note that the ratio of
    two normal variables is not normal itself, but may be approximately normal
    for specific ranges of parameters. The method in [1]_ seems to break down
    for large *mean_den*/*std_den* ratios, however, which is the most useful
    regime for :mod:`scape`. The ratio is therefore modelled as an F-distributed
    random variable instead, by default.

    .. [1] G. Marsaglia, "Ratios of Normal Variables," Journal of Statistical
       Software, vol. 16, no. 4, pp. 1-10, 2006.

    """
    # This follows the notation in [1]
    # Transform num/den to standard form (a + x) / (b + y), with x and y uncorrelated standard normal vars
    # Then num/den is distributed as 1/r (a + x) / (b + y) + s
    # Therefore determine parameters a and b, scale r and translation s
    s = corrcoef * std_num / std_den
    b = mean_den / std_den
    a = mean_num - s * mean_den
    # Pick the sign of h so that a and b have the same sign
    sign_h = 2 * (a >= 0) * (b >= 0) - 1
    h = sign_h * std_num * np.sqrt(1. - corrcoef ** 2)
    a /= h
    r = std_den / h
    if method == 'Marsaglia':
        # Calculate the approximating or "practical" mean and variance of (a + x) / (b + y) a la Marsaglia
        mean_axby = a / (1.01 * b - .2713)
        var_axby = (a**2 + 1) / (b**2 + .108 * b - 3.795) - mean_axby**2
        std_axby = np.sqrt(var_axby)
    else:
        # Calculate the approximate mean and standard deviation of (a + x) / (b + y) a la F-distribution
        mean_axby = a * b / (b**2 - 1)
        std_axby = np.abs(b) / (b**2 - 1) * np.sqrt((a**2 + b**2 - 1) / (b**2 - 2))
    # Translate by s and scale by r
    return s + mean_axby / r, std_axby / np.abs(r)

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  identify_rfi_channels
#--------------------------------------------------------------------------------------------------

def identify_rfi_channels(data, sigma=8.0, min_bad_scans=0.25, extra_outputs=False):
    """Identify potential RFI-corrupted channels.

    This is a simple RFI detection procedure that assumes that there are less
    RFI-corrupted channels than good channels, and that the desired signal is
    broadband/continuum with similar features across the entire spectrum.

    Parameters
    ----------
    data : :class:`DataSet`, :class:`CompoundScan` or :class:`Scan` object (list)
        Scans to check (entire data set, compound scan, scan or scan list)
    sigma : float, optional
        Threshold for deviation from signal (non-RFI) template, as a factor
        of the expected standard deviation of the error under the null
        hypothesis. By increasing it, less channels are marked as bad. This
        factor should typically be larger than expected (in the order of 6
        to 8), as the null hypothesis is quite stringent.
    min_bad_scans : float, optional
        The fraction of scans in which a channel has to be marked as bad in
        order to qualify as RFI-corrupted. This allows for some intermittence
        in RFI corruption.
    extra_outputs : {False, True}, optional
        True if extra information should be returned (intended for plots)

    Returns
    -------
    rfi_channels : list of ints
        List of potential RFI-corrupted channel indices

    """
    # Create list of scans from whatever input form the data takes (DataSet and CompoundScan have .scans)
    scans = data if isinstance(data, list) else getattr(data, 'scans', [data])
    if len(scans) == 0:
        raise ValueError('No scans found to analyse')
    dataset = scans[0].compscan.dataset

    rfi_count = np.zeros(len(dataset.freqs))
    dof = 4.0 * (dataset.bandwidths * 1e6) / dataset.dump_rate
    rfi_data = []
    for scan in scans:
        # Normalise power in scan by removing spikes, offsets and differences in scale
        power = remove_spikes(scan.pol('I'))
        offset = power.min(axis=0)
        scale = power.max(axis=0) - offset
        scale[scale <= 0.0] = 1.0
        norm_power = (power - offset[np.newaxis, :]) / scale[np.newaxis, :]
        # Form a template of the desired signal as a function of time
        template = np.median(norm_power, axis=1)
        # Use this as average power, after adding back scaling and offset
        mean_signal_power = np.outer(template, scale) + offset[np.newaxis, :]
        # Determine expected standard deviation of power data, assuming it has chi-square distribution
        # Also divide by an extra sqrt(template) factor, which allows more leeway where template is small
        # This is useful for absorbing small discrepancies in baseline when scanning across a source
        expected_std = np.sqrt(2.0 / dof[np.newaxis, :]) * mean_signal_power / scale[np.newaxis, :] / \
                       np.sqrt(template[:, np.newaxis])
        channel_sumsq = (((norm_power - template[:, np.newaxis]) / expected_std) ** 2).sum(axis=0)
        # The sum of squares over time is again a chi-square distribution, with different dof
        lower, upper = chi2_conf_interval(power.shape[0], power.shape[0], sigma)
        rfi_count += (channel_sumsq < lower) | (channel_sumsq > upper)
        if extra_outputs:
            rfi_data.append((norm_power, template, expected_std))
    # Count the number of bad scans per channel, and threshold it
    rfi_channels = (rfi_count > max(min_bad_scans * len(scans), 1.0)).nonzero()[0]
    if extra_outputs:
        return rfi_channels, rfi_count, rfi_data
    else:
        return rfi_channels
