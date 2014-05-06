"""Gain calibration via noise injection."""

import numpy as np
import re

import fitting
from fitting import Spline1DFit, Polynomial1DFit, Spline2DGridFit
from fitting import randomise as fitting_randomise
from stats import robust_mu_sigma
from scan import scape_pol_sd

def load_csv_with_header(csv_file):
    """Load CSV file containing commented-out header with key-value pairs.

    Parameters
    ----------
    csv_file : file object or string
        File object of opened CSV file, or string containing the file name

    Returns
    -------
    csv : array, shape (N, M)
        CSV data as a 2-dimensional array with N rows and M columns
    attrs : dict
        Key-value pairs extracted from header

    """
    csv_file = file(csv_file) if isinstance(csv_file, basestring) else csv_file
    start = csv_file.tell()
    csv = np.loadtxt(csv_file, comments='#', delimiter=',')
    csv_file.seek(start)
    header = [line[1:].strip() for line in csv_file.readlines() if line[0] == '#']
    keyvalue = re.compile('\A([a-z]\w*)\s*[:=]\s*(.+)')
    attrs = dict([keyvalue.match(line).groups() for line in header if keyvalue.match(line)])
    return csv, attrs

#--------------------------------------------------------------------------------------------------
#--- CLASS :  NoiseDiodeModel
#--------------------------------------------------------------------------------------------------

class NoiseDiodeNotFound(Exception):
    """No noise diode characteristics were found in data file."""
    pass

class NoiseDiodeModel(object):
    """Container for the measured spectrum of a single noise diode.

    The noise temperature of a noise diode is measured in K at a sequence of
    frequencies in MHz. An interpolator allows the noise diode temperature to be
    evaluated at any frequency, and optionally smoothes the data. Additionally,
    metadata related to the noise diode are available as attributes of this
    object.

    The noise diode model may be constructed explicitly from a frequency and
    temperature array and a dictionary of attributes, or it may be loaded from
    a text file if the first parameter is a filename string. If no parameters
    are specified, the default model has a temperature of 1 K for all
    frequencies.

    Parameters
    ----------
    freq : real array-like, shape (N,) or file or string or None, optional
        Array of frequencies where temperature was measured, in MHz.
        Alternatively, a CSV file (or its name) containing noise diode data.
    temp : real array-like, shape (N,) or None, optional
        Array of temperature measurements, in K (ignored if filename was given)
    interp : string or None, optional
        Interpolator to use, as a string that can be evaluated in namespace of
        :mod:`fitting` module to create a :class:`fitting.ScatterFit` object
        (default is piecewise linear interpolator)
    kwargs : dict, optional
        Additional parameters that are turned into attributes of this object.
        Example parameters include *antenna*, *pol*, *diode* and *date*.

    Notes
    -----
    An example of a noise diode model text file is::

      # antenna = ant1
      # pol = H
      # diode = coupler
      # date = 2010-08-05
      # interp = PiecewisePolynomial1DFit(max_degree=3)
      #
      # freq [Hz], T_nd [K]
      1190562500, 4.194
      1191343750, 4.226
      1192125000, 4.226
      ..snip

    The frequencies are specified in Hz, and the noise diode temperature in K.
    These form two required comma-separated columns. Additionally, optional
    attributes may be added as key=value pairs, commented out with hashes at
    the top of the file.

    """
    def __init__(self, freq=None, temp=None, interp=None, **kwargs):
        # The default noise diode model has temperature of 1 K at all frequencies
        if freq is None and temp is None and interp is None:
            freq, temp = np.array([1.]), np.array([1.])
            interp = 'Polynomial1DFit(max_degree=0)'
        # If filename or file-like object is given, load data from file instead
        elif isinstance(freq, basestring) or hasattr(freq, 'readlines'):
            csv, attrs = load_csv_with_header(freq)
            # Keyword arguments override parameters in file
            attrs.update(kwargs)
            kwargs = attrs
            interp = kwargs.get('interp', None) if interp is None else interp
            freq, temp = csv[:, 0] / 1e6, csv[:, 1]
        assert len(freq) == len(temp), \
               'Frequency and temperature arrays should have the same length (%d vs %d)' % (len(freq), len(temp))
        self.freq = freq
        self.temp = temp
        self.interp = 'PiecewisePolynomial1DFit(max_degree=1)' if interp is None else interp
        for key, val in kwargs.iteritems():
            setattr(self, key, val)

    def __eq__(self, other):
        """Equality comparison operator."""
        return (vars(self).keys() == vars(other).keys()) and \
               np.all([np.all(getattr(self, attr) == getattr(other, attr)) for attr in vars(self)])

    def __ne__(self, other):
        """Inequality comparison operator."""
        return not self.__eq__(other)

    def __str__(self):
        """Verbose human-friendly string representation of noise diode model object."""
        label = ''
        for k, v in vars(self).iteritems():
            if k in ('freq', 'temp'):
                continue
            label += '%s = %s\n' % (k, v)
        label += 'freq range: %.0f to %.0f MHz\n' % (self.freq.min(), self.freq.max())
        label += 'average temp: %.1f K' % (self.temp.mean(),)
        return label

    def __repr__(self):
        """Short human-friendly string representation of noise diode model object."""
        label = '%s ' % (getattr(self, 'antenna'),) if hasattr(self, 'antenna') else ''
        label += '%s ' % (getattr(self, 'pol'),) if hasattr(self, 'pol') else ''
        label += '%s ' % (getattr(self, 'diode'),) if hasattr(self, 'diode') else ''
        label = "for '%s' diode" % (label.strip(),) if label else 'object'
        return "<scape.gaincal.NoiseDiodeModel %s at 0x%x>" % (label, id(self))

    def temperature(self, freqs, randomise=False):
        """Obtain noise diode temperature at given frequencies.

        Obtain interpolated noise diode temperature at desired frequencies.
        Optionally, randomise the smooth fit to the noise diode power spectrum,
        to represent some uncertainty as part of a larger Monte Carlo iteration.

        Parameters
        ----------
        freqs : float or array-like, shape (*F*,)
            Frequency (or frequencies) at which to evaluate temperature, in MHz
        randomise : {False, True}, optional
            True if noise diode spectrum smoothing should be randomised

        Returns
        -------
        temp : real array, shape (*F*,)
            Noise diode temperature interpolated to the frequencies in *freqs*

        """
        # Instantiate interpolator object and fit to measurements
        interp = eval(self.interp, vars(fitting))
        interp.fit(self.freq, self.temp)
        # Optionally perturb the fit
        if randomise:
            interp = fitting_randomise(interp, self.freq, self.temp, 'shuffle')
        # Evaluate the smoothed spectrum at the desired frequencies
        return interp(freqs)

#--------------------------------------------------------------------------------------------------
#--- FUNCTIONS
#--------------------------------------------------------------------------------------------------

class NoSuitableNoiseDiodeDataFound(Exception):
    """No suitable noise diode on/off blocks were found in data set."""
    pass

def estimate_nd_jumps(dataset, min_samples=3, min_duration=None, jump_significance=10.0):
    """Estimate jumps in power when noise diode toggles state in data set.

    This examines all time instants where the noise diode flag changes state
    (both off -> on and on -> off). The average power is calculated for the time
    segments immediately before and after the jump, for all frequencies and
    polarisations, using robust statistics. All jumps with a significant
    difference between these two power levels are returned.

    Parameters
    ----------
    dataset : :class:`dataset.DataSet` object
        Data set to analyse
    min_samples : int, optional
        Minimum number of samples in each time segment, to ensure good estimates
    min_duration : float, optional
        Minimum duration of each time segment in seconds. If specified, it
        overrides the *min_samples* value.
    jump_significance : float, optional
        The jump in power level should be at least this number of standard devs

    Returns
    -------
    nd_jump_times : list of floats
        Timestamps at which jumps occur
    nd_jump_power_mu : list of arrays, shape (*F*, 4)
        Mean power level changes at each jump, stored as an array of shape
        (*F*, 4), where *F* is the number of channels/bands
    nd_jump_power_sigma : list of arrays, shape (*F*, 4)
        Standard deviation of power level changes at each jump, stored as an
        array of shape (*F*, 4), where *F* is the number of channels/bands
    nd_jump_info : list of tuples
        Diagnostic information for each jump, stored as a tuple of (scan index,
        jump sample index, "off" sample indices, "on" sample indices)

    """
    nd_jump_times, nd_jump_power_mu, nd_jump_power_sigma, nd_jump_info = [], [], [], []
    if min_duration is not None:
        min_samples = int(np.ceil(dataset.dump_rate * min_duration))
    for scan_ind, scan in enumerate(dataset.scans):
        num_times = len(scan.timestamps)
        # In absence of valid flag, all data is valid
        valid_flag = scan.flags['valid'] if 'valid' in scan.flags.dtype.names else np.tile(True, num_times)
        # Find indices where noise diode flag changes value, or continue on to the next scan
        jumps = (np.diff(scan.flags['nd_on']).nonzero()[0] + 1).tolist()
        if len(jumps) == 0:
            continue
        # The samples immediately before and after the noise diode changes state are invalid for gain calibration
        valid_flag[np.array(jumps) - 1] = False
        valid_flag[jumps] = False
        before_jump = [0] + jumps[:-1]
        at_jump = jumps
        after_jump = jumps[1:] + [num_times]
        # For every jump, obtain segments before and after jump with constant noise diode state
        for start, mid, end in zip(before_jump, at_jump, after_jump):
            # Restrict these segments to indices where data is valid
            before_segment = valid_flag[start:mid].nonzero()[0] + start
            after_segment = valid_flag[mid:end].nonzero()[0] + mid
            # Skip the jump if one or both segments are too short
            if min(len(before_segment), len(after_segment)) < min_samples:
                continue
            # Utilise both off -> on and on -> off transitions
            # (mid is the first sample of the segment after the jump)
            if scan.flags['nd_on'][mid]:
                off_segment, on_segment = before_segment, after_segment
            else:
                on_segment, off_segment = before_segment, after_segment
            # Calculate mean and standard deviation of the *averaged* power data in the two segments.
            # Use robust estimators to suppress spikes and transients in data. Since the estimated mean
            # of data is less variable than the data itself, we have to divide the data sigma by sqrt(N).
            nd_off_mu, nd_off_sigma = robust_mu_sigma(scan.data[off_segment, :, :])
            nd_off_sigma /= np.sqrt(len(off_segment))
            nd_on_mu, nd_on_sigma = robust_mu_sigma(scan.data[on_segment, :, :])
            nd_on_sigma /= np.sqrt(len(on_segment))
            # Obtain mean and standard deviation of difference between averaged power in the segments
            nd_delta_mu, nd_delta_sigma = nd_on_mu - nd_off_mu, np.sqrt(nd_on_sigma ** 2 + nd_off_sigma ** 2)
            # Only keep jumps with significant *increase* in power (focus on the positive HH/VV)
            # This discards segments where noise diode did not fire as expected
            norm_jump = nd_delta_mu / nd_delta_sigma
            norm_jump = norm_jump[:, :2]
            # Remove NaNs which typically occur with perfect simulated data (zero mu and zero sigma)
            norm_jump[np.isnan(norm_jump)] = 0.0
            if np.mean(norm_jump, axis=0).max() > jump_significance:
                nd_jump_times.append(scan.timestamps[mid])
                nd_jump_power_mu.append(nd_delta_mu)
                nd_jump_power_sigma.append(nd_delta_sigma)
                nd_jump_info.append((scan_ind, mid, off_segment, on_segment))
    return nd_jump_times, nd_jump_power_mu, nd_jump_power_sigma, nd_jump_info

def _partition_into_bins(x, width):
    """Partition values into bins of given width.

    This partitions the sequence *x* into bins, where the values in each bin are
    within *width* of each other. The bins are populated sequentially from the
    sorted values of *x*. If *width* is 'all', all values are binned together in
    a single bin, while a width of 0 or 'none' puts each value in its own bin.

    Parameters
    ----------
    x : sequence of numbers
        Sequence of values to partition
    width : number or 'all' or 'none'
        Bin width, so that the range of values in each bin are less than *width*

    Returns
    -------
    bins : list of arrays of ints
        Indices of values in *x*, with one array per bin indicating the members
        of that bin
    centres : array of float, shape (*N*,)
        Centre (mean value) of each bin

    """
    x = np.atleast_1d(x)
    bins, bin_start = [], 0
    ind = np.argsort(x)
    if width == 'all':
        return [ind], np.array([x.mean()])
    elif (width <= 0) or (width == 'none'):
        return [[n] for n in ind], x[ind]
    relative_x = x[ind] - x[ind[0]]
    while bin_start < len(relative_x):
        relative_x -= relative_x[bin_start]
        bin_inds = ind[bin_start + (relative_x[bin_start:] < width).nonzero()[0]]
        bins.append(bin_inds)
        bin_start += len(bin_inds)
    return bins, np.hstack([x[bin].mean() for bin in bins])

def estimate_gain(dataset, interp_degree=1, time_width=900.0, freq_width='all', save=True, randomise=False, **kwargs):
    """Estimate gain and relative phase of both polarisations via injected noise.

    Each successful noise diode transition in the data set is used to estimate
    the gain and relative phase in the two receiver chains for the H and V
    polarisations at the instant of the transition. The H and V gains are power
    gains (i.e. the square of the voltage gains found in the Jones matrix), and
    the phase *phi* is that of the H chain relative to the V chain. The gains
    and phase are measured per frequency channel in the data set. The measured
    gains are then averaged in time-frequency bins of size *time_width* seconds
    by *freq_width* MHz. The function returns spline functions that will
    interpolate these averaged gains to any desired time and frequency. The
    measurements may also be perturbed as part of a Monte Carlo simulation.

    Parameters
    ----------
    dataset : :class:`dataset.DataSet` object
        Data set to analyse
    interp_degree : integer or sequence of 2 integers, optional
        Maximum degree of spline interpolating gains between averaged
        measurements (single degree or separate degrees for time and frequency)
    time_width : float or 'all' or 'none', optional
        Width of averaging bin along time axis, in seconds. Gains measured within
        *time_width* seconds of each other will be averaged together. If this is
        the string 'all', gains are averaged together for all times. If this is
        0 or the string 'none', no averaging is done along the time axis.
    freq_width : float or 'all' or 'none', optional
        Width of averaging bin along frequency axis, in MHz. Gains measured in
        channels within *freq_width* MHz of each other are averaged together.
        If this is string 'all', gains are averaged together over all channels.
        If this is 0 or string 'none', no averaging is done along frequency axis.
    save : {True, False}, optional
        True if estimated gain functions are stored in dataset object
    randomise : {False, True}, optional
        True if raw data and noise diode spectrum smoothing should be randomised
    kwargs : dict, optional
        Extra keyword arguments are passed to :func:`estimate_nd_jumps`

    Returns
    -------
    gain_hh, gain_vv : function, signature ``gain = f(time, freq)``
        Power gain of H and V chain, in units of counts/K, each a function of
        time and frequency. The two inputs to these functions are arrays of
        timestamps and frequencies, of shapes (*T*,) and (*F*,), respectively,
        and the functions output real gain arrays of shape (*T*, *F*).
    delta_re_hv, delta_im_hv : function, signature ``delta = f(time, freq)``
        Terms proportional to *cos(phi)* and *sin(phi)*, where *phi* is the
        phase of H relative to V. Each term is a function of time and frequency,
        and accepts arrays of timestamps and frequencies as inputs, of shapes
        (*T*,) and (*F*,), respectively. The output is a real array of shape
        (*T*, *F*). The phase angle *phi* can be obtained as
        ``phi = np.arctan2(delta_im_hv, delta_re_hv)``.

    Raises
    ------
    NoSuitableNoiseDiodeDataFound
        If no suitable noise diode on/off blocks were found in data set

    """
    interp_degree = list(interp_degree) if hasattr(interp_degree, '__getitem__') else [interp_degree, interp_degree]
    # Obtain noise diode firings
    nd_jump_times, nd_jump_power_mu, nd_jump_power_sigma = estimate_nd_jumps(dataset, **kwargs)[:3]
    if not nd_jump_power_mu:
        raise NoSuitableNoiseDiodeDataFound
    # delta = Pon - Poff = power increase (in counts) when noise diode fires
    deltas = np.concatenate([p[np.newaxis] for p in nd_jump_power_mu])
    if randomise:
        deltas += np.concatenate([p[np.newaxis] for p in nd_jump_power_sigma]) * \
                  np.random.standard_normal(deltas.shape)
    # Interpolate noise diode models to channel frequencies
    temp_nd_h = dataset.nd_h_model.temperature(dataset.freqs, randomise)
    temp_nd_v = dataset.nd_v_model.temperature(dataset.freqs, randomise)
    # The HH (and VV) gain is defined as (Pon - Poff) / Tcal, in counts per K
    deltas[:, :, scape_pol_sd.index('HH')] /= temp_nd_h
    deltas[:, :, scape_pol_sd.index('VV')] /= temp_nd_v

    # Create time-frequency bins and average data into these bins
    time_bins, time_avg = _partition_into_bins(nd_jump_times, time_width)
    freq_bins, freq_avg = _partition_into_bins(dataset.freqs, freq_width)
    deltas_tavg = np.array([deltas[bin, :, :].mean(axis=0) for bin in time_bins])
    deltas_avg = np.hstack([deltas_tavg[:, bin, :].mean(axis=1)[:, np.newaxis, :] for bin in freq_bins])
    # Extend domain of spline extrapolation to full time-frequency range (otherwise extrapolation will be flat)
    timestamps = np.hstack([scan.timestamps for scan in dataset.scans])
    bbox = [timestamps.min(), timestamps.max(), dataset.freqs.min(), dataset.freqs.max()]

    # Make sure at least a first-order spline can be fit, by duplicating a single data point along any axis
    if len(time_avg) == 1:
        deltas_avg = np.tile(deltas_avg, (2, 1, 1))
        time_avg = np.array([time_avg[0], time_avg[0] + 1])
    if len(freq_avg) == 1:
        deltas_avg = np.tile(deltas_avg, (1, 2, 1))
        freq_avg = np.array([freq_avg[0], freq_avg[0] + 1])
    # Reduce spline degree if not enough data is available
    interp_degree[0] = min(interp_degree[0], 2 * (len(time_avg) // 2) - 1)
    interp_degree[1] = min(interp_degree[1], 2 * (len(freq_avg) // 2) - 1)

    # Do a spline interpolation of HH and VV gains between time-frequency bins
    gain_hh_interp = Spline2DGridFit(degree=interp_degree, bbox=bbox)
    gain_hh_interp.fit([time_avg, freq_avg], deltas_avg[:, :, scape_pol_sd.index('HH')])
    gain_vv_interp = Spline2DGridFit(degree=interp_degree, bbox=bbox)
    gain_vv_interp.fit([time_avg, freq_avg], deltas_avg[:, :, scape_pol_sd.index('VV')])
    # Also interpolate the complex-valued HV (real and imag separately), used to derive phase angle between H and V
    delta_re_hv_interp = Spline2DGridFit(degree=interp_degree, bbox=bbox)
    delta_re_hv_interp.fit([time_avg, freq_avg], deltas_avg[:, :, scape_pol_sd.index('ReHV')])
    delta_im_hv_interp = Spline2DGridFit(degree=interp_degree, bbox=bbox)
    delta_im_hv_interp.fit([time_avg, freq_avg], deltas_avg[:, :, scape_pol_sd.index('ImHV')])

    # Return interpolators in convenient form, where t and f are separate arguments (instead of a single list input)
    gain_hh, gain_vv = lambda t, f: gain_hh_interp([t, f]), lambda t, f: gain_vv_interp([t, f])
    delta_re_hv, delta_im_hv = lambda t, f: delta_re_hv_interp([t, f]), lambda t, f: delta_im_hv_interp([t, f])
    if save:
        dataset.nd_gain = {'gain_hh' : gain_hh, 'gain_vv' : gain_vv,
                           'delta_re_hv' : delta_re_hv, 'delta_im_hv' : delta_im_hv}
    return gain_hh, gain_vv, delta_re_hv, delta_im_hv

def calibrate_gain(dataset, **kwargs):
    """Calibrate H and V gains and relative phase, based on noise injection.

    This converts the raw power measurements in the data set to temperatures,
    based on the change in levels caused by switching the noise diode on and off.
    At the same time it corrects for different gains in the H and V polarisation
    receiver chains and for relative phase shifts between them.

    Parameters
    ----------
    dataset : :class:`dataset.DataSet` object
        Data set to calibrate
    kwargs : dict, optional
        Extra keyword arguments are passed to :func:`estimate_gain`

    """
    # Use stored gain functions if available, otherwise estimate the gain first
    if dataset.nd_gain is not None:
        gain_hh, gain_vv = dataset.nd_gain['gain_hh'], dataset.nd_gain['gain_vv']
        delta_re_hv, delta_im_hv = dataset.nd_gain['delta_re_hv'], dataset.nd_gain['delta_im_hv']
    else:
        gain_hh, gain_vv, delta_re_hv, delta_im_hv = estimate_gain(dataset, **kwargs)
    hh, vv, re_hv, im_hv = [scape_pol_sd.index(pol) for pol in ['HH', 'VV', 'ReHV', 'ImHV']]
    for scan in dataset.scans:
        # Interpolate gains based on scan timestamps
        smooth_gain_hh = gain_hh(scan.timestamps, dataset.freqs)
        smooth_gain_vv = gain_vv(scan.timestamps, dataset.freqs)
        smooth_delta_re_hv = delta_re_hv(scan.timestamps, dataset.freqs)
        smooth_delta_im_hv = delta_im_hv(scan.timestamps, dataset.freqs)
        # Remove instances of zero gain, which would lead to NaNs or Infs in the data
        # Usually these are associated with missing H or V polarisations (which get filled in with zeros)
        # Replace them with Infs instead, which suppresses the corresponding channels / polarisations
        # Similar to pseudo-inverse, where scale factors of 1/0 associated with zero eigenvalues are replaced by 0
        smooth_gain_hh[smooth_gain_hh == 0.0] = np.inf
        smooth_gain_vv[smooth_gain_vv == 0.0] = np.inf
        # Scale HH and VV with respective power gains
        scan.data[:, :, hh] /= smooth_gain_hh
        scan.data[:, :, vv] /= smooth_gain_vv
        u, v = scan.data[:, :, re_hv].copy(), scan.data[:, :, im_hv].copy()
        # Rotate U and V through angle -phi, using K cos(phi) and K sin(phi) terms
        scan.data[:, :, re_hv] =  smooth_delta_re_hv * u + smooth_delta_im_hv * v
        scan.data[:, :, im_hv] = -smooth_delta_im_hv * u + smooth_delta_re_hv * v
        # Divide U and V by g_h g_v, as well as length of sin + cos terms above
        gain_hv = np.sqrt(smooth_gain_hh * smooth_gain_vv * (smooth_delta_re_hv ** 2 + smooth_delta_im_hv ** 2))
        # Gain_HV is NaN if HH or VV gain is Inf and Re/Im HV gain is zero (typical of the single-pol case)
        gain_hv[np.isnan(gain_hv)] = np.inf
        scan.data[:, :, re_hv] /= gain_hv
        scan.data[:, :, im_hv] /= gain_hv
    dataset.data_unit = 'K'
    return dataset
