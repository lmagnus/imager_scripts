"""Canned plots."""

import time
import logging

import numpy as np

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    pass

from katpoint import rad2deg, Timestamp, construct_azel_target
from .fitting import PiecewisePolynomial1DFit
from .stats import robust_mu_sigma, remove_spikes, minimise_angle_wrap, identify_rfi_channels
from .beam_baseline import fwhm_to_sigma, extract_measured_beam, interpolate_measured_beam
from .plots_basic import plot_segments, plot_line_segments, plot_compacted_images, plot_marker_3d, \
                         gaussian_ellipses, plot_db_contours, ordinal_suffix

logger = logging.getLogger("scape.plots_canned")

def create_enviro_extractor(dataset, quantity):
    """Create function to extract and interpolate environmental sensor data.

    Parameters
    ----------
    dataset : :class:`DataSet` object
        Data set that contains environmental data
    quantity : string
        Name of environmental quantity to extract

    Returns
    -------
    func : function
        Function that takes Scan object as input and returns interpolated data

    """
    sensor = dataset.enviro[quantity]
    interp = PiecewisePolynomial1DFit(max_degree=0)
    interp.fit(sensor['timestamp'], sensor['value'])
    return lambda scan: interp(scan.timestamps)

class SingleAxisData(object):
    """Struct that contains data for one axis in a plot, with its label and type.

    Parameters
    ----------
    data : list of arrays
        Data, as a list with one element per scan
    type : {'t', 'f', 'tf'}
        Type of data, as inferred from shapes of returned arrays, which can be
        either time-like ('t'), frequency-like ('f') or time-frequency-like ('tf')
    label : string
        Axis label for data

    """
    def __init__(self, data, data_type, label):
        self.data = data
        self.type = data_type
        self.label = label

    def __repr__(self):
        """Short human-friendly string representation of data struct."""
        return "<scape.plots_canned.SingleAxisData '%s' at 0x%x>" % (self.label, id(self))

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  extract_scan_data
#--------------------------------------------------------------------------------------------------

def extract_scan_data(scans, quantity, pol='I'):
    """Extract data from list of Scan objects for plotting purposes.

    Parameters
    ----------
    scans : list of :class:`Scan` objects
        List of Scan objects from which to extract data (assumed to be non-empty)
    quantity : string, or (label, function) tuple
        Quantity to extract from scans, given as either a standard name or a
        tuple containing the axis label and function which will produce the data
        when run on a Scan object
    pol : {'HH', 'VV', 'VH', 'HV', 'XX', 'YY', 'XY', 'YX', 'I', 'Q', 'U', 'V'}
        Polarisation term to extract in the case of visibility data

    Returns
    -------
    data : :class:`SingleAxisData` object
        Extracted data with associated data type and axis label

    Raises
    ------
    ValueError
        If quantity name is unknown

    """
    # Assume all scans come from the same dataset
    dataset = scans[0].compscan.dataset
    # Extract earliest timestamp, used if quantity is 'time' (assumes timestamps are ordered within a scan)
    start = np.min([s.timestamps[0] for s in scans])
    # Create dict of standard quantities and the corresponding functions that will extract them from a Scan object,
    # plus their axis labels
    lf = {# Time-like quantities
          'abs_time' : ('Time (seconds since Unix epoch)', lambda scan: scan.timestamps),
          'time'     : ('Time (s), since %s' % (Timestamp(start).local(),), lambda scan: scan.timestamps - start),
          'az'       : ('Azimuth angle (deg)', lambda scan: rad2deg(scan.pointing['az'])),
          'el'       : ('Elevation angle (deg)', lambda scan: rad2deg(scan.pointing['el'])),
          'ra'       : ('Right ascension (J2000 deg)',
                        lambda scan: rad2deg(np.array([construct_azel_target(az, el).radec(t, dataset.antenna)[0]
                                                       for az, el, t in zip(scan.pointing['az'], scan.pointing['el'],
                                                                            scan.timestamps)]))),
          'dec'      : ('Declination (J2000 deg)',
                        lambda scan: rad2deg(np.array([construct_azel_target(az, el).radec(t, dataset.antenna)[1]
                                                       for az, el, t in zip(scan.pointing['az'], scan.pointing['el'],
                                                                            scan.timestamps)]))),
          'target_x' : ('Target coordinate x (deg)', lambda scan: rad2deg(scan.target_coords[0])),
          'target_y' : ('Target coordinate y (deg)', lambda scan: rad2deg(scan.target_coords[1])),
          # Instantaneous mount coordinates are back on the sphere, but at a single central time instant for all points in compound scan
          'instant_az' : ('Instant azimuth angle (deg)',
                          lambda scan: rad2deg(scan.compscan.target.plane_to_sphere(scan.target_coords[0], scan.target_coords[1],
                                               np.median(np.hstack([s.timestamps for s in scan.compscan.scans])), dataset.antenna)[0])),
          'instant_el' : ('Instant elevation angle (deg)',
                          lambda scan: rad2deg(scan.compscan.target.plane_to_sphere(scan.target_coords[0], scan.target_coords[1],
                                               np.median(np.hstack([s.timestamps for s in scan.compscan.scans])), dataset.antenna)[1])),
          'parangle'   : ('Parallactic angle (deg)', lambda scan: rad2deg(scan.parangle)),
          # Frequency-like quantities
          'freq'     : ('Frequency (MHz)', lambda scan: dataset.freqs),
          'chan'     : ('Channel index', lambda scan: np.arange(len(dataset.freqs))),
          # Time-frequency-like quantities
          'amp'           : ('%s amplitude (%s)' % (pol, dataset.data_unit), lambda scan: np.abs(scan.pol(pol))),
          'phase'         : ('%s phase (deg)' % (pol,), lambda scan: rad2deg(np.angle(scan.pol(pol)))),
          'real'          : ('Real part of %s (%s)' % (pol, dataset.data_unit), lambda scan: scan.pol(pol).real),
          'imag'          : ('Imaginary part of %s (%s)' % (pol, dataset.data_unit), lambda scan: scan.pol(pol).imag),
          'unspiked_amp'  : ('%s amplitude (%s)' % (pol, dataset.data_unit),
                             lambda scan: remove_spikes(np.abs(scan.pol(pol)))),
          'unspiked_real' : ('Real part of %s (%s)' % (pol, dataset.data_unit),
                             lambda scan: remove_spikes(scan.pol(pol).real))}
    # Add enviro sensors as plottable time-like quantities, if they are available
    if 'temperature' in dataset.enviro:
        lf['temperature'] = ('Temperature (deg C)', create_enviro_extractor(dataset, 'temperature'))
    if 'pressure' in dataset.enviro:
        lf['pressure'] = ('Pressure (mbar)', create_enviro_extractor(dataset, 'pressure'))
    if 'humidity' in dataset.enviro:
        lf['humidity'] = ('Relative humidity (%)', create_enviro_extractor(dataset, 'humidity'))
    if 'wind_speed' in dataset.enviro:
        lf['wind_speed'] = ('Wind speed (m/s)', create_enviro_extractor(dataset, 'wind_speed'))
    if 'wind_direction' in dataset.enviro:
        lf['wind_direction'] = ('Wind direction (deg E of N)', create_enviro_extractor(dataset, 'wind_direction'))

    # Obtain axis label and extraction function
    try:
        label, func = lf[quantity] if isinstance(quantity, basestring) else quantity
    except KeyError:
        raise ValueError("Unknown quantity '%s' - choose one of %s (or define your own...)" % (quantity, lf.keys(),))
    # Extract data from scans
    data = [func(s) for s in scans]
    # Infer data type by comparing shape of data to that of timestamps, frequency channels and visibility data
    dt = {tuple([s.data.shape[:2] for s in scans]) : 'tf',
          tuple([dataset.freqs.shape for s in scans]) : 'f',
          tuple([s.timestamps.shape for s in scans]) : 't'}
    return SingleAxisData(data, dt.get(tuple([np.shape(segm) for segm in data])), label)

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  extract_xyz_data
#--------------------------------------------------------------------------------------------------

def extract_xyz_data(data, x, y, z=None, pol='I', band='all', monotonic_axis='auto', scan_labels=None, full_output=False):
    """Extract data from scans for plotting purposes.

    This extracts quantities from a sequence of scans for plotting purposes. The
    scans may be specified by a :class:`DataSet` or :class:`CompoundScan` object
    (both of which contain a list of scans), a single :class:`Scan` object or a
    list of :class:`Scan` objects. The quantities are specified either by name
    (for standard quantities such as time, frequency and visibility amplitude),
    or as a function that will extract the relevant quantity from a :class:`Scan`
    object (which is extremely flexible).

    The function compares the shape of the extracted quantity data with that of
    the scan timestamps, channel frequencies and visibility data to decide
    whether each quantity is time-like, frequency-like or time-frequency-like.
    Based on this classification and a few rules, the data is processed for use
    in line plots, bar plots, image plots and marker plots (by e.g. selecting a
    frequency channel or averaging over frequency or time).

    For visibility data only a single polarisation is used, as specified by the
    *pol* parameter. A single frequency channel may be selected via the *band*
    parameter. The data is returned as :class:`SingleAxisData` objects, which
    combine the actual data (a list of arrays, one per scan) with the axis label
    and data shape type.

    Parameters
    ----------
    data : :class:`DataSet`, :class:`CompoundScan` or :class:`Scan` object (list)
        Part of data set to plot (entire data set, compound scan, scan or scan
        list)
    x, y, z : string or None, or (label, function) tuple
        Quantities to extract from scans and form x, y and z coordinates of plot,
        given either as standard names or as tuples containing the axis label and
        function which will produce the data when run on a :class:`Scan` object.
        Only z is allowed to be None (i.e. omitted).
    pol : {'I', 'Q', 'U', 'V', 'HH', 'VV', 'HV', 'VH', 'XX', 'YY', 'XY', 'YX'}, optional
        Polarisation term to plot if x, y or z specifies visibility data
    band : 'all' or integer, optional
        Single frequency channel/band to select from visibility data (the default
        is to select all channels)
    monotonic_axis : {'auto', 'x', 'y', None}, optional
        This overrides the automatic detection of monotonic axis in underlying
        :func:`plot_segments` function - only needed if detection fails
    scan_labels : sequence of strings, optional
        Sequence of text labels, one per scan (default is scan index)
    full_output : {False, True}, optional
        True if full output should be generated, as needed by func:`plot_xyz`

    Returns
    -------
    xdata, ydata, zdata : :class:`SingleAxisData` objects or None
        Data for each quantity, combined with axis label and data shape type
    tf_stats : list of lists of arrays, or None, optional
        Extra statistics of time-frequency data, of the form (mean, standard
        deviation, min, max) per scan, used by :func:`plot_xyz`
    monotonic_axis : {'auto', 'x', 'y', None}, optional
        Detected monotonic axis, used by :func:`plot_xyz`
    scan_labels : list of strings, optional
        Extracted scan labels, used by :func:`plot_xyz`
    dataset : :class:`DataSet` object, optional
        Data set containing the scans being processed, used by :func:`plot_xyz`

    Raises
    ------
    ValueError
        If scan list is empty or the combination of x, y and z is incompatible

    """
    # Create list of scans from whatever input form the data takes (DataSet and CompoundScan have .scans)
    scans = data if isinstance(data, list) else getattr(data, 'scans', [data])
    if len(scans) == 0:
        raise ValueError('No scans found to plot')
    if scan_labels is None:
        scan_labels = [str(n) for n, s in enumerate(scans)]
    dataset = scans[0].compscan.dataset
    # If there is only one frequency channel/band in the data set, select it
    if (band == 'all') and (len(dataset.freqs) == 1):
        band = 0
    # Hardwire standard monotonic axes (with preference to time) and leave auto selection for user-specified axes only
    if monotonic_axis == 'auto':
        monotonic_axis = 'x' if x == 'time' else 'y' if y == 'time' else \
                         'x' if x in ('freq', 'chan') else 'y' if y in ('freq', 'chan') else 'auto'

    # Extract x, y data from scans
    xdata, ydata = extract_scan_data(scans, x, pol), extract_scan_data(scans, y, pol)
    zdata, tf_stats = None, None

    # If x or y is a visibility ('tf') type, convert it into compatible 1-dimensional form by averaging or selection
    if 'tf' in (xdata.type, ydata.type):
        tf, other = (xdata, ydata) if xdata.type == 'tf' else (ydata, xdata)
        if other.type == 't':
            if band == 'all':
                # Average over all frequency channels
                tf_stats = [list(robust_mu_sigma(d, axis=1)) + [d.min(axis=1), d.max(axis=1)] for d in tf.data]
                tf.data = [stats[0] for stats in tf_stats]
            else:
                # Select one frequency channel / band
                tf.data = [d[:, band] for d in tf.data]
                tf.type = 't'
        elif other.type == 'f':
            tf.data, other.data = np.vstack(tf.data), [other.data[0]]
            # Average over all time samples
            tf_stats = [list(robust_mu_sigma(tf.data, axis=0)) + [tf.data.min(axis=0), tf.data.max(axis=0)]]
            tf.data = [tf_stats[0][0]]
            scan_labels = scan_labels[0] if len(scan_labels) > 0 else []
        elif other.type == 'tf':
            if band == 'all':
                raise ValueError('Please pick single frequency band when plotting visibility data against itself')
            # Select one frequency channel / band for x and y data
            tf.data, other.data = [segm[:, band] for segm in tf.data], [segm[:, band] for segm in other.data]
            tf.type, other.type = 't', 't'
        else:
            raise ValueError('Unsure how to plot time-frequency-like quantity against an unrecognized quantity')

    # If z data is given, check combination of (x, y, z) shapes and transpose and average z data if necessary
    if z is not None:
        zdata = extract_scan_data(scans, z, pol)
        types = (xdata.type, ydata.type, zdata.type)
        # If x and y have the *same* type, convert z into compatible 1-dimensional form by averaging or selection
        if types == ('t', 't', 'tf'):
            if band == 'all':
                # Average over all frequency channels
                zdata.data = [robust_mu_sigma(segm, axis=1)[0] for segm in zdata.data]
            else:
                # Select one frequency channel / band
                zdata.data = [segm[:, band] for segm in zdata.data]
            zdata.type = 't'
        elif types == ('f', 'f', 'tf'):
            xdata.data, ydata.data = [xdata.data[0]], [ydata.data[0]]
            # Average over all time samples
            zdata.data = [robust_mu_sigma(np.vstack(zdata.data), axis=0)[0]]
            scan_labels = scan_labels[0] if len(scan_labels) > 0 else []
            zdata.type = 'f'
        # If required, transpose tf arrays so that rows/columns match y/x types, respectively
        elif types == ('t', 'f', 'tf'):
            zdata.data = [np.asarray(segm).transpose() for segm in zdata.data]
        types = (xdata.type, ydata.type, zdata.type)
        good_shapes = [('t', 't', 't'), ('f', 'f', 'f'), ('t', 'f', 'tf'), ('f', 't', 'tf')]
        if not (types in good_shapes):
            raise ValueError('Bad combination of quantities: (x, y, z) types are %s, should be one of %s' %
                             (types, good_shapes))

    if full_output:
        return xdata, ydata, zdata, tf_stats, monotonic_axis, scan_labels, dataset
    else:
        return xdata, ydata, zdata

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  plot_xyz
#--------------------------------------------------------------------------------------------------

def plot_xyz(data, x='time', y='amp', z=None, pol='I', labels=None, sigma=1.0, band='all',
             power_in_dB=False, compact=True, monotonic_axis='auto', ax=None, **kwargs):
    """Generic plotting of 2 or 3 quantities extracted from sequence of scans.

    This is a very generic plotting routine that plots quantities extracted from
    a sequence of scans against each other. The scans may be specified by a
    :class:`DataSet` or :class:`CompoundScan` object (both of which contain a
    list of scans), a single :class:`Scan` object or a list of :class:`Scan`
    objects. The quantities are specified either by name (for standard quantities
    such as time, frequency and visibility amplitude), or as a function that will
    extract the relevant quantity from a :class:`Scan` object (which is extremely
    flexible).

    If two (*x* and *y*) quantities are given, a two-dimensional plot will
    result (typically a line or bar plot). If three (*x*, *y* and *z*) quantities
    are given, a three-dimensional plot will result (typically a series of images
    or a 3-D marker plot).

    The function compares the shape of the extracted quantity data with that of
    the scan timestamps, channel frequencies and visibility data to decide
    whether each quantity is time-like, frequency-like or time-frequency-like.
    Based on this classification and a few rules, the data is processed into
    line plots, bar plots, image plots and marker plots (by e.g. selecting a
    frequency channel or averaging over frequency or time).

    For visibility data only a single polarisation term is plotted, as specified
    in the *pol* parameter. A single frequency channel may be selected via the
    *band* parameter, which is useful for fringe phase plots. The plot may be
    added to an existing plot by giving an *ax* parameter, else it will be added
    to the current plot. If no current plot exists, a new plot will be created.

    Some quick examples::

       d = scape.DataSet(filename)
       # The default plot is visibility amplitude vs time as a 2-D plot
       # If the data set contains more than one frequency channel, the data is
       # averaged in frequency and plotted with error bars to indicate spread
       # over frequency channels
       scape.plot_xyz(d)
       # Total power spectrum plot of entire data set, which averages the data
       # in time and plots it with error bars to indicate the spread over time
       scape.plot_xyz(d, 'freq', 'amp')
       # Spectrogram of first scan, which shows all the time-frequency data in
       # the form of an image (with false colours indicating amplitude)
       scape.plot_xyz(d.scans[0], 'time', 'freq', 'amp')
       # Plot of antenna coordinates as it scanned across the sky
       scape.plot_xyz(d, 'az', 'el')
       # Plot of total power in projected coordinates relative to the target
       # being scanned across, for the first compound scan
       scape.plot_xyz(d.compscans[0], 'target_x', 'target_y', 'amp')
       # Plot temperature vs time for entire data set
       scape.plot_xyz(d, 'time', 'temperature')
       # If d is an interferometric (single-baseline) data set, this will produce
       # a fringe plot for the specified frequency channel and compound scan
       scape.plot_xyz(d.compscans[0], 'real', 'imag', band=250)

    Parameters
    ----------
    data : :class:`DataSet`, :class:`CompoundScan` or :class:`Scan` object (list)
        Part of data set to plot (entire data set, compound scan, scan or scan
        list)
    x, y, z : string or None, or (label, function) tuple
        Quantities to extract from scans and form x, y and z coordinates of plot,
        given either as standard names or as tuples containing the axis label and
        function which will produce the data when run on a :class:`Scan` object.
        Only z is allowed to be None (i.e. omitted).
    pol : {'I', 'Q', 'U', 'V', 'HH', 'VV', 'HV', 'VH', 'XX', 'YY', 'XY', 'YX'}, optional
        Polarisation term to plot if x, y or z specifies visibility data
    labels : sequence of strings, optional
        Sequence of text labels, one per scan (default is scan index)
    sigma : float, optional
        Error bars are this factor of standard deviation above and below mean
        in bar segment plots (e.g. when averaging visibility data)
    band : 'all' or integer, optional
        Single frequency channel/band to select from visibility data (the default
        is to select all channels)
    power_in_dB : {False, True}, optional
        True if visibility amplitudes should be converted to decibel (dB) scale
    compact : {True, False}, optional
        Plot with no gaps between segments along monotonic axis (only makes
        sense if there is such an axis)
    monotonic_axis : {'auto', 'x', 'y', None}, optional
        This overrides the automatic detection of monotonic axis in underlying
        :func:`plot_segments` function - only needed if detection fails
    ax : :class:`matplotlib.axes.Axes` object, optional
        Matplotlib axes object to receive plot (default is current axes)

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes` object
        Matplotlib Axes object representing plot

    Raises
    ------
    ValueError
        If scan list is empty or the combination of x, y and z is incompatible

    """
    # Extract (x, y, z) data from scans - obtain full output
    xdata, ydata, zdata, tf_stats, monotonic_axis, scan_labels, dataset = \
           extract_xyz_data(data, x, y, z, pol, band, monotonic_axis, labels, full_output=True)

    # The segment widths are set for standard quantities
    width = 1.0 / dataset.dump_rate if 'time' in (x, y) else \
            dataset.bandwidths if 'freq' in (x, y) else 1.0 if 'chan' in (x, y) else 0.0

    # Only now get ready to plot...
    if ax is None:
        ax = plt.gca()
    # If data still has a time-freq x or y component, plot bars indicating data min-max and standard deviation ranges
    if 'tf' in (xdata.type, ydata.type):
        tf = 0 if xdata.type == 'tf' else 1
        min_max = [np.column_stack((tf_min, tf_max)) for tf_mean, tf_stdev, tf_min, tf_max in tf_stats]
        # Clip the stdev range to stay within min-max range
        std_range = [np.column_stack((np.clip(tf_mean - sigma * tf_stdev, tf_min, np.inf),
                                      np.clip(tf_mean + sigma * tf_stdev, -np.inf, tf_max)))
                     for tf_mean, tf_stdev, tf_min, tf_max in tf_stats]
        # Set dB scale right before the plot (don't do arithmetic on dB values!)
        if power_in_dB and ((tf == 0 and x in ('amp', 'unspiked_amp')) or (tf == 1 and y in ('amp', 'unspiked_amp'))):
            min_max, std_range = [10.0 * np.log10(s) for s in min_max], [10.0 * np.log10(s) for s in std_range]
        old_add_breaks, old_color = kwargs.get('add_breaks', True), kwargs.pop('color', None)
        kwargs['add_breaks'] = False
        if tf == 0:
            plot_segments(min_max, ydata.data, width=width, compact=compact,
                          monotonic_axis=monotonic_axis, ax=ax, color='0.8', **kwargs)
            plot_segments(std_range, ydata.data, width=width, compact=compact,
                          monotonic_axis=monotonic_axis, ax=ax, color='0.6', **kwargs)
        else:
            plot_segments(xdata.data, min_max, width=width, compact=compact,
                          monotonic_axis=monotonic_axis, ax=ax, color='0.8', **kwargs)
            plot_segments(xdata.data, std_range, width=width, compact=compact,
                          monotonic_axis=monotonic_axis, ax=ax, color='0.6', **kwargs)
        kwargs['add_breaks'] = old_add_breaks
        if old_color is not None:
            kwargs['color'] = old_color
    # Set dB scale right before the plot (don't do arithmetic on dB values!)
    if power_in_dB:
        if x in ('amp', 'unspiked_amp'):
            xdata.data = [10.0 * np.log10(segm) for segm in xdata.data]
            xdata.label = '%s amplitude (dB %s)' % (pol, dataset.data_unit)
        if y in ('amp', 'unspiked_amp'):
            ydata.data = [10.0 * np.log10(segm) for segm in ydata.data]
            ydata.label = '%s amplitude (dB %s)' % (pol, dataset.data_unit)
        if z in ('amp', 'unspiked_amp'):
            zdata.data = [10.0 * np.log10(segm) for segm in zdata.data]
            zdata.label = '%s amplitude (dB %s)' % (pol, dataset.data_unit)
    # Certain (x, y, z) shapes dictate a scatter plot instead
    if zdata and (xdata.type, ydata.type, zdata.type) in [('t', 't', 't'), ('f', 'f', 'f')]:
        if 'color' not in kwargs:
            kwargs['color'] = 'b'
        plot_marker_3d(np.hstack(xdata.data), np.hstack(ydata.data), np.hstack(zdata.data), ax=ax, alpha=0.75, **kwargs)
        for n, label in enumerate(scan_labels):
            # Add text label just before the start of segment, with white background to make it readable above data
            xsegm, ysegm = xdata.data[n], ydata.data[n]
            lx, ly = xsegm[0] - 0.03 * (xsegm[-1] - xsegm[0]), ysegm[0] - 0.03 * (ysegm[-1] - ysegm[0])
            ax.text(lx, ly, label, ha='center', va='center', clip_on=True, backgroundcolor='w')
    else:
        # Plot main line or image segments
        plot_segments(xdata.data, ydata.data, zdata.data if zdata else None, labels=scan_labels, width=width,
                      compact=compact, monotonic_axis=monotonic_axis, ax=ax, **kwargs)
    # For these pairs of axes (with same units) that typically go together, give plot a square aspect ratio
    if np.any([set((x, y)).issubset(set(p)) for p in (('az', 'el'), ('instant_az', 'instant_el'),
                                                      ('target_x', 'target_y'), ('real', 'imag'))]):
        # Pad the plot with extra space around the edges
        x_range, y_range = ax.xaxis.get_data_interval(), ax.yaxis.get_data_interval()
        extra_space = 0.1 * max(x_range[1] - x_range[0], y_range[1] - y_range[0])
        ax.set_xlim(x_range + extra_space * np.array([-1.0, 1.0]))
        ax.set_ylim(y_range + extra_space * np.array([-1.0, 1.0]))
        ax.set_aspect('equal')
    ax.set_xlabel(xdata.label)
    ax.set_ylabel(ydata.label)
    if zdata:
        ax.set_title(zdata.label)
    return ax

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  plot_spectrum
#--------------------------------------------------------------------------------------------------

def plot_spectrum(dataset, pol='I', scan=-1, sigma=1.0, vertical=True, dB=True, ax=None):
    """Spectrum plot of power data as a function of frequency.

    This plots the power spectrum of the given scan (either in Stokes or
    coherency form), with error bars indicating the variation in the data
    (+/- *sigma* times the standard deviation). Robust statistics are used for
    the plot (median and standard deviation derived from interquartile range).

    Parameters
    ----------
    dataset : :class:`scape.DataSet` object
        Data set to plot
    pol : {'I', 'Q', 'U', 'V', 'HH', 'VV', 'XX', 'YY'}, optional
        The coherency / Stokes parameter to display (must be real)
    scan : int, optional
        Index of scan in data set to plot (-1 to plot all scans together)
    sigma : float, optional
        The error bar is this factor of standard deviation above and below mean
    vertical : {True, False}, optional
        True if frequency is on the x-axis and power is on the y-axis, and False
        if it is the other way around
    dB : {True, False}, optional
        True to plot power logarithmically in dB of the underlying unit
    ax : :class:`matplotlib.axes.Axes` object, optional
        Matplotlib axes object to receive plot (default is current axes)

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes` object
        Matplotlib Axes object representing plot
    power_lim : list of 2 floats
        Overall minimum and maximum value of data, useful for setting plot limits

    Raises
    ------
    ValueError
        If *pol* is not one of the allowed names

    """
    if not pol in ('I', 'Q', 'U', 'V', 'HH', 'VV', 'XX', 'YY'):
        raise ValueError("Polarisation key should be one of 'I', 'Q', 'U', 'V', 'HH', 'VV', 'XX' or 'YY' (i.e. real)")
    if ax is None:
        ax = plt.gca()
    if scan >= 0:
        data = np.abs(dataset.scans[scan].pol(pol))
    else:
        data = np.vstack([np.abs(s.pol(pol)) for s in dataset.scans])
    power_mu, power_sigma = robust_mu_sigma(data)
    power_min, power_max = data.min(axis=0), data.max(axis=0)
    del data

    # Form makeshift rectangular patches indicating power variation in each channel
    power_mean = np.repeat(power_mu, 3)
    power_upper = np.repeat(np.clip(power_mu + sigma * power_sigma, -np.inf, power_max), 3)
    power_lower = np.repeat(np.clip(power_mu - sigma * power_sigma, power_min, np.inf), 3)
    power_min, power_max = np.repeat(power_min, 3), np.repeat(power_max, 3)
    if dB:
        power_mean = 10 * np.log10(power_mean)
        power_min, power_max = 10 * np.log10(power_min), 10 * np.log10(power_max)
        power_upper, power_lower = 10 * np.log10(power_upper), 10 * np.log10(power_lower)
    freqs = np.array([dataset.freqs - 0.999 * dataset.bandwidths / 2.0,
                      dataset.freqs + 0.999 * dataset.bandwidths / 2.0,
                      dataset.freqs + dataset.bandwidths / 2.0]).transpose().ravel()
    mask = np.arange(len(power_mean)) % 3 == 2

    # Fill_between (which uses poly path) is much faster than a Rectangle patch collection
    if vertical:
        ax.fill_between(freqs, power_min, power_max, where=~mask, facecolors='0.8', edgecolors='0.8')
        ax.fill_between(freqs, power_lower, power_upper, where=~mask, facecolors='0.6', edgecolors='0.6')
        ax.plot(freqs, np.ma.masked_array(power_mean, mask), color='b', lw=2)
        ax.plot(dataset.freqs, power_mean[::3], 'ob')
        ax.set_xlim(dataset.freqs[0]  - dataset.bandwidths[0] / 2.0,
                    dataset.freqs[-1] + dataset.bandwidths[-1] / 2.0)
        freq_label, power_label = ax.set_xlabel, ax.set_ylabel
    else:
        ax.fill_betweenx(freqs, power_min, power_max, where=~mask, facecolors='0.8', edgecolors='0.8')
        ax.fill_betweenx(freqs, power_lower, power_upper, where=~mask, facecolors='0.6', edgecolors='0.6')
#        ax.plot(np.ma.masked_array(power_mean, mask), freqs, color='b', lw=2)
        ax.plot(power_mean, np.ma.masked_array(freqs, mask), color='b', lw=2)
        ax.plot(power_mean[::3], dataset.freqs, 'ob')
        ax.set_ylim(dataset.freqs[0]  - dataset.bandwidths[0] / 2.0,
                    dataset.freqs[-1] + dataset.bandwidths[-1] / 2.0)
        freq_label, power_label = ax.set_ylabel, ax.set_xlabel
    freq_label('Frequency (MHz)')
    db_str = 'dB ' if dB else ''
    if dataset.data_unit == 'Jy':
        power_label('Flux density (%sJy)' % db_str)
    elif dataset.data_unit == 'K':
        power_label('Temperature (%sK)' % db_str)
    else:
        power_label('Raw power (%scounts)' % db_str)
    return ax, [power_min.min(), power_max.max()]

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  plot_waterfall
#--------------------------------------------------------------------------------------------------

def plot_waterfall(dataset, title='', channel_skip=None, fig=None):
    """Waterfall plot of power data as a function of time and frequency.

    Parameters
    ----------
    dataset : :class:`scape.DataSet` object
        Data set to plot
    title : string, optional
        Title to add to figure
    channel_skip : int, optional
        Number of channels skipped at a time (i.e. plot every channel_skip'th
        channel). The default results in about 32 channels.
    fig : :class:`matplotlib.figure.Figure` object, optional
        Matplotlib Figure object to contain plots (default is current figure)

    Returns
    -------
    axes_list : list of :class:`matplotlib.axes.Axes` objects
        List of matplotlib Axes objects, one per plot

    """
    if not channel_skip:
        channel_skip = max(len(dataset.freqs) // 32, 1)
    if fig is None:
        fig = plt.gcf()
    # Set up axes: one figure with custom subfigures for waterfall and spectrum plots, with shared x and y axes
    axes_list = []
    axes_list.append(fig.add_axes([0.125, 6 / 11., 0.6, 4 / 11.]))
    axes_list.append(fig.add_axes([0.125, 0.1, 0.6, 4 / 11.],
                                               sharex=axes_list[0], sharey=axes_list[0]))
    axes_list.append(fig.add_axes([0.74, 6 / 11., 0.16, 4 / 11.], sharey=axes_list[0]))
    axes_list.append(fig.add_axes([0.74, 0.1, 0.16, 4 / 11.],
                                  sharex=axes_list[2], sharey=axes_list[0]))

    # Use relative time axis and obtain data limits (of smoothed data) per channel
    scans = dataset.scans
    if not scans:
        logger.error('Data set is empty')
        return
    channel_freqs_MHz = dataset.freqs
    channel_bandwidths_MHz = dataset.bandwidths
    num_channels = len(channel_freqs_MHz)
    data_min = {'HH': np.tile(np.inf, (len(scans), num_channels)),
                'VV': np.tile(np.inf, (len(scans), num_channels))}
    data_max = {'HH': np.zeros((len(scans), num_channels)),
                'VV': np.zeros((len(scans), num_channels))}
    time_origin = np.double(np.inf)
    for n, scan in enumerate(scans):
        time_origin = min(time_origin, scan.timestamps.min())
        for pol in ['HH', 'VV']:
            smoothed_power = remove_spikes(np.abs(scan.pol(pol)))
            channel_min = smoothed_power.min(axis=0)
            data_min[pol][n] = np.where(channel_min < data_min[pol][n], channel_min, data_min[pol][n])
            channel_max = smoothed_power.max(axis=0)
            data_max[pol][n] = np.where(channel_max > data_max[pol][n], channel_max, data_max[pol][n])
    # Obtain minimum and maximum values in each channel (of smoothed data)
    for pol in ['HH', 'VV']:
        data_min[pol] = data_min[pol].min(axis=0)
        data_max[pol] = data_max[pol].max(axis=0)
    channel_list = np.arange(0, num_channels, channel_skip, dtype='int')
    offsets = np.column_stack((np.zeros(len(channel_list), dtype='float'), channel_freqs_MHz[channel_list]))
    scale = 0.08 * num_channels

    # Plot of raw HH and YY power in all channels
    t_limits, p_limits = [], []
    for ax_ind, pol in enumerate(['HH', 'VV']):
        # Time-frequency waterfall plots
        ax = axes_list[ax_ind]
        for compscan_ind, compscan in enumerate(dataset.compscans):
            for scan in compscan.scans:
                # Grey out RFI-tagged channels using alpha transparency
                if scan.label == 'scan':
                    colors = [(0.0, 0.0, 1.0, 1.0 - 0.6 * (chan not in dataset.channel_select))
                              for chan in channel_list]
                else:
                    colors = [(0.0, 0.0, 0.0, 1.0 - 0.6 * (chan not in dataset.channel_select))
                              for chan in channel_list]
                time_line = scan.timestamps - time_origin
                # Normalise the data in each channel to lie between 0 and (channel bandwidth * scale)
                norm_power = scale * channel_bandwidths_MHz[np.newaxis, :] * \
                            (np.abs(scan.pol(pol)) - data_min[pol][np.newaxis, :]) / \
                            (data_max[pol][np.newaxis, :] - data_min[pol][np.newaxis, :])
                segments = [np.vstack((time_line, norm_power[:, chan])).transpose() for chan in channel_list]
                if len(segments) > 1:
                    lines = mpl.collections.LineCollection(segments, colors=colors, offsets=offsets)
                    lines.set_linewidth(0.5)
                    ax.add_collection(lines)
                else:
                    ax.plot(segments[0][:, 0] + offsets.squeeze()[0],
                            segments[0][:, 1] + offsets.squeeze()[1], color=colors[0], lw=0.5)
                t_limits += [time_line.min(), time_line.max()]
            # Add compound scan target name and partition lines between compound scans
            if compscan.scans:
                start_time_ind = len(t_limits) - 2 * len(compscan.scans)
                if compscan_ind >= 1:
                    border_time = (t_limits[start_time_ind - 1] + t_limits[start_time_ind]) / 2.0
                    ax.plot([border_time, border_time], [0.0, 10.0 * channel_freqs_MHz.max()], '--k')
                ax.text((t_limits[start_time_ind] + t_limits[-1]) / 2.0,
                        offsets[0, 1] - scale * channel_bandwidths_MHz[0], compscan.target.name,
                        ha='center', va='bottom', clip_on=True)
        # Set up title and axis labels
        nth_str = ''
        if channel_skip > 1:
            nth_str = '%d%s' % (channel_skip, ordinal_suffix(channel_skip))
        if dataset.data_unit == 'Jy':
            waterfall_title = '%s flux density in every %s channel' % (pol, nth_str)
        if dataset.data_unit == 'K':
            waterfall_title = '%s temperature in every %s channel' % (pol, nth_str)
        else:
            waterfall_title = 'Raw %s power in every %s channel' % (pol, nth_str)
        if pol == 'HH':
            if title:
                title_obj = ax.set_title(title + '\n' + waterfall_title + '\n')
            else:
                title_obj = ax.set_title(waterfall_title + '\n')
            extra_title = '\n\nGreyed-out channels are RFI-flagged'
            # This is more elaborate because the subplot axes are shared
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            title_obj = ax.set_title(waterfall_title + '\n')
            extra_title = '\n(blue = normal scans, black = cal/Tsys scans)'
            ax.set_xlabel('Time (s), since %s' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_origin)))
        # Shrink the font of the second line of the title to make it fit
        title_pos = title_obj.get_position()
        ax.text(title_pos[0], title_pos[1], extra_title, fontsize='smaller', transform=ax.transAxes, ha='center')
        ax.set_ylabel('Frequency (MHz)')
        # Power spectrum box plots
        ax = axes_list[ax_ind + 2]
        ax, power_lim = plot_spectrum(dataset, pol=pol, scan=-1, vertical=False, ax=ax)
        # Add extra ticks on the right to indicate channel numbers
        # second_axis = ax.twinx()
        # second_axis.yaxis.tick_right()
        # second_axis.yaxis.set_label_position('right')
        # second_axis.set_ylabel('Channel number')
        # second_axis.set_yticks(channel_freqs_MHz[channel_list])
        # second_axis.set_yticklabels([str(chan) for chan in channel_list])
        p_limits += power_lim
        ax.set_ylabel('')
        ax.set_title('%s power spectrum' % pol)
        if pol == 'HH':
            # This is more elaborate because the subplot axes are shared
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.set_xlabel('')
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
    # Fix limits globally
    t_limits = np.array(t_limits)
    y_range = channel_freqs_MHz.max() - channel_freqs_MHz.min()
    if y_range < channel_bandwidths_MHz[0]:
        y_range = 10.0 * channel_bandwidths_MHz[0]
    for ax in axes_list[:2]:
        ax.set_xlim(t_limits.min(), t_limits.max())
        ax.set_ylim(channel_freqs_MHz.min() - 0.1 * y_range, channel_freqs_MHz.max() + 0.1 * y_range)
    p_limits = np.array(p_limits)
    for ax in axes_list[2:]:
        ax.set_xlim(p_limits.min(), p_limits.max())

    return axes_list

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  plot_spectrogram
#--------------------------------------------------------------------------------------------------

def plot_spectrogram(dataset, pol='I', add_scan_ids=True, dB=True, ax=None):
    """Plot spectrogram of all scans in data set in compacted form.

    This plots the spectrogram of each scan in the data set on a single set of
    axes, with no gaps between the spectrogram images. This is done for all times
    and all channels. The tick labels on the *x* axis are modified to reflect
    the correct timestamps, and the breaks between scans are indicated by
    vertical lines. RFI-flagged channels are greyed out in the display.

    Parameters
    ----------
    dataset : :class:`scape.DataSet` object
        Data set to plot
    pol : {'I', 'Q', 'U', 'V', 'HH', 'VV', 'HV', 'VH', 'XX', 'YY', 'XY', 'YX'}, optional
        The coherency / Stokes parameter to display (must be real for single-dish)
    add_scan_ids : {True, False}, optional
        True if scan index numbers are to be added to plot
    dB : {True, False}, optional
        True to plot power logarithmically in dB of the underlying unit
    ax : :class:`matplotlib.axes.Axes` object, optional
        Matplotlib axes object to receive plot (default is current axes)

    Returns
    -------
    images : list of :class:`matplotlib.image.AxesImage` objects
        List of spectrogram images
    border_lines : :class:`matplotlib.collections.LineCollection` object
        Collection of vertical lines separating the segments
    text_labels : list of :class:`matplotlib.text.Text` objects
        List of added text labels

    Raises
    ------
    ValueError
        If *pol* is not one of the allowed names

    """
    if ax is None:
        ax = plt.gca()
    db_func = (lambda x: 10.0 * np.log10(x)) if dB else (lambda x: x)
    if dataset.scans[0].has_autocorr:
        if not pol in ('I', 'Q', 'U', 'V', 'HH', 'VV', 'XX', 'YY'):
            raise ValueError("Polarisation key should be one of 'I', 'Q', 'U', 'V', " +
                             "'HH', 'VV', 'XX' or 'YY' (i.e. real) for single-dish data")
        imdata = [db_func(scan.pol(pol)).transpose() for scan in dataset.scans]
    else:
        imdata = [db_func(np.abs(scan.pol(pol))).transpose() for scan in dataset.scans]
    xticks = [scan.timestamps for scan in dataset.scans]
    time_origin = np.min([x.min() for x in xticks])
    labels = [str(n) for n in xrange(len(dataset.scans))] if add_scan_ids else []
    ylim = (dataset.freqs[0], dataset.freqs[-1])
    clim = [np.double(np.inf), np.double(-np.inf)]
    for scan in dataset.scans:
        if dataset.scans[0].has_autocorr:
            smoothed_power = db_func(remove_spikes(scan.pol(pol)))
        else:
            smoothed_power = db_func(remove_spikes(np.abs(scan.pol(pol))))
        clim = [min(clim[0], smoothed_power.min()), max(clim[1], smoothed_power.max())]
    grey_rows = list(set(range(len(dataset.freqs))) - set(dataset.channel_select))
    images, border_lines, text_labels = plot_compacted_images(imdata, xticks, labels, ylim, clim, grey_rows, ax)
    ax.set_xlabel('Time (s), since %s' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_origin)))
    ax.set_ylabel('Channel frequency (MHz)')
    return images, border_lines, text_labels

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  plot_fringes
#--------------------------------------------------------------------------------------------------

def plot_fringes(dataset, pol='I', add_scan_ids=True, ax=None):
    """Plot fringe phase of all scans in data set in compacted form.

    This plots the *fringe phase* (phase as a function of time and frequency) of
    each scan in the data set on a single set of axes as a set of images, with no
    gaps between the images. This is done for all times and all channels. The
    tick labels on the *x* axis are modified to reflect the correct timestamps,
    and the breaks between scans are indicated by vertical lines. RFI-flagged
    channels are greyed out in the display.

    Parameters
    ----------
    dataset : :class:`scape.DataSet` object
        Data set to plot
    pol : {'I', 'Q', 'U', 'V', 'HH', 'VV', 'HV', 'VH', 'XX', 'YY', 'XY', 'YX'}, optional
        The coherency / Stokes parameter to display
    add_scan_ids : {True, False}, optional
        True if scan index numbers are to be added to plot
    ax : :class:`matplotlib.axes.Axes` object, optional
        Matplotlib axes object to receive plot (default is current axes)

    Returns
    -------
    images : list of :class:`matplotlib.image.AxesImage` objects
        List of fringe phase images
    border_lines : :class:`matplotlib.collections.LineCollection` object
        Collection of vertical lines separating the segments
    text_labels : list of :class:`matplotlib.text.Text` objects
        List of added text labels

    Raises
    ------
    ValueError
        If *pol* is not one of the allowed names

    """
    if ax is None:
        ax = plt.gca()
    imdata = [np.angle(scan.pol(pol)).transpose() for scan in dataset.scans]
    xticks = [scan.timestamps for scan in dataset.scans]
    time_origin = np.min([x.min() for x in xticks])
    labels = [str(n) for n in xrange(len(dataset.scans))] if add_scan_ids else []
    ylim = (dataset.freqs[0], dataset.freqs[-1])
    clim = [-np.pi, np.pi]
    grey_rows = list(set(range(len(dataset.freqs))) - set(dataset.channel_select))
    images, border_lines, text_labels = plot_compacted_images(imdata, xticks, labels, ylim, clim, grey_rows, ax)
    ax.set_xlabel('Time (s), since %s' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_origin)))
    ax.set_ylabel('Channel frequency (MHz)')
    return images, border_lines, text_labels

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  plot_rfi_segmentation
#--------------------------------------------------------------------------------------------------

def plot_rfi_segmentation(dataset, sigma=8.0, min_bad_scans=0.25, channel_skip=None, add_scan_ids=True, fig=None):
    """Plot separate time series of data classified as RFI and non-RFI."""
    num_channels = len(dataset.freqs)
    if not channel_skip:
        channel_skip = max(num_channels // 32, 1)
    if fig is None:
        fig = plt.gcf()
    # Set up axes: one figure with custom subfigures for signal and RFI plots, with shared x and y axes
    axes_list = []
    axes_list.append(fig.add_axes([0.125, 6 / 11., 0.8, 4 / 11.]))
    axes_list.append(fig.add_axes([0.125, 0.1, 0.8, 4 / 11.], sharex=axes_list[0], sharey=axes_list[0]))

    labels = [str(n) for n in xrange(len(dataset.scans))] if add_scan_ids else []
    start = np.array([scan.timestamps.min() for scan in dataset.scans])
    end = np.array([scan.timestamps.max() for scan in dataset.scans])
    compacted_start = [0.0] + np.cumsum(end - start).tolist()
    time_origin = start.min()
    # Identify RFI channels, and return extra data
    rfi_channels, rfi_count, rfi_data = identify_rfi_channels(dataset, sigma, min_bad_scans, extra_outputs=True)
    channel_list = np.arange(0, num_channels, channel_skip, dtype='int')
    non_rfi_channels = list(set(range(num_channels)) - set(rfi_channels))
    rfi_channels = [n for n in channel_list if n in rfi_channels]
    non_rfi_channels = [n for n in channel_list if n in non_rfi_channels]
    template = [np.column_stack((scan.timestamps - time_origin, rfi_data[s][1]))
                for s, scan in enumerate(dataset.scans)]
    # Do signal (non-RFI) display
    ax = axes_list[0]
    for s, scan in enumerate(dataset.scans):
        timeline = scan.timestamps - start[s] + compacted_start[s]
        average_std = np.sqrt(np.sqrt(2) / len(timeline)) * rfi_data[s][2][:, non_rfi_channels].mean(axis=1)
        lower, upper = rfi_data[s][1] - np.sqrt(sigma) * average_std, rfi_data[s][1] + np.sqrt(sigma) * average_std
        ax.fill_between(timeline, upper, lower, edgecolor='0.7', facecolor='0.7', lw=0)
        data_segments = [np.column_stack((timeline, rfi_data[s][0][:, n])) for n in non_rfi_channels]
        ax.add_collection(mpl.collections.LineCollection(data_segments))
    plot_line_segments(template, labels, ax=ax, lw=2, color='k')
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('Normalised power')
    # do RFI display
    ax = axes_list[1]
    for s, scan in enumerate(dataset.scans):
        timeline = scan.timestamps - start[s] + compacted_start[s]
        average_std = np.sqrt(np.sqrt(2) / len(timeline)) * rfi_data[s][2][:, rfi_channels].mean(axis=1)
        lower, upper = rfi_data[s][1] - np.sqrt(sigma) * average_std, rfi_data[s][1] + np.sqrt(sigma) * average_std
        ax.fill_between(timeline, upper, lower, edgecolor='0.7', facecolor='0.7', lw=0)
        data_segments = [np.column_stack((timeline, rfi_data[s][0][:, n])) for n in rfi_channels]
        ax.add_collection(mpl.collections.LineCollection(data_segments))
    plot_line_segments(template, labels, ax=ax, lw=2, color='k')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Time (s), since %s' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_origin)))
    ax.set_ylabel('Normalised power')
    return axes_list

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  plot_compound_scan_in_time
#--------------------------------------------------------------------------------------------------

def plot_compound_scan_in_time(compscan, pol='I', add_scan_ids=True, spike_width=0, band=0, ax=None):
    """Plot compound scan data in time with superimposed beam/baseline fit.

    This plots time series of the selected polarisation power in all the scans
    comprising a compound scan, with the beam and baseline fits superimposed.
    It highlights the success of the beam and baseline fitting procedure. It is
    assumed that the beam and baseline was fit to the selected polarisation.

    Parameters
    ----------
    compscan : :class:`compoundscan.CompoundScan` object
        Compound scan object to plot
    pol : {'I', 'Q', 'U', 'V', 'XX', 'YY'}, optional
        The coherency / Stokes parameter to display (must be real)
    add_scan_ids : {True, False}, optional
        True if scan index numbers are to be added to plot
    spike_width : int, optional
        Spikes with widths up to this limit (in samples) will be removed from
        data before determining axis limits. This prevents large spikes from
        messing up the axis limits. A width of <= 0 implies no spike removal.
    band : int, optional
        Frequency band to plot
    ax : :class:`matplotlib.axes.Axes` object, optional
        Matplotlib axes object to receive plot (default is current axes)

    Returns
    -------
    axes_list : list of :class:`matplotlib.axes.Axes` objects
        List of matplotlib Axes objects, one per plot

    Raises
    ------
    ValueError
        If *pol* is not one of the allowed names

    """
    if not pol in ('I', 'Q', 'U', 'V', 'HH', 'VV', 'ReHV', 'ImHV', 'XX', 'YY'):
        raise ValueError("Polarisation key should be one of 'I', 'Q', 'U', 'V', 'HH', 'VV', "
                         "'ReHV', 'ImHV', 'XX' or 'YY' (i.e. real)")
    if ax is None:
        ax = plt.gca()
    time_origin = np.array([scan.timestamps.min() for scan in compscan.scans]).min()
    power_limits, data_segments = [], []
    baseline_segments, beam_segments, inner_beam_segments = [], [], []

    # Construct segments to be plotted
    for scan in compscan.scans:
        timeline = scan.timestamps - time_origin
        measured_power = np.abs(scan.pol(pol)[:, band])
        smooth_power = remove_spikes(measured_power, spike_width=spike_width)
        power_limits.extend([smooth_power.min(), smooth_power.max()])
        data_segments.append(np.column_stack((timeline, measured_power)))
        if scan.baseline:
            baseline_power = scan.baseline(scan.timestamps)
            baseline_segments.append(np.column_stack((timeline, baseline_power)))
        elif compscan.baseline:
            baseline_power = compscan.baseline(scan.target_coords)
            baseline_segments.append(np.column_stack((timeline, baseline_power)))
        else:
            baseline_segments.append(np.column_stack((timeline, np.tile(np.nan, len(timeline)))))
        beam_power, inner_beam_power = np.tile(np.nan, len(timeline)), np.tile(np.nan, len(timeline))
        if compscan.beam:
            if (compscan.beam.refined and scan.baseline) or (not compscan.beam.refined and compscan.baseline):
                beam_power = compscan.beam(scan.target_coords) + baseline_power
            if scan.baseline:
                radius = np.sqrt(((scan.target_coords - compscan.beam.center[:, np.newaxis]) ** 2).sum(axis=0))
                inner = radius < 0.6 * np.mean(compscan.beam.width)
                inner_beam_power = beam_power.copy()
                inner_beam_power[~inner] = np.nan
        beam_segments.append(np.column_stack((timeline, beam_power)))
        inner_beam_segments.append(np.column_stack((timeline, inner_beam_power)))
    # Get overall y limits
    power_range = max(power_limits) - min(power_limits)
    if power_range == 0.0:
        power_range = 1.0
    # Plot segments from back to front
    labels = [str(n) for n in xrange(len(compscan.scans))] if add_scan_ids else []
    plot_line_segments(data_segments, labels, ax=ax, color='b', lw=1)
    beam_color = ('r' if compscan.beam.refined else 'g') if compscan.beam and compscan.beam.is_valid else 'y'
    baseline_colors = [('r' if scan.baseline else 'g') for scan in compscan.scans]
    plot_line_segments(baseline_segments, ax=ax, color=baseline_colors, lw=2)
    if compscan.beam and compscan.beam.refined:
        plot_line_segments(beam_segments, ax=ax, color=beam_color, lw=2, linestyles='dashed')
        plot_line_segments(inner_beam_segments, ax=ax, color=beam_color, lw=2)
    else:
        plot_line_segments(beam_segments, ax=ax, color=beam_color, lw=2)
    ax.set_ylim(min(power_limits) - 0.05 * power_range, max(power_limits) + 0.05 * power_range)
    # Format axes
    ax.set_xlabel('Time (s), since %s' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_origin)))
    ax.set_ylabel('Pol %s' % pol)
    return ax

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  plot_compound_scan_on_target
#--------------------------------------------------------------------------------------------------

def plot_compound_scan_on_target(compscan, pol='I', subtract_baseline=True, levels=None,
                                 add_scan_ids=True, spike_width=0, band=0, ax=None):
    """Plot compound scan data in target space with beam fit.

    This plots contour ellipses of a Gaussian beam function fitted to the scans
    of a compound scan, as well as the selected power of the scans as a pseudo-3D
    plot. It highlights the success of the beam fitting procedure.

    Parameters
    ----------
    compscan : :class:`compoundscan.CompoundScan` object
        Compound scan object to plot
    pol : {'I', 'HH', 'VV', 'XX', 'YY'}, optional
        The coherency / Stokes parameter to display (must be real and positive)
    subtract_baseline : {True, False}, optional
        True to subtract baselines (only scans with baselines are then shown)
    levels : float, or real array-like, shape (K,), optional
        Contour level (or sequence of levels) to plot for Gaussian beam, as
        factor of beam height. The default is [0.5, 0.1].
    add_scan_ids : {True, False}, optional
        True if scan index numbers are to be added to plot
    spike_width : int, optional
        Spikes with widths up to this limit (in samples) will be removed from
        data before plotting it. This prevents large spikes from messing up the
        3D marker autoscaling. A width of <= 0 implies no spike removal.
    band : int, optional
        Frequency band to plot
    ax : :class:`matplotlib.axes.Axes` object, optional
        Matplotlib axes object to receive plot (default is current axes)

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes` object
        Matplotlib Axes object representing plot

    Raises
    ------
    ValueError
        If *pol* is not one of the allowed names

    """
    if not pol in ('I', 'HH', 'VV', 'XX', 'YY'):
        raise ValueError("Polarisation key should be one of 'I', 'HH', 'VV', 'XX' or 'YY' (i.e. positive)")
    if ax is None:
        ax = plt.gca()
    if levels is None:
        levels = [0.5, 0.1]
    # Check that there are any baselines to plot
    if subtract_baseline and not np.any([scan.baseline for scan in compscan.scans]):
        subtract_baseline = False
        logger.warning('No scans were found with baselines - setting subtract_baseline to False')
    # Extract total power and target coordinates (in degrees) of all scans (or those with baselines)
    if subtract_baseline:
        compscan_power = np.hstack([remove_spikes(np.abs(scan.pol(pol)[:, band]), spike_width=spike_width)
                                    - scan.baseline(scan.timestamps) for scan in compscan.scans if scan.baseline])
        target_coords = rad2deg(np.hstack([scan.target_coords for scan in compscan.scans if scan.baseline]))
    else:
        compscan_power = np.hstack([remove_spikes(np.abs(scan.pol(pol)[:, band]), spike_width=spike_width)
                                    for scan in compscan.scans])
        target_coords = rad2deg(np.hstack([scan.target_coords for scan in compscan.scans]))

    # Show the locations of the scan samples themselves, with marker sizes indicating power values
    plot_marker_3d(target_coords[0], target_coords[1], compscan_power, ax=ax, color='b', alpha=0.75)
    # Plot the fitted Gaussian beam function as contours
    if compscan.beam:
        if compscan.beam.is_valid:
            ell_type, center_type = 'r-', 'r+'
        else:
            ell_type, center_type = 'y-', 'y+'
        var = fwhm_to_sigma(compscan.beam.width) ** 2.0
        if np.isscalar(var):
            var = [var, var]
        ellipses = gaussian_ellipses(compscan.beam.center, np.diag(var), contour=levels)
        for ellipse in ellipses:
            ax.plot(rad2deg(ellipse[:, 0]), rad2deg(ellipse[:, 1]), ell_type, lw=2)
        expected_var = fwhm_to_sigma(compscan.beam.expected_width) ** 2.0
        if np.isscalar(expected_var):
            expected_var = [expected_var, expected_var]
        expected_ellipses = gaussian_ellipses(compscan.beam.center, np.diag(expected_var), contour=levels)
        for ellipse in expected_ellipses:
            ax.plot(rad2deg(ellipse[:, 0]), rad2deg(ellipse[:, 1]), 'k--', lw=2)
        ax.plot([rad2deg(compscan.beam.center[0])], [rad2deg(compscan.beam.center[1])],
                center_type, ms=12, aa=False, mew=2)
    # Add scan number label next to the start of each scan
    if add_scan_ids:
        for n, scan in enumerate(compscan.scans):
            if subtract_baseline and not scan.baseline:
                continue
            start, end = rad2deg(scan.target_coords[:, 0]), rad2deg(scan.target_coords[:, -1])
            start_offset = start - 0.03 * (end - start)
            ax.text(start_offset[0], start_offset[1], str(n), ha='center', va='center')
    # Axis settings and labels
    x_range = [target_coords[0].min(), target_coords[0].max()]
    y_range = [target_coords[1].min(), target_coords[1].max()]
    if not np.any(np.isnan(x_range + y_range)):
        extra_space = 0.1 * max(x_range[1] - x_range[0], y_range[1] - y_range[0])
        ax.set_xlim(x_range + extra_space * np.array([-1.0, 1.0]))
        ax.set_ylim(y_range + extra_space * np.array([-1.0, 1.0]))
    ax.set_aspect('equal')
    ax.set_xlabel('x (deg)')
    ax.set_ylabel('y (deg)')
    return ax

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  plot_data_set_in_mount_space
#--------------------------------------------------------------------------------------------------

def plot_data_set_in_mount_space(dataset, levels=None, spike_width=0, band=0, ax=None):
    """Plot total power scans of all compound scans in mount space with beam fits.

    This plots the total power of all scans in the data set as a pseudo-3D plot
    in 'instantaneous mount' space. This space has azimuth and elevation
    coordinates like the standard antenna pointing data, but assumes that each
    compound scan occurred instantaneously at the center time of the compound
    scan. This has the advantage that both fixed and moving targets are frozen
    in mount space, which makes the plots easier to interpret. Its advantage
    over normal target space is that it can display multiple compound scans on
    the same plot.

    For each compound scan, contour ellipses of the fitted Gaussian beam function
    are added, if it exists. It highlights the success of the beam fitting
    procedure.

    Parameters
    ----------
    dataset : :class:`scape.DataSet` object
        Data set to plot
    levels : float, or real array-like, shape (K,), optional
        Contour level (or sequence of levels) to plot for each Gaussian beam, as
        factor of beam height. The default is [0.5, 0.1].
    spike_width : int, optional
        Spikes with widths up to this limit (in samples) will be removed from
        data before plotting it. This prevents large spikes from messing up the
        3D marker autoscaling. A width of <= 0 implies no spike removal.
    band : int, optional
        Frequency band to plot
    ax : :class:`matplotlib.axes.Axes` object, optional
        Matplotlib axes object to receive plot (default is current axes)

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes` object
        Matplotlib Axes object representing plot

    """
    if ax is None:
        ax = plt.gca()
    if levels is None:
        levels = [0.5, 0.1]

    for compscan in dataset.compscans:
        total_power = np.hstack([remove_spikes(np.abs(scan.pol('I')[:, band]), spike_width=spike_width)
                                 for scan in compscan.scans])
        target_coords = np.hstack([scan.target_coords for scan in compscan.scans])
        center_time = np.median(np.hstack([scan.timestamps for scan in compscan.scans]))
        # Instantaneous mount coordinates are back on the sphere, but at a single time instant for all points
        mount_coords = list(compscan.target.plane_to_sphere(target_coords[0], target_coords[1],
                                                            center_time, dataset.antenna))
        # Obtain ellipses and center, and unwrap az angles for all objects simultaneously to ensure they stay together
        if compscan.beam:
            var = fwhm_to_sigma(compscan.beam.width) ** 2.0
            if np.isscalar(var):
                var = [var, var]
            target_ellipses = gaussian_ellipses(compscan.beam.center, np.diag(var), contour=levels)
            mount_ellipses = list(compscan.target.plane_to_sphere(target_ellipses[:, :, 0], target_ellipses[:, :, 1],
                                                                  center_time, dataset.antenna))
            mount_center = list(compscan.target.plane_to_sphere(compscan.beam.center[0], compscan.beam.center[1],
                                                                center_time, dataset.antenna))
            all_az = np.concatenate((mount_coords[0], [mount_center[0]], mount_ellipses[0].flatten()))
            all_az = minimise_angle_wrap(all_az)
            mount_coords[0] = all_az[:len(mount_coords[0])]
            mount_center[0] = all_az[len(mount_coords[0])]
            mount_ellipses[0] = all_az[len(mount_coords[0]) + 1:].reshape(mount_ellipses[0].shape[:2])
        else:
            mount_coords[0] = minimise_angle_wrap(mount_coords[0])

        # Show the locations of the scan samples themselves, with marker sizes indicating power values
        plot_marker_3d(rad2deg(mount_coords[0]), rad2deg(mount_coords[1]), total_power, ax=ax, alpha=0.75)
        # Plot the fitted Gaussian beam function as contours
        if compscan.beam:
            ell_type, center_type = 'r-', 'r+'
            for ellipse in np.dstack(mount_ellipses):
                ax.plot(rad2deg(ellipse[:, 0]), rad2deg(ellipse[:, 1]), ell_type, lw=2)
            ax.plot([rad2deg(mount_center[0])], [rad2deg(mount_center[1])], center_type, ms=12, aa=False, mew=2)

    # Axis settings and labels
    ax.set_aspect('equal')
    ax.set_xlabel('az (deg)')
    ax.set_ylabel('el (deg)')
    return ax

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  plot_measured_beam_pattern
#--------------------------------------------------------------------------------------------------

def plot_measured_beam_pattern(compscan, pol='I', band=0, subtract_baseline=True,
                               add_samples=True, add_colorbar=True, ax=None, **kwargs):
    """Plot measured beam pattern contained in compound scan.

    Parameters
    ----------
    compscan : :class:`compoundscan.CompoundScan` object
        Compound scan object to plot
    pol : {'I', 'Q', 'U', 'V', 'HH', 'VV', 'XX', 'YY'}, optional
        The coherency / Stokes parameter to display (must be real)
    band : int, optional
        Frequency band to plot
    subtract_baseline : {True, False}, optional
        True to subtract baselines (only scans with baselines are then shown)
    add_samples : {True, False}, optional
        True if scan sample locations are to be added
    add_colorbar : {True, False}, optional
        True if color bar indicating contour levels is to be added
    ax : :class:`matplotlib.axes.Axes` object, optional
        Matplotlib axes object to receive plot (default is current axes)
    kwargs : dict, optional
        Extra keyword arguments are passed to underlying :func:`plot_db_contours`
        function

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes` object
        Matplotlib Axes object representing plot
    cset : :class:`matplotlib.contour.ContourSet` object
        Set of filled contour regions (useful for setting up color bar)

    Raises
    ------
    ValueError
        If *pol* is not one of the allowed names

    """
    if ax is None:
        ax = plt.gca()
    # Extract beam pattern as smoothed data on a regular grid
    x, y, power = extract_measured_beam(compscan, pol, band, subtract_baseline)
    # Interpolate scattered data onto regular grid
    grid_x, grid_y, smooth_power = interpolate_measured_beam(x, y, power)
    # Plot contours and associated color bar (if requested)
    cset = plot_db_contours(rad2deg(grid_x), rad2deg(grid_y), smooth_power, ax=ax, **kwargs)
    if add_colorbar:
        plt.colorbar(cset, cax=plt.axes([0.9, 0.1, 0.02, 0.8]), format='%d')
        plt.gcf().text(0.96, 0.5, 'dB')
    # Show the locations of the scan samples themselves
    if add_samples:
        ax.plot(rad2deg(x), rad2deg(y), '.k', ms=2)
    # Axis settings and labels
    ax.set_aspect('equal')
    ax.set_title('Pol %s' % pol)
    return ax, cset
