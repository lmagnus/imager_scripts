"""Container for the data of a single-dish or single-baseline experiment."""

import os.path
import logging

import numpy as np

import katpoint
from .compoundscan import CompoundScan
from .gaincal import calibrate_gain, NoSuitableNoiseDiodeDataFound

# Try to import all available formats
try:
    from .xdmfits import load_dataset as xdmfits_load
    xdmfits_found = True
except ImportError:
    xdmfits_found = False
try:
    from .hdf5 import load_dataset as hdf5_load
    from .hdf5 import save_dataset as hdf5_save
    hdf5_found = True
except ImportError:
    hdf5_found = False

logger = logging.getLogger("scape.dataset")

#--------------------------------------------------------------------------------------------------
#--- CLASS :  DataSet
#--------------------------------------------------------------------------------------------------

class DataSet(object):
    """Container for the data of an experiment (single-dish or a single baseline).

    This is the top-level container for experimental data, which is either
    autocorrelation data for a single dish or cross-correlation data for a
    single interferometer baseline, combined with the appropriate metadata.

    Given a data filename, the initialiser determines the appropriate file
    format to use, based on the file extension. If the filename is blank, the
    :class:`DataSet` can also be directly initialised from its constituent
    parts, which is useful for simulations and creating the data sets from
    scratch. The :class:`DataSet` object contains a list of compound scans, as
    well as experiment information, the correlator configuration, antenna
    details, receiver chain models and weather sensor data.

    Parameters
    ----------
    filename : string
        Name of data set file, or blank string if the other parameters are given
    compscanlist : list of :class:`compoundscan.CompoundScan` objects, optional
        List of compound scans
    experiment_id : string, optional
        Experiment ID, a unique string used to link the data files of an
        experiment together with blog entries, etc.
    observer : string, optional
        Name of person that recorded the data set
    description : string, optional
        Short description of the purpose of the data set
    data_unit : {'counts', 'K', 'Jy'}, optional
        Physical unit of power data
    corrconf : :class:`compoundscan.CorrelatorConfig` object, optional
        Correlator configuration object
    antenna : :class:`katpoint.Antenna` object or string, optional
        Antenna that produced the data set, as object or description string. For
        interferometer data, this is the first antenna.
    antenna2 : :class:`katpoint.Antenna` object or string or None, optional
        Second antenna of baseline, as object or description string. This is
        *None* for single-dish autocorrelation data.
    nd_h_model, nd_v_model : :class:`gaincal.NoiseDiodeModel` objects, optional
        Noise diode models for H and V polarisations on first antenna
    enviro : dict of record arrays, optional
        Environmental (weather) measurements. The keys of the dict are strings
        indicating the type of measurement ('temperature', 'pressure', etc),
        while the values of the dict are record arrays with three elements per
        record: 'timestamp', 'value' and 'status'. The 'timestamp' field is a
        timestamp in UTC seconds since epoch, the 'value' field is the
        corresponding value and the 'status' field is a string indicating the
        sensor status.
    kwargs : dict, optional
        Extra keyword arguments are passed to selected :func:`load_dataset` function

    Attributes
    ----------
    freqs : real array, shape (*F*,)
        Centre frequency of each channel/band, in MHz (same as in *corrconf*)
    bandwidths : real array, shape (*F*,)
        Bandwidth of each channel/band, in MHz (same as in *corrconf*)
    channel_select : list of ints
        Selected channel indices (same as in *corrconf*)
    dump_rate : float
        Correlator dump rate, in Hz (same as in *corrconf*)
    scans : list of :class:`scan.Scan` objects
        Flattened list of all scans in data set
    nd_gain : dict of functions or None
        Receiver gains as functions of time and frequency (per polarisation),
        derived from noise diode calibration (None if no calibration was done)

    Raises
    ------
    ImportError
        If file extension is known, but appropriate module would not import
    ValueError
        If file extension is unknown or parameter is invalid

    """
    def __init__(self, filename, compscanlist=None, experiment_id=None, observer=None,
                 description=None, data_unit=None, corrconf=None, antenna=None, antenna2=None,
                 nd_h_model=None, nd_v_model=None, enviro=None, **kwargs):
        # Load dataset from file
        if filename:
            ext = os.path.splitext(filename)[1]
            if ext == '.fits':
                if not xdmfits_found:
                    raise ImportError('XDM FITS support could not be loaded - please check xdmfits module')
                compscanlist, data_unit, corrconf, \
                antenna, nd_h_model, nd_v_model, enviro = xdmfits_load(filename, **kwargs)
            elif (ext == '.h5') or (ext == '.hdf5'):
                if not hdf5_found:
                    raise ImportError('HDF5 support could not be loaded - please check hdf5 module')
                compscanlist, experiment_id, observer, description, data_unit, \
                corrconf, antenna, antenna2, nd_h_model, nd_v_model, enviro = hdf5_load(filename, **kwargs)
            else:
                raise ValueError("File extension '%s' not understood" % ext)

        self.compscans = compscanlist
        self.experiment_id = experiment_id
        self.observer = observer
        self.description = description
        self.data_unit = data_unit
        self.corrconf = corrconf
        if isinstance(antenna, katpoint.Antenna):
            self.antenna = antenna
        else:
            self.antenna = katpoint.Antenna(antenna)
        if antenna2 is None or isinstance(antenna2, katpoint.Antenna):
            self.antenna2 = antenna2
        else:
            self.antenna2 = katpoint.Antenna(antenna2)
        self.nd_h_model = nd_h_model
        self.nd_v_model = nd_v_model
        self.enviro = enviro
        self.nd_gain = None

        # Fill in caches and links in lower-level objects
        for compscan in self.compscans:
            # Add link to parent in each compound scan object
            compscan.dataset = self
            # Set default antenna on the target object to the (first) data set antenna
            compscan.target.antenna = self.antenna
            good_scans = []
            for scan in compscan.scans:
                # Add link to parent in each scan object
                scan.compscan = compscan
                # Only keep scans with good cached coordinates
                try:
                    scan.calc_cached_coords()
                except ValueError:
                    logger.warning("Discarded scan '%s' - bad target coordinates or parallactic angle" % (scan.path,))
                    continue
                else:
                    good_scans.append(scan)
            compscan.scans = good_scans

    def __eq__(self, other):
        """Equality comparison operator."""
        if len(self.compscans) != len(other.compscans):
            return False
        for self_compscan, other_compscan in zip(self.compscans, other.compscans):
            if self_compscan != other_compscan:
                return False
        return (self.experiment_id == other.experiment_id) and (self.observer == other.observer) and \
               (self.description == other.description) and (self.data_unit == other.data_unit) and \
               (self.corrconf == other.corrconf) and (self.antenna.description == other.antenna.description) and \
               ((self.antenna2 == other.antenna2 == None) or \
                ((self.antenna2 is not None) and (other.antenna2 is not None) and \
                 self.antenna2.description == other.antenna2.description)) and \
               (self.nd_h_model == other.nd_h_model) and (self.nd_v_model == other.nd_v_model) and \
               np.all([key == key2 for key, key2 in zip(self.enviro.iterkeys(), other.enviro.iterkeys())]) and \
               np.all([np.all(val == val2) for val, val2 in zip(self.enviro.itervalues(), other.enviro.itervalues())])

    def __ne__(self, other):
        """Inequality comparison operator."""
        return not self.__eq__(other)

    # Provide properties to access the attributes of the correlator configuration directly
    # This uses a standard trick to create the properties, which leads to less class namespace clutter,
    # but more pylint uneasiness (shame).
    # pylint: disable-msg=E0211,E0202,W0612,W0142
    def freqs():
        """Class method which creates freqs property."""
        doc = 'Centre frequency of each channel/band, in MHz.'
        def fget(self):
            return self.corrconf.freqs
        def fset(self, value):
            self.corrconf.freqs = value
        return locals()
    freqs = property(**freqs())

    # pylint: disable-msg=E0211,E0202,W0612,W0142
    def bandwidths():
        """Class method which creates bandwidths property."""
        doc = 'Bandwidth of each channel/band, in MHz.'
        def fget(self):
            return self.corrconf.bandwidths
        def fset(self, value):
            self.corrconf.bandwidths = value
        return locals()
    bandwidths = property(**bandwidths())

    # pylint: disable-msg=E0211,E0202,W0612,W0142
    def channel_select():
        """Class method which creates channel_select property."""
        doc = 'List of selected channels.'
        def fget(self):
            return self.corrconf.channel_select
        def fset(self, value):
            self.corrconf.channel_select = value
        return locals()
    channel_select = property(**channel_select())

    # pylint: disable-msg=E0211,E0202,W0612,W0142
    def dump_rate():
        """Class method which creates dump_rate property."""
        doc = 'Correlator dump rate, in Hz.'
        def fget(self):
            return self.corrconf.dump_rate
        def fset(self, value):
            self.corrconf.dump_rate = value
        return locals()
    dump_rate = property(**dump_rate())

    def __str__(self):
        """Verbose human-friendly string representation of data set object."""
        descr = ["%s | %s" % (self.experiment_id if self.experiment_id is not None else 'No experiment ID',
                              self.observer if self.observer is not None else 'No observer'),
                 "'%s'" % (self.description if self.description is not None else 'No description',),
                 "%s, data_unit=%s, bands=%d, freqs=%f - %f MHz, total bw=%f MHz, dumprate=%f Hz" %
                 ("antenna='%s'" % self.antenna.name if self.antenna2 is None else \
                  "baseline='%s - %s'" % (self.antenna.name, self.antenna2.name),
                  self.data_unit, len(self.freqs), self.freqs[0], self.freqs[-1],
                  self.bandwidths.sum(), self.dump_rate)]
        for compscan_ind, compscan in enumerate(self.compscans):
            descr.append("%4d: %starget='%s' [%s]" %
                         (compscan_ind, "'%s', " % (compscan.label,) if compscan.label else '',
                          compscan.target.name, compscan.target.body_type))
            if compscan.baseline:
                descr[-1] += ', initial baseline offset=%f' % (compscan.baseline.poly[-1],)
            if compscan.beam:
                descr[-1] += ', beam height=%f' % (compscan.beam.height,)
            for scan_ind, scan in enumerate(compscan.scans):
                descr.append('      %4d: %s' % (scan_ind, str(scan)))
        return '\n'.join(descr)

    def __repr__(self):
        """Short human-friendly string representation of data set object."""
        return "<scape.DataSet '%s' %s compscans=%d at 0x%x>" % (self.experiment_id,
               "antenna='%s'" % self.antenna.name if self.antenna2 is None else
               "baseline='%s - %s'" % (self.antenna.name, self.antenna2.name),
               len(self.compscans), id(self))

    @property
    def scans(self):
        """Flattened list of all scans in data set."""
        return np.concatenate([cs.scans for cs in self.compscans]).tolist()

    def save(self, filename, **kwargs):
        """Save data set object to file.

        This automatically figures out which file format to use based on the
        file extension.

        Parameters
        ----------
        filename : string
            Name of output file
        kwargs : dict, optional
            Extra keyword arguments are passed on to underlying save function

        Raises
        ------
        IOError
            If output file already exists
        ValueError
            If file extension is unknown or unsupported
        ImportError
            If file extension is known, but appropriate module would not import

        """
        if os.path.exists(filename):
            raise IOError('File %s already exists - please remove first!' % filename)
        ext = os.path.splitext(filename)[1]
        if ext == '.fits':
            raise ValueError('XDM FITS writing support not implemented')
        elif (ext == '.h5') or (ext == '.hdf5'):
            if not hdf5_found:
                raise ImportError('HDF5 support could not be loaded - please check hdf5 module')
            hdf5_save(self, filename, **kwargs)
        else:
            raise ValueError("File extension '%s' not understood" % ext)

    def select(self, labelkeep=None, flagkeep=None, freqkeep=None, copy=True):
        """Select subset of data set, based on scan label, flags and frequency.

        This returns a data set with a possibly reduced number of time samples,
        frequency channels/bands and scans, based on the selection criteria.
        Since each scan potentially has a different number of time samples,
        it is less useful to filter directly on sample index. Instead, the
        flags are used to select a subset of time samples in each scan. The
        list of flags are ANDed together to determine which parts are kept. It
        is also possible to invert flags by prepending a ~ (tilde) character.

        Based on the value of *copy*, the new data set contains either a view of
        the original data or a copy. All criteria are optional, and with no
        parameters the returned data set is unchanged. This can be used to make
        a copy of the data set.

        Parameters
        ----------
        labelkeep : string or list of strings, optional
            All scans with labels in this list will be kept. The default is
            None, which means all labels are kept.
        flagkeep : string or list of strings, optional
            List of flags used to select time ranges in each scan. The time
            samples for which all the flags in the list are true are kept.
            Individual flags can be negated by prepending a ~ (tilde) character.
            The default is None, which means all time samples are kept.
        freqkeep : sequence of bools or ints, optional
            Sequence of indicators of which frequency channels/bands to keep
            (either integer indices or booleans that are True for the values to
            be kept). The default is None, which keeps all channels/bands.
        copy : {True, False}, optional
            True if the new scan is a copy, False if it is a view, The default
            currently is True, because views do not always behave as expected
            when using freqkeep or flagkeep (safe for labelkeep).

        Returns
        -------
        dataset : :class:`DataSet` object
            New data set with selection of scans (possibly shared with self)

        Raises
        ------
        KeyError
            If flag in *flagkeep* is unknown

        """
        # Handle the cases of a single input string (not in a list)
        if isinstance(labelkeep, basestring):
            labelkeep = [labelkeep]
        if isinstance(flagkeep, basestring):
            flagkeep = [flagkeep]
        compscanlist = []
        for compscan in self.compscans:
            scanlist = []
            for scan in compscan.scans:
                # Convert flag selection to time sample selection
                if flagkeep is None:
                    timekeep = None
                else:
                    # By default keep all time samples
                    timekeep = np.tile(True, len(scan.timestamps))
                    for flag in flagkeep:
                        invert = False
                        # Flags prepended with ~ get inverted
                        if flag[0] == '~':
                            invert = True
                            flag = flag[1:]
                        # Ignore unknown flags
                        try:
                            flag_data = scan.flags[flag]
                        except KeyError:
                            raise KeyError("Unknown flag '%s'" % flag)
                        if invert:
                            timekeep &= ~flag_data
                        else:
                            timekeep &= flag_data
                if (labelkeep is None) or (scan.label in labelkeep):
                    scanlist.append(scan.select(timekeep, freqkeep, copy))
            if scanlist:
                compscanlist.append(CompoundScan(scanlist, compscan.target))
        return DataSet(None, compscanlist, self.experiment_id, self.observer,
                       self.description, self.data_unit, self.corrconf.select(freqkeep, copy),
                       self.antenna, self.antenna2, self.nd_h_model, self.nd_v_model, self.enviro)

    def convert_power_to_temperature(self, randomise=False, **kwargs):
        """Convert raw power into temperature (K) based on noise injection.

        This is a convenience function that converts the raw power measurements
        in the data set to temperatures, based on the change in levels caused by
        switching the noise diode on and off. At the same time it corrects for
        different gains in the X and Y polarisation receiver chains and for
        relative phase shifts between them. It should be called before averaging
        the data, as gain calibration should happen on the finest available
        frequency scale.

        Parameters
        ----------
        randomise : {False, True}, optional
            True if noise diode data and smoothing should be randomised
        kwargs : dict, optional
            Extra keyword arguments are passed to underlying :mod:`gaincal` functions

        Returns
        -------
        dataset : :class:`DataSet` object
            Data set with calibrated power data (modifies self too)

        """
        # Only operate on raw data
        if self.data_unit != 'counts':
            logger.warning("Expected raw power data to convert to temperature, got data with units '" +
                           self.data_unit + "' instead.")
            return self
        if self.nd_h_model is None or self.nd_v_model is None:
            logger.warning('No noise diode model found in data set - gain calibration not done')
            return self
        try:
            return calibrate_gain(self, randomise=randomise, **kwargs)
        except NoSuitableNoiseDiodeDataFound:
            logger.warning('No suitable noise diode on/off blocks were found - gain calibration not done')

    def average(self, channels_per_band='all', time_window=1):
        """Average data in time and/or frequency.

        If *channels_per_band* is not `None`, the frequency channels are grouped
        into bands, and the power data is merged and averaged within each band.
        Each band contains the average power of its constituent channels. If
        *time_window* is larger than 1, the power data is averaged in time in
        non-overlapping windows of this length, and the rest of the time-varying
        data is averaged accordingly. The default behaviour is to average all
        channels into one band, in line with the continuum focus of :mod:`scape`.

        Parameters
        ----------
        channels_per_band : List of lists of ints, optional
            List of lists of channel indices (one list per band), indicating
            which channels are averaged together to form each band. If this is
            the string 'all', all channels are averaged together into 1 band.
            If this is None, no averaging is done (each channel becomes a band).
        time_window : int, optional
            Window length in samples, within which to average data in time. If
            this is 1 or None, no averaging is done in time.

        Returns
        -------
        dataset : :class:`DataSet` object
            Data set with averaged power data (modifies self too)

        Notes
        -----
        The average power is simpler to use than the total power in each band,
        as total power is dependent on the bandwidth of each band. This method
        should be called *after* :meth:`convert_power_to_temperature`.

        """
        # The string 'all' means average all channels together
        if channels_per_band == 'all':
            channels_per_band = [range(len(self.freqs))]
        # None means no frequency averaging (band == channel)
        if channels_per_band is None:
            channels_per_band = np.expand_dims(range(len(self.freqs)), axis=1).tolist()
        # None means no time averaging too
        if (time_window is None) or (time_window < 1):
            time_window = 1
        # Prune all empty bands
        channels_per_band = [chans for chans in channels_per_band if len(chans) > 0]
        for scan in self.scans:
            # Average power data along frequency axis first
            num_bands = len(channels_per_band)
            band_data = np.zeros((scan.data.shape[0], num_bands, 4), dtype=scan.data.dtype)
            for band_index, band_channels in enumerate(channels_per_band):
                band_data[:, band_index, :] = scan.data[:, band_channels, :].mean(axis=1)
            scan.data = band_data
            # Now average along time axis, if required
            if time_window > 1:
                # Cap the time window length by the scan length, and only keep an integral number of windows
                num_samples = scan.data.shape[0]
                window = min(max(time_window, 1), num_samples)
                new_len = num_samples // window
                cutoff = new_len * window
                scan.data = scan.data[:cutoff, :, :].reshape((new_len, window, num_bands, 4)).mean(axis=1)
                # Also adjust other time-dependent arrays
                scan.timestamps = scan.timestamps[:cutoff].reshape((new_len, window)).mean(axis=1)
                # The record arrays are more involved - form view, average, and reassemble fields
                num_fields = len(scan.pointing.dtype.names)
                view = scan.pointing.view(dtype=np.float32).reshape((num_samples, num_fields))
                view = view[:cutoff, :].reshape((new_len, window, num_fields)).mean(axis=1)
                scan.pointing = view.view(scan.pointing.dtype).squeeze()
                # All flags in a window are OR'ed together to form the 'average'
                num_fields = len(scan.flags.dtype.names)
                view = scan.flags.view(dtype=np.bool).reshape((num_samples, num_fields))
                view = (view[:cutoff, :].reshape((new_len, window, num_fields)).sum(axis=1) > 0)
                scan.flags = view.view(scan.flags.dtype).squeeze()
                if not scan.target_coords is None:
                    scan.target_coords = scan.target_coords[:, :cutoff].reshape((2, new_len, window)).mean(axis=2)
        if time_window > 1:
            self.corrconf.dump_rate /= time_window
        self.corrconf.merge(channels_per_band)
        return self

    def fit_beams_and_baselines(self, *args, **kwargs):
        """Simultaneously fit beams and baselines to all compound scans.

        This fits a beam pattern and baseline to the power data of all the scans
        comprising each compound scan, and stores the resulting fitted functions
        in each :class:`CompoundScan` and :class:`Scan` object. For details on
        the parameters, see :meth:`CompoundScan.fit_beams_and_baselines`.

        Returns
        -------
        dataset : :class:`DataSet` object
            Data set with fitted beam/baseline functions added (modifies self too)

        """
        for compscan in self.compscans:
            compscan.fit_beam_and_baselines(*args, **kwargs)
        return self

    def perturb(self):
        """Perturb visibility data according to expected uncertainty.

        This generates random Gaussian data with the theoretical standard
        deviation of the visibility data (individually determined per timestamp,
        frequency channel and polarisation term) and adds it to the data. This
        is typically used to perturb the data to generate a surrogate data set
        as part of a Monte Carlo run. It assumes that each visibility is a good
        estimate of the true underlying visibility, i.e. that the data standard
        deviation is small (valid for typically large bandwidths and dump
        periods). It also assumes that the visibilities are uncorrelated in
        time, frequency and across polarisation terms.

        Returns
        -------
        dataset : :class:`DataSet` object
            Perturbed data set (modifies self too)

        """
        # This is the ratio of mean power to standard deviation of power (aka signal-to-noise ratio)
        # and is equal to sqrt(K), where K is number of samples averaged together in power estimate.
        # K is also known as accum_per_int (number of spectral samples going into one dump).
        # Since spectral samples are complex, the time-bandwidth product does not include a factor of 2.
        snr = np.sqrt((self.bandwidths * 1e6) / self.dump_rate)
        for scan in self.scans:
            if not scan.has_autocorr:
                logger.warning('Perturbation of data set currently only works for single-dish data')
                break
            hh, vv, rehv, imhv = [scan.data[:, :, n] for n in range(4)]
            # These formulas are based on the moments of real and complex Wishart distributions
            # It replaces the true covariance elements with their sample estimates (assuming the estimates are good)
            std = np.dstack([hh, vv, np.sqrt((hh * vv + rehv ** 2 - imhv ** 2) / 2),
                                     np.sqrt((hh * vv - rehv ** 2 + imhv ** 2) / 2)])
            std /= snr[np.newaxis, :, np.newaxis]
            scan.data += std * np.random.standard_normal(scan.data.shape)
        return self
