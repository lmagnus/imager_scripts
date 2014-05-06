"""Read and write HDF5 files."""

from __future__ import with_statement

import logging
import re
import os.path

import h5py
import numpy as np
import katpoint

from gaincal import NoiseDiodeModel
from scan import Scan, scape_pol_if
from compoundscan import CorrelatorConfig, CompoundScan
from fitting import PiecewisePolynomial1DFit

logger = logging.getLogger("scape.hdf5")

# Parse baseline string into antenna identifiers
baseline_pattern = re.compile('A(\w+)A(\w+)')
# Parse antenna name to extract identifier
antenna_name_pattern = re.compile('ant(\w+)')

def has_data(group, dataset_name):
    """Check whether HDF5 group contains a non-empty dataset with given name.

    Parameters
    ----------
    group : :class:`h5py.Group` object
        HDF5 group to query
    dataset_name : string
        Name of HDF5 dataset to query

    Returns
    -------
    has_data : {True, False}
        True if dataset exists in group and is non-empty

    """
    return dataset_name in group and group[dataset_name].shape != ()

def remove_duplicates(sensor):
    """Remove duplicate timestamp values from sensor data.

    This sorts the 'timestamp' field of the sensor record array and removes any
    duplicate values, updating the corresponding 'value' and 'status' fields as
    well. If more than one timestamp have the same value, the value and status
    of the last of these timestamps are selected. If the values differ for the
    same timestamp, a warning is logged (and the last one is still picked).

    Parameters
    ----------
    sensor : :class:`h5py.Dataset` object, shape (N,)
        Sensor dataset, which acts like a record array with fields 'timestamp',
        'value' and 'status'

    Returns
    -------
    unique_sensor : record array, shape (M,)
        Sensor data with duplicate timestamps removed (M <= N)

    """
    x = np.atleast_1d(sensor['timestamp'])
    y = np.atleast_1d(sensor['value'])
    z = np.atleast_1d(sensor['status'])
    # Sort x via mergesort, as it is usually already sorted and stability is important
    sort_ind = np.argsort(x, kind='mergesort')
    x, y = x[sort_ind], y[sort_ind]
    # Array contains True where an x value is unique or the last of a run of identical x values
    last_of_run = np.asarray(list(np.diff(x) != 0) + [True])
    # Discard the False values, as they represent duplicates - simultaneously keep last of each run of duplicates
    unique_ind = last_of_run.nonzero()[0]
    # Determine the index of the x value chosen to represent each original x value (used to pick y values too)
    replacement = unique_ind[len(unique_ind) - np.cumsum(last_of_run[::-1])[::-1]]
    # All duplicates should have the same y and z values - complain otherwise, but continue
    if not np.all(y[replacement] == y) or not np.all(z[replacement] == z):
        logger.warning("Sensor '%s' has duplicate timestamps with different values or statuses" % sensor.name)
        for ind in (y[replacement] != y).nonzero()[0]:
            logger.debug("At %s, sensor '%s' has values of %s and %s - keeping last one" %
                         (katpoint.Timestamp(x[ind]).local(), sensor.name, y[ind], y[replacement][ind]))
        for ind in (z[replacement] != z).nonzero()[0]:
            logger.debug("At %s, sensor '%s' has statuses of '%s' and '%s' - keeping last one" %
                         (katpoint.Timestamp(x[ind]).local(), sensor.name, z[ind], z[replacement][ind]))
    return np.rec.fromarrays([x[unique_ind], y[unique_ind], z[unique_ind]], dtype=sensor.dtype)

def get_single_value(group, name):
    """Return single value from attribute or dataset with given name in group.

    If `name` is an attribute of the HDF5 group `group`, it is returned,
    otherwise it is interpreted as an HDF5 dataset of `group` and the last value
    of `name` is returned. This is meant to retrieve static configuration values
    that potentially get set more than once during capture initialisation, but
    then does not change during actual capturing.

    Parameters
    ----------
    group : :class:`h5py.Group` object
        HDF5 group to query
    name : string
        Name of HDF5 attribute or dataset to query

    Returns
    -------
    value : object
        Attribute or last value of dataset

    """
    return group.attrs[name] if name in group.attrs else group[name].value[-1]

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  load_dataset
#--------------------------------------------------------------------------------------------------

def load_dataset(filename, *args, **kwargs):
    """Load data set from HDF5 file (auto-detecting format)."""
    f = h5py.File(filename, 'r')
    version = f.attrs.get('version', '1.x')
    f.close()
    return load_dataset_v2(filename, *args, **kwargs) if version.startswith('2.') else \
           load_dataset_v1(filename, *args, **kwargs)

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  load_dataset_v1
#--------------------------------------------------------------------------------------------------

# Mapping of desired fields to KAT sensor names (format version 1)
sensor_name_v1 = {'temperature' : 'enviro_air_temperature',
                  'pressure' : 'enviro_air_pressure',
                  'humidity' : 'enviro_air_relative_humidity',
                  'wind_speed' : 'enviro_wind_speed',
                  'wind_direction' : 'enviro_wind_direction',
                  'coupler_nd_on' : 'rfe3_rfe15_noise_coupler_on',
                  'pin_nd_on' : 'rfe3_rfe15_noise_pin_on'}

# pylint: disable-msg=W0613
def load_dataset_v1(filename, baseline='AxAx', selected_pointing='pos_actual_scan',
                    noise_diode=None, nd_models=None, time_offset=0.0, **kwargs):
    """Load data set from HDF5 file (KAT format version 1).

    This loads a data set from an HDF5 file. The file contains all the baselines
    for the experiment, but :mod:`scape` only operates on one baseline at a time.
    The *baseline* parameter selects which one will be loaded from the file.

    Parameters
    ----------
    filename : string
        Name of input HDF5 file
    baseline : string, optional
        Selected baseline as *AxAy*, where *x* is the number of the first antenna
        and *y* is the number of the second antenna (1-based), and *x* < *y*.
        For single-dish data the antenna number is repeated, e.g. 'A1A1'.
        Alternatively, the baseline may be 'AxAx' for the first single-dish
        baseline or 'AxAy' for the first interferometric baseline in the file.
    selected_pointing : string, optional
        Identifier of (az, el) sensors that will provide all the pointing data
        of the data set. The actual sensor names are formed by appending '_azim'
        and '_elev' to *selected_pointing*. This is ignored if the data set has
        already been processed to contain a single source of pointing data.
    noise_diode : {None, 'coupler', 'pin'}, optional
        Load the model and on/off flags of this noise diode. The default is to
        pick the noise diode with on/off activity if it is the only one showing
        activity, otherwise defaulting to 'coupler'.
    nd_models : None or string, optional
        Override the noise diode models in the HDF5 file using the ones in this
        directory. This assumes that the model files have the format
        '%(antenna).%(diode).%(pol).csv' (e.g. 'ant1.coupler.h.csv').
    time_offset : float, optional
        Offset to add to correlator timestamps, in seconds
    kwargs : dict, optional
        Extra keyword arguments are ignored, as they typically apply to other formats

    Returns
    -------
    compscanlist : list of :class:`compoundscan.CompoundScan` objects
        List of compound scans
    experiment_id : string
        Experiment ID, a unique string used to link the data files of an
        experiment together with blog entries, etc.
    observer : string
        Name of person that recorded the data set
    description : string
        Short description of the purpose of the data set
    data_unit : {'counts', 'K', 'Jy'}
        Physical unit of power data
    corrconf : :class:`compoundscan.CorrelatorConfig` object
        Correlator configuration object
    antenna : string
        Description string of single-dish antenna or first antenna of baseline pair
    antenna2 : string or None
        Description string of second antenna of baseline pair (None for single-dish)
    nd_h_model, nd_v_model : :class:`NoiseDiodeModel` objects
        Noise diode models for H and V polarisations on first antenna
    enviro : dict of record arrays
        Environmental (weather) measurements. The keys of the dict are strings
        indicating the type of measurement ('temperature', 'pressure', etc),
        while the values of the dict are record arrays with three elements per
        record: 'timestamp', 'value' and 'status'. The 'timestamp' field is a
        timestamp in UTC seconds since epoch, the 'value' field is the
        corresponding value and the 'status' field is a string indicating the
        sensor status.

    Raises
    ------
    ValueError
        If file has not been augmented to contain all data fields, or some fields
        are missing
    h5py.H5Error
        If HDF5 error occurred

    """
    # pylint: disable-msg=R0914
    with h5py.File(filename, 'r') as f:
        # Only continue if file has been properly augmented
        if not 'augment' in f.attrs:
            raise ValueError('HDF5 file not augmented - please run augment4.py (provided by k7augment package)')

        # Get attributes at the data set level, with defaults
        experiment_id = f.attrs.get('experiment_id', None)
        observer = f.attrs.get('observer', None)
        description = f.attrs.get('description', None)
        data_unit = f.attrs['data_unit']
        data_timestamps_at_sample_centers = f.attrs['data_timestamps_at_sample_centers']

        # Load antennas group
        ants_group = f['Antennas']
        if baseline == 'AxAx':
            # First single-dish baseline found
            try:
                antA = antB = ants_group.keys()[0]
            except IndexError:
                raise ValueError('Could not load first single-dish baseline - no antennas found in file')
            logger.info("Loading single-dish baseline 'A%sA%s'" % (antA[7:], antB[7:]))
        elif baseline == 'AxAy':
            # First interferometric baseline found
            try:
                antA, antB = ants_group.keys()[:2]
            except IndexError:
                raise ValueError('Could not load first interferometric baseline - less than 2 antennas found in file')
            logger.info("Loading interferometric baseline 'A%sA%s'" % (antA[7:], antB[7:]))
        else:
            # Select antennas involved in specified baseline
            parsed_antenna_ids = baseline_pattern.match(baseline)
            if parsed_antenna_ids is None:
                raise ValueError("Please specify baseline with notation 'AxAy', "
                                 "where x is identifier of first antenna and y is identifier of second antenna")
            antA, antB = ['Antenna' + ident for ident in parsed_antenna_ids.groups()]
        # Check that requested antennas are in data set
        if antA not in ants_group or antB not in ants_group:
            raise ValueError('Requested antenna pair not found in HDF5 file (wanted %s but file only contains %s)'
                             % ([antA, antB], ants_group.keys()))
        antA_group, antB_group = ants_group[antA], ants_group[antB]
        # Get antenna description strings (antenna2 is None for single-dish data)
        antenna, antenna2 = antA_group.attrs['description'], antB_group.attrs['description']
        if antenna2 == antenna:
            antenna2 = None

        # Use first scan group to check for 'flags' and 'pointing' datasets (associated with processed / saved files)
        try:
            first_scan_group = f['Scans'].values()[0].values()[0]
        except IndexError, AttributeError:
            first_scan_group = {}

        # Use antenna A for sensor data (weather, pointing, noise diode)
        sensors_group = antA_group['Sensors']

        # Load weather sensor data
        enviro = {}
        for quantity in ['temperature', 'pressure', 'humidity', 'wind_speed', 'wind_direction']:
            sensor = sensor_name_v1[quantity]
            # Environment sensors are optional
            if has_data(sensors_group, sensor):
                enviro[quantity] = remove_duplicates(sensors_group[sensor])

        # Load pointing sensor data and fit interpolator as a function of time
        pointing_interp = {}
        for coord in ('azim', 'elev'):
            sensor = '%s_%s' % (selected_pointing, coord)
            if has_data(sensors_group, sensor):
                # Ensure pointing timestamps are unique before interpolation
                original_coord = remove_duplicates(sensors_group[sensor])
                # Linearly interpolate pointing coordinates to correlator data timestamps
                # As long as azimuth is in natural antenna coordinates, no special angle interpolation required
                pointing_interp[coord] = PiecewisePolynomial1DFit(max_degree=1)
                pointing_interp[coord].fit(original_coord['timestamp'], original_coord['value'])
            elif 'pointing' not in first_scan_group:
                raise ValueError("Selected pointing sensor '%s' was not found in HDF5 file" % (sensor,))

        # Autodetect the noise diode to use, based on which sensor shows any activity
        if not noise_diode:
            # In a processed / intermediate file, there are no noise diode sensors - rather check model attribute
            if 'H' in antA_group and 'nd_model' in antA_group['H']:
                noise_diode = antA_group['H']['nd_model'].attrs.get('diode', None)
            elif 'V' in antA_group and 'nd_model' in antA_group['V']:
                noise_diode = antA_group['V']['nd_model'].attrs.get('diode', None)
            else:
                nd_fired = {}
                for nd in ('coupler', 'pin'):
                    sensor = sensor_name_v1[nd + '_nd_on']
                    nd_fired[nd] = np.any(sensors_group[sensor]['value'] == '1') \
                                   if has_data(sensors_group, sensor) else False
                if np.sum(nd_fired.values()) == 1:
                    noise_diode = nd_fired.keys()[nd_fired.values().index(True)]
                    logger.info("Using '%s' noise diode as it is the only one firing in data set" % noise_diode)
                else:
                    noise_diode = 'coupler'
                    logger.info("Defaulting to '%s' noise diode (either no or both diodes are firing)" % noise_diode)
        # Load noise diode flags and fit interpolator as a function of time
        nd_interp = None
        sensor = sensor_name_v1[noise_diode + '_nd_on']
        if has_data(sensors_group, sensor):
            # Ensure noise diode timestamps are unique before interpolation
            nd_on = remove_duplicates(sensors_group[sensor])
            # Do step-wise interpolation (as flag is either 0 or 1 and holds its value until it toggles)
            nd_interp = PiecewisePolynomial1DFit(max_degree=0)
            try:
                # Assumes that flag values are '1' and '0' (or 1 and 0) for True and False, respectively
                nd_flags = nd_on['value'].astype(int)
            except ValueError:
                # Assumes that flag values are 'True' and 'False' for True and False, respectively
                nd_flags = [(1 if flag == 'True' else 0) for flag in nd_on['value']]
            nd_interp.fit(nd_on['timestamp'], nd_flags)
        elif 'flags' not in first_scan_group:
            logger.warning("Selected noise diode sensor '%s' not found in HDF5 file - setting nd_on to False" % sensor)
        # First try to load external noise diode models, if provided
        nd_h_model = nd_v_model = None
        if nd_models:
            if not noise_diode:
                raise ValueError('Unable to pick right noise diode model file, as noise diode could not be identified')
            antA_name = antenna.split(',')[0]
            if not antA_name:
                raise ValueError('Unable to pick right noise diode model file, as antenna could not be identified')
            nd_h_file = os.path.join(nd_models, '%s.%s.h.csv' % (antA_name, noise_diode))
            nd_h_model = NoiseDiodeModel(nd_h_file)
            logger.info("Loaded H noise diode model from '%s'" % (nd_h_file,))
            nd_v_file = os.path.join(nd_models, '%s.%s.v.csv' % (antA_name, noise_diode))
            nd_v_model = NoiseDiodeModel(nd_v_file)
            logger.info("Loaded V noise diode model from '%s'" % (nd_v_file,))
        else:
            def load_nd_dataset(nd_dataset_name):
                """Load H and V noise diode models from selected HDF5 dataset."""
                nd_h_model = nd_v_model = None
                if 'H' in antA_group:
                    nd_dataset = antA_group['H'][nd_dataset_name]
                    nd_h_model = NoiseDiodeModel(nd_dataset[:, 0] / 1e6, nd_dataset[:, 1], **dict(nd_dataset.attrs))
                    if 'V' not in antA_group:
                        nd_v_model = NoiseDiodeModel()
                if 'V' in antA_group:
                    nd_dataset = antA_group['V'][nd_dataset_name]
                    nd_v_model = NoiseDiodeModel(nd_dataset[:, 0] / 1e6, nd_dataset[:, 1], **dict(nd_dataset.attrs))
                    if 'H' not in antA_group:
                        nd_h_model = NoiseDiodeModel()
                return nd_h_model, nd_v_model
            # Now try to load generic noise diode model (typically found in processed / intermediate file)
            try:
                nd_h_model, nd_v_model = load_nd_dataset('nd_model')
            except KeyError:
                # Now try to load models for selected noise diode
                try:
                    nd_h_model, nd_v_model = load_nd_dataset(noise_diode + '_nd_model')
                except KeyError:
                    # No models were found for the selected diode - quit and report any ones that are there instead
                    # Find noise diode model datasets common to both 'H' and 'V' (if these feeds exist)
                    nd_set = None
                    if 'H' in antA_group:
                        nd_set = set([name.rpartition('_nd_model')[0] for name in antA_group['H']])
                        nd_set.discard('')
                    if 'V' in antA_group:
                        nd_set2 = set([name.rpartition('_nd_model')[0] for name in antA_group['V']])
                        nd_set2.discard('')
                        nd_set = nd_set2 if nd_set is None else nd_set.intersection(nd_set2)
                    raise ValueError("Unknown noise diode '%s', found the following models instead: %s" %
                                     (noise_diode, list(nd_set)))

        # Load correlator configuration group
        corrconf_group = f['Correlator']
        # If center_freqs dataset is available, use it - otherwise, reconstruct it from DBE attributes
        center_freqs = corrconf_group.get('center_freqs', None)
        if center_freqs:
            center_freqs = center_freqs.value / 1e6
            bandwidths = corrconf_group['bandwidths'].value / 1e6
            num_chans = len(center_freqs)
        else:
            band_center = corrconf_group.attrs['center_frequency_hz'] / 1e6
            channel_bw = corrconf_group.attrs['channel_bandwidth_hz'] / 1e6
            num_chans = corrconf_group.attrs['num_freq_channels']
            # Assume that lower-sideband downconversion has been used, which flips frequency axis
            # Also subtract half a channel width to get frequencies at center of each channel
            center_freqs = band_center - channel_bw * (np.arange(num_chans) - num_chans / 2 + 0.5)
            bandwidths = np.tile(np.float64(channel_bw), num_chans)
        channel_select = corrconf_group['channel_select'].value.nonzero()[0].tolist()
        dump_rate = corrconf_group.attrs['dump_rate_hz']
        sample_period = 1.0 / dump_rate
        corrconf = CorrelatorConfig(center_freqs, bandwidths, channel_select, dump_rate)

        # Figure out mapping of antennas and feeds to the correlation products involved
        if 'input_map' in corrconf_group:
            # Obtain DBE input associated with each polarisation of selected antennas (indicating unavailable feeds)
            antA_H = antA_group['H'].attrs['dbe_input'] if 'H' in antA_group else 'UNAVAILABLE'
            antA_V = antA_group['V'].attrs['dbe_input'] if 'V' in antA_group else 'UNAVAILABLE'
            antB_H = antB_group['H'].attrs['dbe_input'] if 'H' in antB_group else 'UNAVAILABLE'
            antB_V = antB_group['V'].attrs['dbe_input'] if 'V' in antB_group else 'UNAVAILABLE'
            # Mapping of polarisation product to DBE input string identifying the pair of inputs multiplied together
            pol_to_dbestr = {'HH' : antA_H + antB_H, 'VV' : antA_V + antB_V,
                             'HV' : antA_H + antB_V, 'VH' : antA_V + antB_H}
            # Correlator mapping of DBE input string to correlation product index (Miriad-style numbering)
            input_map = corrconf_group['input_map'].value
            dbestr_to_corr_id = dict(zip(input_map['dbe_inputs'], input_map['correlator_product_id']))
            # Overall mapping from polarisation product to correlation product index (None for unavailable products)
            pol_to_corr_id = dict([(pol, dbestr_to_corr_id.get(pol_to_dbestr[pol])) for pol in scape_pol_if])
        else:
            # Simplified mapping, used in processed / intermediate files
            pol_to_corr_id = dict([(pol, n) for n, pol in enumerate(scape_pol_if)])

        # Load each compound scan group
        compscanlist = []
        for compscan in f['Scans']:
            compscan_group = f['Scans'][compscan]
            compscan_target = compscan_group.attrs.get('target', 'Nothing, special')
            compscan_label = compscan_group.attrs.get('label', '')

            # Load each scan group within compound scan
            scanlist = []
            for scan in compscan_group:
                scan_group = compscan_group[scan]
                data = scan_group['data']
                # Convert from millisecs to secs since Unix epoch, and be sure to use float64 to preserve digits
                data_timestamps = scan_group['timestamps'].value.astype(np.float64) / 1000.0 + time_offset
                num_times = len(data_timestamps)
                # Move correlator data timestamps from start of each sample to the middle
                if not data_timestamps_at_sample_centers:
                    data_timestamps += 0.5 * sample_period
                # If data timestamps have problems, warn and discard the scan
                if np.any(data_timestamps < 1000000000.0) or (num_times > 1 and np.diff(data_timestamps).min() == 0.0):
                    logger.warning("Discarded %s/%s - bad correlator timestamps (duplicates or way out of date)" %
                                   (compscan, scan))
                    continue

                # Load correlation data either in float64 (single-dish) form or complex128 (interferometer) form
                # Data has already been normalised by number of samples in integration (accum_per_int)
                # Data of missing feeds are set to zero, as this simplifies the data structure (always 4 products)
                if antenna2 is None:
                    # Single-dish uses HH, VV, Re{HV}, Im{HV}
                    corr_id = [pol_to_corr_id[pol] for pol in ['HH', 'VV', 'HV', 'HV']]
                    scan_data = [data[str(cid)] if cid is not None else np.zeros((num_times, num_chans), np.float64)
                                 for cid in corr_id]
                    scan_data = np.dstack([scan_data[0].real, scan_data[1].real,
                                           scan_data[2].real, scan_data[3].imag]).astype(np.float64)
                else:
                    corr_id = [pol_to_corr_id[pol] for pol in scape_pol_if]
                    scan_data = [data[str(cid)] if cid is not None else np.zeros((num_times, num_chans), np.complex128)
                                 for cid in corr_id]
                    scan_data = np.dstack(scan_data).astype(np.complex128)

                # Use per-scan pointing if it is available, otherwise interpolate original data
                scan_pointing = scan_group['pointing'].value if 'pointing' in scan_group else \
                                np.rec.fromarrays([pointing_interp['azim'](data_timestamps).astype(np.float32),
                                                   pointing_interp['elev'](data_timestamps).astype(np.float32)],
                                                  names=('az', 'el'))
                # Convert contents of pointing from degrees to radians
                # pylint: disable-msg=W0612
                pointing_view = scan_pointing.view(np.float32)
                pointing_view *= np.float32(np.pi / 180.0)

                # Use per-scan flags if it is available, otherwise interpolate original data
                scan_flags = scan_group['flags'].value if 'flags' in scan_group else \
                             np.rec.fromarrays([nd_interp(data_timestamps).astype(bool)],
                                               names=('nd_on',)) if nd_interp else \
                             np.rec.fromarrays([np.tile(False, data_timestamps.shape)], names=('nd_on',))
                scan_label = scan_group.attrs.get('label', '')

                scanlist.append(Scan(scan_data, data_timestamps, scan_pointing, scan_flags, scan_label,
                                     filename + '/Scans/%s/%s' % (compscan, scan)))

            if len(scanlist) > 0:
                # Sort scans chronologically, as h5py seems to scramble them based on group name
                scanlist.sort(key=lambda scan: scan.timestamps[0])
                compscanlist.append(CompoundScan(scanlist, compscan_target, compscan_label))

        if len(compscanlist) > 0:
            # Sort compound scans chronologically too
            compscanlist.sort(key=lambda compscan: compscan.scans[0].timestamps[0])
        return compscanlist, experiment_id, observer, description, data_unit, \
               corrconf, antenna, antenna2, nd_h_model, nd_v_model, enviro

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  load_dataset_v2
#--------------------------------------------------------------------------------------------------

# Mapping of desired fields to KAT sensor names (format version 2)
sensor_name_v2 = {'temperature' : 'asc.air.temperature',
                  'pressure' : 'asc.air.pressure',
                  'humidity' : 'asc.air.relative-humidity',
                  'wind_speed' : 'asc.wind.speed',
                  'wind_direction' : 'asc.wind.direction',
                  'coupler_nd_on' : 'rfe3.rfe15.noise.coupler.on',
                  'pin_nd_on' : 'rfe3.rfe15.noise.pin.on',
                  'pos_actual_scan' : 'pos.actual-scan',
                  'pos_actual_refrac' : 'pos.actual-refrac',
                  'pos_actual_pointm' : 'pos.actual-pointm',
                  'pos_request_scan' : 'pos.request-scan',
                  'pos_request_refrac' : 'pos.request-refrac',
                  'pos_request_pointm' : 'pos.request-pointm'}

# pylint: disable-msg=W0613
def load_dataset_v2(filename, baseline='sd', selected_pointing='pos_actual_scan',
                    noise_diode=None, nd_models=None, time_offset=0.0, **kwargs):
    """Load data set from HDF5 file (KAT format version 2).

    This loads a data set from an HDF5 file. The file contains all the baselines
    for the experiment, but :mod:`scape` only operates on one baseline at a time.
    The *baseline* parameter selects which one will be loaded from the file.

    Parameters
    ----------
    filename : string
        Name of input HDF5 file
    baseline : string, optional
        Selected baseline as *AxAy*, where *x* is the identifier of the first
        antenna (i.e. the part following 'ant' in its name) and *y* is the
        identifier of the second antenna, and *x* < *y*. For single-dish data
        the identifier is repeated, e.g. 'A1A1'. Alternatively, the baseline may
        be 'sd' for the first single-dish baseline or 'if' for the first
        interferometric baseline in the file.
    selected_pointing : string, optional
        Identifier of (az, el) sensors that will provide all the pointing data
        of the data set. The actual sensor names are formed by appending '_azim'
        and '_elev' to *selected_pointing*. This is ignored if the data set has
        already been processed to contain a single source of pointing data.
    noise_diode : {None, 'coupler', 'pin'}, optional
        Load the model and on/off flags of this noise diode. The default is to
        pick the noise diode with on/off activity if it is the only one showing
        activity, otherwise defaulting to 'coupler'.
    nd_models : None or string, optional
        Override the noise diode models in the HDF5 file using the ones in this
        directory. This assumes that the model files have the format
        '%(antenna).%(diode).%(pol).csv' (e.g. 'ant1.coupler.h.csv').
    time_offset : float, optional
        Offset to add to correlator timestamps, in seconds
    kwargs : dict, optional
        Extra keyword arguments are ignored, as they typically apply to other formats

    Returns
    -------
    compscanlist : list of :class:`compoundscan.CompoundScan` objects
        List of compound scans
    experiment_id : string
        Experiment ID, a unique string used to link the data files of an
        experiment together with blog entries, etc.
    observer : string
        Name of person that recorded the data set
    description : string
        Short description of the purpose of the data set
    data_unit : {'counts', 'K', 'Jy'}
        Physical unit of power data
    corrconf : :class:`compoundscan.CorrelatorConfig` object
        Correlator configuration object
    antenna : string
        Description string of single-dish antenna or first antenna of baseline pair
    antenna2 : string or None
        Description string of second antenna of baseline pair (None for single-dish)
    nd_h_model, nd_v_model : :class:`NoiseDiodeModel` objects
        Noise diode models for H and V polarisations on first antenna
    enviro : dict of record arrays
        Environmental (weather) measurements. The keys of the dict are strings
        indicating the type of measurement ('temperature', 'pressure', etc),
        while the values of the dict are record arrays with three elements per
        record: 'timestamp', 'value' and 'status'. The 'timestamp' field is a
        timestamp in UTC seconds since epoch, the 'value' field is the
        corresponding value and the 'status' field is a string indicating the
        sensor status.

    Raises
    ------
    ValueError
        If file has not been augmented to contain all data fields, or some fields
        are missing
    h5py.H5Error
        If HDF5 error occurred

    """
    # pylint: disable-msg=R0914
    with h5py.File(filename, 'r') as f:
        # Only continue if file is correct version and has been properly augmented
        version = f.attrs.get('version', '1.x')
        if not version.startswith('2.'):
            raise ValueError("Attempting to load version '%s' file with version 2 loader" % (version,))
        if not 'augment_ts' in f.attrs:
            raise ValueError('HDF5 file not augmented - please run k7_augment.py (provided by katsdisp package)')

        # Get observation script attributes, with defaults
        script_attrs = f['MetaData/Configuration/Observation'].attrs
        experiment_id = script_attrs.get('script_experiment_id', None)
        observer = script_attrs.get('script_observer', None)
        description = script_attrs.get('script_description', None)

        # Load antenna configuration group
        ant_config = f['MetaData/Configuration/Antennas']
        # These antennas were used by the observation script
        ants_used = script_attrs.get('script_ants', '').split(',')
        if baseline == 'sd':
            # First single-dish baseline found
            try:
                antA = antB = ants_used[0] if len(ants_used) > 0 else ant_config.keys()[0]
            except IndexError:
                raise ValueError('Could not load first single-dish baseline - no antennas found in file')
            logger.info("Loading single-dish baseline 'A%sA%s'" % (antA[3:], antB[3:]))
        elif baseline == 'if':
            # First interferometric baseline found
            try:
                antA, antB = ants_used[:2] if len(ants_used) > 1 else ant_config.keys()[:2]
            except IndexError:
                raise ValueError('Could not load first interferometric baseline - less than 2 antennas found in file')
            logger.info("Loading interferometric baseline 'A%sA%s'" % (antA[3:], antB[3:]))
        else:
            # Select antennas involved in specified baseline
            parsed_antenna_ids = baseline_pattern.match(baseline)
            if parsed_antenna_ids is None:
                raise ValueError("Please specify baseline with notation 'AxAy', " +
                                 "where x is identifier of first antenna and y is identifier of second antenna")
            antA, antB = ['ant' + ident for ident in parsed_antenna_ids.groups()]
        # Check that requested antennas are in data set
        if antA not in ant_config or antB not in ant_config:
            raise ValueError('Requested antenna pair not found in HDF5 file (wanted %s but file only contains %s)'
                             % ([antA, antB], ant_config.keys()))
        antA_config, antB_config = ant_config[antA], ant_config[antB]
        # Get antenna description strings (antenna2 is None for single-dish data)
        antenna, antenna2 = antA_config.attrs['description'], antB_config.attrs['description']
        if antenna2 == antenna:
            antenna2 = None

        # Use antenna A for pointing sensors, noise diode flags and activity sensors (used to partition data)
        sensors_group = f['MetaData/Sensors']
        ant_sensors = sensors_group['Antennas'][antA]
        ped_sensors = sensors_group['Pedestals/ped' + antA[3:]]

        # Autodetect the noise diode to use, based on which sensor shows any activity
        if not noise_diode:
            nd_fired = {}
            for nd in ('coupler', 'pin'):
                sensor = sensor_name_v2[nd + '_nd_on']
                nd_fired[nd] = np.any(ped_sensors[sensor]['value'] == '1') if has_data(ped_sensors, sensor) else False
            if np.sum(nd_fired.values()) == 1:
                noise_diode = nd_fired.keys()[nd_fired.values().index(True)]
                logger.info("Using '%s' noise diode as it is the only one firing in data set" % noise_diode)
            else:
                noise_diode = 'coupler'
                logger.info("Defaulting to '%s' noise diode (either no or both diodes are firing)" % noise_diode)
        # First try to load external noise diode models, if provided
        nd_h_model = nd_v_model = None
        if nd_models:
            nd_h_file = os.path.join(nd_models, '%s.%s.h.csv' % (antA, noise_diode))
            nd_h_model = NoiseDiodeModel(nd_h_file)
            logger.info("Loaded H noise diode model from '%s'" % (nd_h_file,))
            nd_v_file = os.path.join(nd_models, '%s.%s.v.csv' % (antA, noise_diode))
            nd_v_model = NoiseDiodeModel(nd_v_file)
            logger.info("Loaded V noise diode model from '%s'" % (nd_v_file,))
        else:
            # Now try to load models for selected noise diode
            try:
                nd_dataset = antA_config['h_' + noise_diode + '_noise_diode_model']
                nd_h_model = NoiseDiodeModel(nd_dataset[:, 0] / 1e6, nd_dataset[:, 1], **dict(nd_dataset.attrs))
                nd_dataset = antA_config['v_' + noise_diode + '_noise_diode_model']
                nd_v_model = NoiseDiodeModel(nd_dataset[:, 0] / 1e6, nd_dataset[:, 1], **dict(nd_dataset.attrs))
            except KeyError:
                # No models were found for the selected diode - quit and report any ones that are there instead
                nd_set = set([name[1:].replace('noise_diode_model', '').strip('_')
                              for name in antA_config if name.endswith('noise_diode_model')])
                nd_set.discard('')
                raise ValueError("Unknown noise diode '%s', found the following models instead: %s" %
                                 (noise_diode, list(nd_set)))

        # Get weather data
        enviro_sensors = sensors_group['Enviro']
        enviro = {}
        for quantity in ['temperature', 'pressure', 'humidity', 'wind_speed', 'wind_direction']:
            sensor = sensor_name_v2[quantity]
            # Environment sensors are optional
            if has_data(enviro_sensors, sensor):
                enviro[quantity] = remove_duplicates(enviro_sensors[sensor])

        # Get centre frequency in MHz, assuming it hasn't changed during the experiment
        rfe_sensors, center_freq_sensor = sensors_group['RFE'], 'center-frequency-hz'
        if not has_data(rfe_sensors, center_freq_sensor):
            raise ValueError("Centre frequency sensor '%s/%s' not found" % (rfe_sensors.name, center_freq_sensor))
        band_center = rfe_sensors[center_freq_sensor]['value'][0] / 1e6

        # Load correlator configuration group
        corr_config = f['MetaData/Configuration/Correlator']
        # Construct channel centre frequencies (in MHz) from DBE attributes
        num_chans = get_single_value(corr_config, 'n_chans')
        channel_bw = get_single_value(corr_config, 'bandwidth') / 1e6 / num_chans
        # Assume that lower-sideband downconversion has been used, which flips frequency axis
        # Also subtract half a channel width to get frequencies at center of each channel
        center_freqs = band_center - channel_bw * (np.arange(num_chans) - num_chans / 2 + 0.5)
        bandwidths = np.tile(np.float64(channel_bw), num_chans)
        channel_select = range(num_chans)
        sample_period = get_single_value(corr_config, 'int_time')
        dump_rate = 1.0 / sample_period
        corrconf = CorrelatorConfig(center_freqs, bandwidths, channel_select, dump_rate)

        # Figure out mapping of antennas and feeds to the correlation products involved
        # Mapping of polarisation product to pair of input labels indicating inputs that are multiplied together
        pol_to_inputpair = {'HH' : (antA + 'H', antB + 'H'), 'VV' : (antA + 'V', antB + 'V'),
                            'HV' : (antA + 'H', antB + 'V'), 'VH' : (antA + 'V', antB + 'H')}
        # Map from input label pair to correlation product index (which typically follows Miriad-style numbering)
        corrprod_map = dict([(tuple(input_pair), corr_id) for corr_id, input_pair in
                             enumerate(get_single_value(corr_config, 'bls_ordering'))])
        # Overall mapping from polarisation product to correlation product index (-1 for unavailable products)
        pol_to_corr_id = dict([(pol, corrprod_map.get(pol_to_inputpair[pol], -1)) for pol in scape_pol_if])

        # Obtain visibility data and timestamps
        data = f['Data/correlator_data']
        # Load timestamps as UT seconds since Unix epoch, and move them from start of each sample to the middle
        data_timestamps = f['Data/timestamps'].value + 0.5 * sample_period + time_offset
        # If data timestamps have problems, warn about it
        if np.any(data_timestamps < 1e9) or (len(data_timestamps) > 1 and np.diff(data_timestamps).min() == 0.0):
            logger.warning("Bad correlator timestamps (duplicates or way out of date)")
        dump_endtimes = data_timestamps + 0.5 * sample_period

        # Load pointing sensor data and interpolate it to data timestamps (storing it in radians)
        pointing = []
        for coord in ('azim', 'elev'):
            try:
                sensor = '%s-%s' % (sensor_name_v2[selected_pointing], coord)
            except KeyError:
                raise ValueError("Unknown pointing sensor group '%s' specified" % (selected_pointing,))
            if not has_data(ant_sensors, sensor):
                raise ValueError("Selected pointing sensor '%s/%s' not found" % (ant_sensors.name, sensor))
            # Ensure pointing timestamps are unique before interpolation
            original_coord = remove_duplicates(ant_sensors[sensor])
            # Linearly interpolate pointing coordinate to data timestamps
            # As long as azimuth is in natural antenna coordinates, no special angle interpolation required
            interp = PiecewisePolynomial1DFit(max_degree=1)
            interp.fit(original_coord['timestamp'], original_coord['value'])
            pointing.append(interp(data_timestamps).astype(np.float32) * np.float32(np.pi / 180.0))
        pointing = np.rec.fromarrays(pointing, names=('az', 'el'))

        # Load noise diode flags and interpolate it to data timestamps
        sensor = sensor_name_v2[noise_diode + '_nd_on']
        if has_data(ped_sensors, sensor):
            # Ensure noise diode timestamps are unique before interpolation
            nd_on = remove_duplicates(ped_sensors[sensor])
            # Do step-wise interpolation (as flag is either 0 or 1 and holds its value until it toggles)
            interp = PiecewisePolynomial1DFit(max_degree=0)
            try:
                # Assumes that flag values are '1' and '0' (or 1 and 0) for True and False, respectively
                nd_flags = nd_on['value'].astype(int)
            except ValueError:
                # Assumes that flag values are 'True' and 'False' for True and False, respectively
                nd_flags = [(1 if flag == 'True' else 0) for flag in nd_on['value']]
            interp.fit(nd_on['timestamp'], nd_flags)
            flags = np.rec.fromarrays([interp(data_timestamps).astype(bool)], names=('nd_on',))
        else:
            flags = np.rec.fromarrays([np.tile(False, data_timestamps.shape)], names=('nd_on',))

        # Use the activity sensor of Antenna A to partition the data set into scans (and to label the scans)
        activity_sensor = remove_duplicates(ant_sensors['activity'])
        activity, activity_timestamps = activity_sensor['value'], activity_sensor['timestamp']
        # Simplify the activities to derive the basic state of the antenna (slewing, scanning, tracking, stopped)
        simplify = {'scan': 'scan', 'track': 'track', 'slew': 'slew', 'scan_ready': 'slew', 'scan_complete': 'slew'}
        state = np.array([simplify.get(act, 'stop') for act in activity])
        # Cull spurious short-lived activities (e.g. track immediately preceding scan_ready)
        state_durations = np.diff(np.r_[activity_timestamps, dump_endtimes[-1]])
        non_spurious = state_durations > 0.5 * sample_period
        state, activity_timestamps = state[non_spurious], activity_timestamps[non_spurious]
        # Identify times where the state changes - these become scan boundaries
        state_changes = [n for n in xrange(len(state)) if (n == 0) or (state[n] != state[n - 1])]
        scan_labels, scan_timestamps = state[state_changes], activity_timestamps[state_changes]
        # Convert scan boundary times to sample indices
        scan_starts = dump_endtimes.searchsorted(scan_timestamps)
        scan_ends = np.r_[scan_starts[1:] - 1, len(dump_endtimes) - 1]

        # Use the target sensor of Antenna A to partition the data set into compound scans
        target_sensor = remove_duplicates(ant_sensors['target'])
        target, target_timestamps = target_sensor['value'], target_sensor['timestamp']
        # Ignore empty and repeating targets (but keep any target following an empty one, as well as first target)
        target_changes = [n for n in xrange(len(target)) if target[n] and ((n == 0) or (target[n] != target[n - 1]))]
        compscan_targets, target_timestamps = target[target_changes], target_timestamps[target_changes]
        compscan_starts = dump_endtimes.searchsorted(target_timestamps)
        compscan_ends = np.r_[compscan_starts[1:] - 1, len(dump_endtimes) - 1]

        # Associate labels with each compound scan
        label, label_timestamps = f['Markup/labels']['label'], f['Markup/labels']['timestamp']
        # Start with blank labels, with sufficient capacity in array to store longest label
        max_str_len = (max([len(l) for l in label]) + 1) if len(label) > 0 else 1
        compscan_labels = np.zeros(len(compscan_targets), dtype='S%d' % (max_str_len,))
        if label_timestamps[0] != 0.0:
            # Associate each label with the closest target on the left (i.e. preceding the label)
            associated_targets = (target_timestamps.searchsorted(label_timestamps) - 1).tolist()
            # Pick first label associated with each target
            unique_targets = [ind for ind in np.unique(associated_targets) if ind >= 0]
            label = [label[associated_targets.index(ind)] for ind in unique_targets]
            compscan_labels[unique_targets] = label

        # Create each compound scan, as indicated by range of dump indices
        compscanlist = []
        for compscan_target, compscan_start, compscan_end, compscan_label in \
            zip(compscan_targets, compscan_starts, compscan_ends, compscan_labels):
            # Identify range of scans that fall within compound scan
            first_scan = max(scan_starts.searchsorted(compscan_start, side='right') - 1, 0)
            last_scan = max(scan_starts.searchsorted(compscan_end, side='right') - 1, 0)
            compscan = range(first_scan, last_scan + 1)

            # Create each scan within compound scan, as indicated by range of dump indices
            scanlist = []
            for scan_label, start, end in zip(scan_labels[compscan], scan_starts[compscan], scan_ends[compscan]):
                start, end = max(start, compscan_start), min(end, compscan_end)
                num_times = end + 1 - start
                # Load correlation data either in float32 (single-dish) form or complex64 (interferometer) form
                # Data has already been normalised by number of samples in integration (accum_per_int)
                # Data of missing feeds are set to zero, as this simplifies the data structure (always 4 products)
                if antenna2 is None:
                    # Single-dish uses HH, VV, Re{HV}, Im{HV}
                    # Add real (0) and imag (1) indices for each polarisation product below, i.e. [Re, Re, Re, Im]
                    corr_id = [pol_to_corr_id[pol] for pol in ['HH', 'VV', 'HV', 'HV']]
                    # Extract time slice of appropriate correlation products (keeping all channels)
                    scan_data = np.dstack([(data[start:end + 1, :, bl_id, real_imag] if bl_id >= 0 else
                                            np.zeros((num_times, num_chans), np.float32))
                                           for bl_id, real_imag in zip(corr_id, np.array([0, 0, 0, 1]))])
                else:
                    corr_id = [pol_to_corr_id[pol] for pol in scape_pol_if]
                    scan_data = np.dstack([(data[start:end + 1, :, bl_id].view(np.complex64) if bl_id >= 0 else
                                            np.zeros((num_times, num_chans), np.complex64))
                                           for bl_id in corr_id])
                scanlist.append(Scan(scan_data, data_timestamps[start:end + 1], pointing[start:end + 1],
                                     flags[start:end + 1], scan_label, '%s[%d:%d]' % (filename, start, end + 1)))

            if len(scanlist) > 0:
                compscanlist.append(CompoundScan(scanlist, compscan_target, compscan_label))

        return compscanlist, experiment_id, observer, description, 'counts', \
               corrconf, antenna, antenna2, nd_h_model, nd_v_model, enviro

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  save_dataset
#--------------------------------------------------------------------------------------------------

def save_dataset(dataset, filename):
    """Save data set to HDF5 file (in format version 1).

    This will overwrite any existing file with the same name.

    Parameters
    ----------
    dataset : :class:`dataset.DataSet` object
        Data set object to save
    filename : string
        Name of output HDF5 file

    """
    with h5py.File(filename, 'w') as f:
        # Attributes at data set level
        f['/'].attrs['augment'] = 'File created by scape'
        f['/'].attrs['experiment_id'] = dataset.experiment_id
        f['/'].attrs['observer'] = dataset.observer
        f['/'].attrs['description'] = dataset.description
        f['/'].attrs['data_unit'] = dataset.data_unit
        f['/'].attrs['data_timestamps_at_sample_centers'] = True

        # Create antennas group
        ants_group = f.create_group('Antennas')
        # If antenna names follow standard pattern, name antenna groups appropriately - otherwise number them 1 and 2
        antenna_id = antenna_name_pattern.match(dataset.antenna.name)
        antA_name = ('Antenna%s' % antenna_id.groups()) if antenna_id is not None else 'Antenna1'
        # Create first antenna group
        antA_group = ants_group.create_group(antA_name)
        antA_group.attrs['description'] = dataset.antenna.description
        # Create second antenna group if the data set is interferometric
        if dataset.antenna2 is not None:
            antenna_id = antenna_name_pattern.match(dataset.antenna2.name)
            antB_name = ('Antenna%s' % antenna_id.groups()) if antenna_id is not None else 'Antenna2'
            antB_group = ants_group.create_group(antB_name)
            antB_group.attrs['description'] = dataset.antenna2.description

        # Create receiver chain groups and enviro sensors for first antenna (other antenna left blank)
        sensors_group = antA_group.create_group('Sensors')
        for quantity in ['temperature', 'pressure', 'humidity', 'wind_speed', 'wind_direction']:
            sensor = sensor_name_v1[quantity]
            # Environment sensors are optional
            if quantity in dataset.enviro:
                sensors_group.create_dataset(sensor, data=dataset.enviro[quantity], compression='gzip')
        # For now, both H and V are created, even if original dataset had only a single feed, to keep things simple
        # The correlator data for the missing feed are already zeros in this situation
        h_group, v_group = antA_group.create_group('H'), antA_group.create_group('V')
        for nd_model, pol_group in zip([dataset.nd_h_model, dataset.nd_v_model], [h_group, v_group]):
            if nd_model is not None:
                nd_data = np.column_stack((nd_model.freq * 1e6, nd_model.temp))
                nd_dataset = pol_group.create_dataset('nd_model', data=nd_data, compression='gzip')
                for key, val in vars(nd_model).iteritems():
                    if key not in ('freq', 'temp'):
                        nd_dataset.attrs[key] = val

        # Create correlator configuration group
        corrconf_group = f.create_group('Correlator')
        corrconf_group.create_dataset('center_freqs', data=dataset.corrconf.freqs * 1e6, compression='gzip')
        corrconf_group.create_dataset('bandwidths', data=dataset.corrconf.bandwidths * 1e6, compression='gzip')
        select_flags = np.tile(False, len(dataset.corrconf.freqs))
        select_flags[dataset.corrconf.channel_select] = True
        corrconf_group.create_dataset('channel_select', data=select_flags)
        corrconf_group.attrs['dump_rate_hz'] = dataset.corrconf.dump_rate
        # Simplified correlator ID lookup
        pol_to_corr_id = dict([(pol, n) for n, pol in enumerate(scape_pol_if)])

        scans_group = f.create_group('Scans')
        for compscan_ind, compscan in enumerate(dataset.compscans):
            compscan_group = scans_group.create_group('CompoundScan%d' % compscan_ind)
            compscan_group.attrs['target'] = compscan.target.description
            compscan_group.attrs['label'] = compscan.label

            for scan_ind, scan in enumerate(compscan.scans):
                scan_group = compscan_group.create_group('Scan%d' % scan_ind)
                # Always save power data in complex128 form
                corr_id = [str(pol_to_corr_id[pol]) for pol in scape_pol_if]
                complex_data = np.rec.fromarrays([scan.pol(key) for key in scape_pol_if],
                                                 names=corr_id, formats=['complex128'] * 4)
                scan_group.create_dataset('data', data=complex_data, compression='gzip')
                # Save data timestamps in milliseconds
                scan_group.create_dataset('timestamps', data=1000.0 * scan.timestamps, compression='gzip')
                # Convert contents of pointing from radians to degrees, without disturbing original
                # pylint: disable-msg=W0612
                pointing = scan.pointing.copy()
                pointing_view = pointing.view(np.float32)
                pointing_view *= np.float32(180.0 / np.pi)
                scan_group.create_dataset('pointing', data=pointing, compression='gzip')
                scan_group.create_dataset('flags', data=scan.flags, compression='gzip')
                scan_group.attrs['label'] = scan.label
