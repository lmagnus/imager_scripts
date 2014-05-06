"""Container for the data of a single scan.

A *scan* is the *minimal amount of data taking that can be commanded at the
script level,* which corresponds to the *subscan* of the ALMA Science Data Model.
This includes a single linear sweep across a source, one line in an OTF map, a
single pointing for Tsys measurement, a noise diode on-off measurement, etc. It
contains several *integrations*, or correlator samples, and forms part of an
overall *compound scan*.

This module provides the :class:`Scan` class, which encapsulates all data
and actions related to a single scan across a point source, or a single
scan at a certain pointing. All actions requiring more than one scan are
grouped together in :class:`compoundscan.CompoundScan` instead.

"""

import numpy as np
import time

from katpoint import rad2deg

# Order of polarisation terms on last dimension of correlation data array:
# Standard scape order for real single-dish data
scape_pol_sd = ['HH', 'VV', 'ReHV', 'ImHV']
# Standard scape order for complex interferometer (i.e. single-baseline) data
scape_pol_if = ['HH', 'VV', 'HV', 'VH']
# Mount coherencies
mount_coh = ['VV', 'VH', 'HV', 'HH']
# Sky coherencies
sky_coh = ['XX', 'XY', 'YX', 'YY']
# Stokes parameters
stokes = ['I', 'Q', 'U', 'V']

#--------------------------------------------------------------------------------------------------
#--- CLASS :  Scan
#--------------------------------------------------------------------------------------------------

class Scan(object):
    """Container for the data of a single scan.

    The main data member of this class is the 3-D :attr:`data` array, which
    stores correlation measurements as a function of time, frequency channel/band
    and polarisation index. It is assumed that each antenna has two orthogonal
    linear feeds, labelled *H* (horizontal) and *V* (vertical). The data array
    takes one of two forms:

    - For single-baseline interferometer data, the array contains complex-valued
      cross-correlation data of the form (HH, VV, HV, VH). In this case, the
      first letter of each pair indicates the feed on the first antenna involved
      in the correlation, while the second letter of the pair indicates the feed
      on the second antenna. Therefore, *HH* is the correlation of the signal from
      the *H* feed on antenna 1 with the signal from the *H* feed on antenna 2.
      This matches the correlator format on purpose.

    - For single-dish data, the array contains real-valued autocorrelation data
      of the form (HH, VV, Re{HV}, Im{HV}). *HH* is the non-negative power
      measured in the *H* feed of the first (and only) antenna. Similarly, *VV*
      is the power measured in the *V* feed. *HV* is the complex-valued
      correlation between the *H* and *V* feed signals on the *same* antenna.
      Since *HV* and *VH* are complex conjugates for a single dish, it is only
      necessary to store one of them. This scheme is twice as efficient as the
      correlator format used for interferometer data.

    The class also stores pointing data (azimuth/elevation/rotator angles),
    timestamps and flags, which all vary as a function of time. The number of
    time samples are indicated by *T* and the number of frequency channels/bands
    are indicated by *F* below.

    Parameters
    ----------
    data : float64 or complex128 array, shape (*T*, *F*, 4)
        Correlation measurements. For single-dish data this is real-valued with
        polarisation order (HH, VV, Re{HV}, Im{HV}). For interferometer data this
        is complex-valued with polarisation order (HH, VV, HV, VH).
    timestamps : float64 array, shape (*T*,)
        Sequence of timestamps, one per integration (in UTC seconds since epoch).
        These timestamps should be in the *middle* of each integration.
    pointing : float32 record array, shape (*T*,)
        Pointing coordinates for *first* antenna, with one record per integration
        (in radians). The real-valued fields are 'az', 'el' and optionally 'rot',
        for azimuth, elevation and rotator angle, respectively. The pointing
        should be valid for the *middle* of each integration.
    flags : bool record array, shape (*T*,)
        Flags, with one record per integration. The field names correspond to
        the flag names.
    label : string
        Scan label, used to distinguish e.g. normal and cal scans
    path : string
        Filename or HDF5 path from which scan was loaded
    target_coords : real array, shape (2, *T*), optional
        Coordinates on projected plane, with target as reference, in radians
    parangle : real array, shape (*T*,), optional
        Parallactic angle at the target, in radians
    baseline : :class:`fitting.Polynomial1DFit` object, optional
        Object that describes fitted baseline (base power level as a function of
        time, not to be confused with interferometer or spectral baseline)
    compscan : :class:`CompoundScan` object, optional
        Parent compound scan of which this scan forms a part

    Attributes
    ----------
    has_autocorr : bool
        True if the scan contains autocorrelation data in real single-dish format

    """
    def __init__(self, data, timestamps, pointing, flags, label, path,
                 target_coords=None, parangle=None, baseline=None, compscan=None):
        self.data = data
        self.timestamps = timestamps
        self.pointing = pointing
        self.flags = flags
        self.label = label
        self.path = path
        self.target_coords = target_coords
        self.parangle = parangle
        self.baseline = baseline
        self.has_autocorr = np.isrealobj(self.data)
        self.compscan = compscan

    def __eq__(self, other):
        """Equality comparison operator."""
        # Because of conversion to degrees and back during saving and loading, the last (8th)
        # significant digit of the float32 pointing values may change - do approximate comparison.
        # Since pointing is used to calculate target coords, this is also only approximately equal.
        # Timestamps and pointing are also converted to and from the start and middle of each sample,
        # which causes extra approximateness... (pointing should be OK down to 5 arcsecond level,
        # and time should be OK down to microseconds)
        return (self.has_autocorr == other.has_autocorr) and np.all(self.data == other.data) and \
               np.all(self.flags == other.flags) and (self.label == other.label) and \
               np.allclose(self.timestamps, other.timestamps, rtol=0, atol=1e-6) and \
               np.allclose(self.pointing.view(np.float32), other.pointing.view(np.float32), rtol=1e-6, atol=0) and \
               np.allclose(self.target_coords, other.target_coords, rtol=1e-7, atol=0) and \
               np.allclose(self.parangle, other.parangle, rtol=1e-7, atol=0)

    def __ne__(self, other):
        """Inequality comparison operator."""
        return not self.__eq__(other)

    def __str__(self):
        """Verbose human-friendly string representation of scan object."""
        mean_az = rad2deg(self.pointing['az'].mean())
        mean_el = rad2deg(self.pointing['el'].mean())
        return "%sdata=%s, start='%s', az=%.1f, el=%.1f, path='%s'" % \
               ("'%s', " % (self.label,) if self.label else '', self.data.shape,
                time.strftime('%Y/%m/%d %H:%M:%S', time.gmtime(self.timestamps[0])),
                mean_az, mean_el, self.path)

    def __repr__(self):
        """Short human-friendly string representation of scan object."""
        return "<scape.Scan '%s' data=%s at 0x%x>" % (self.label, self.data.shape, id(self))

    def calc_cached_coords(self, target=None, antenna=None):
        """Calculate cached coordinates, based on target and antenna objects.

        This calculates the coordinates of the antenna pointing on a projected
        plane relative to the target object, as well as the parallactic angle
        (the position angle of the antenna's vertical on the sky at the target).
        The results are cached in the :class:`Scan` object for convenience.

        Parameters
        ----------
        target : :class:`katpoint.Target` object, optional
            Target object which is scanned across, obtained from associated
            :class:`CompoundScan` by default
        antenna : :class:`katpoint.Antenna` object, optional
            Antenna object for antenna that does scanning, obtained from
            associated :class:`DataSet` by default

        Raises
        ------
        ValueError
            If target or antenna is not specified, and no defaults are available

        """
        if target is None:
            if self.compscan:
                target = self.compscan.target
            else:
                raise ValueError('Please specify a target object')
        if antenna is None and self.compscan and self.compscan.dataset:
            antenna = self.compscan.dataset.antenna
        # Copy scan (az, el) coordinates
        az, el = self.pointing['az'][:], self.pointing['el'][:]
        # Fix over-the-top elevations (projections can only handle elevations in range +- 90 degrees)
        over_the_top = (el > np.pi / 2.0) & (el < np.pi)
        az[over_the_top] += np.pi
        el[over_the_top] = np.pi - el[over_the_top]
        target_x, target_y = target.sphere_to_plane(az, el, self.timestamps, antenna)
        self.target_coords = np.vstack((target_x, target_y))
        self.parangle = target.parallactic_angle(self.timestamps, antenna)

    def pol(self, key):
        """Extract a specific polarisation term from correlation data.

        This extracts a single polarisation term from the correlation data (which
        usually contains four polarisation terms). Three different polarisation
        representations are supported:

        - **Mount coherencies** (*VV*, *VH*, *HV*, *HH*). These are correlations
          between the signals of specific feeds on each antenna, and match the
          correlator output and data file format. These correlations are measured
          in the coordinate system of the (az-el) mount, and rotate on the sky
          via the parallactic angle. This is the native format of :mod:`scape`.

        - **Sky coherencies** (*XX*, *XY*, *YX*, *YY*). These correlations are
          measured with respect to an *xy* coordinate system fixed on the sky,
          where *x* points toward the North Celestial Pole and *y* points toward
          the East (left of *x* when looking up at the target). They differ from
          mount coherencies by the parallactic rotation.

        - **Stokes parameters** (*I*, *Q*, *U*, *V*). These parameters are
          obtained by linear transformation of the sky coherencies, and are based
          on the same coordinate system fixed on the sky. The coordinate system
          is important for the correct interpretation of position angles.

        Parameters
        ----------
        key : {'VV', 'VH', 'HV', 'HH', 'ReHV', 'ImHV', 'XX', 'XY', 'YX', 'YY', 'I', 'Q', 'U', 'V'}
            Polarisation term to extract

        Returns
        -------
        pol_data : float64 or complex128 array, shape (*T*, *F*)
            Polarisation term as a function of time and frequency

        Raises
        ------
        KeyError
            If *key* is not one of the allowed polarisation terms

        """
        # pylint: disable-msg=C0103
        # Mount coherencies are the easiest to extract - simply pick correct subarray (mostly)
        if key in scape_pol_sd:
            # Re{HV} and Im{HV} are not explicitly stored in interferometer data - extract them from HV instead
            if not self.has_autocorr and key in ('ReHV', 'ImHV'):
                HV = self.data[:, :, scape_pol_if.index('HV')]
                return HV.real if key == 'ReHV' else HV.imag
            else:
                return self.data[:, :, scape_pol_sd.index(key)]
        elif key in scape_pol_if:
            # HV and VH are not explicitly stored in single-dish data - calculate them instead
            if self.has_autocorr and key in ('HV', 'VH'):
                ReHV, ImHV = self.data[:, :, scape_pol_sd.index('ReHV')], self.data[:, :, scape_pol_sd.index('ImHV')]
                return ReHV + 1j * ImHV if key == 'HV' else ReHV - 1j * ImHV
            else:
                return self.data[:, :, scape_pol_if.index(key)]
        # For convenience, define mount coherencies as views of data array (no copying involved)
        HH, VV = self.data[:, :, scape_pol_sd.index('HH')], self.data[:, :, scape_pol_sd.index('VV')]
        ReHV, ImHV = self.data[:, :, scape_pol_sd.index('ReHV')], self.data[:, :, scape_pol_sd.index('ImHV')]
        HV, VH = ReHV, ImHV
        # The rest of the polarisation terms are sky-based, and need parallactic angle correction.
        # The mount rotation on the sky is equivalent to a *negative* parallactic angle rotation, and
        # is compensated for by rotating the data through the *positive* parallactic angle itself.
        # The implementation of the rest of the terms are also verbose and explicit to keep things fast.
        # A more generic and compact implementation tends to be slow, as it computes large arrays
        # just to throw it away again without being used.
        if key in sky_coh:
            # Form trig elements of coherency transformation matrix:
            # [XX]   [cc -cs -sc  ss] [VV]
            # [XY] = [cs  cc -ss -sc] [VH]
            # [YX]   [sc -ss  cc -cs] [HV]
            # [YY]   [ss  sc  cs  cc] [HH]
            cospa, sinpa = np.cos(self.parangle), np.sin(self.parangle)
            cc, cs = np.expand_dims(cospa * cospa, 1), np.expand_dims(cospa * sinpa, 1)
            if self.has_autocorr:
                if key == 'XX':
                    return HH + (cc * (VV - HH) - 2 * cs * ReHV)
                elif key == 'YY':
                    return (2 * cs * ReHV - cc * (VV - HH)) + VV
                elif key == 'XY':
                    return cs * (VV - HH) + (2 * cc - 1) * ReHV - 1j * ImHV
                elif key == 'YX':
                    return cs * (VV - HH) + (2 * cc - 1) * ReHV + 1j * ImHV
            else:
                if key == 'XX':
                    return HH + (cc * (VV - HH) - cs * (VH + HV))
                elif key == 'YY':
                    return (cs * (VH + HV) - cc * (VV - HH)) + VV
                elif key == 'XY':
                    return (cs * (VV - HH) + cc * (VH + HV)) - HV
                elif key == 'YX':
                    return (cs * (VV - HH) + cc * (VH + HV)) - VH
        elif key in stokes:
            # Form trig elements of Mueller rotation matrix:
            # [I]   [1  0  0  0] [1  0  0  1] [VV]
            # [Q] = [0  c -s  0] [1  0  0 -1] [VH]
            # [U]   [0  s  c  0] [0  1  1  0] [HV]
            # [V]   [0  0  0  1] [0 -j  j  0] [HH]
            if key in ('Q', 'U'):
                cos2pa = np.expand_dims(np.cos(2.0 * self.parangle), 1)
                sin2pa = np.expand_dims(np.sin(2.0 * self.parangle), 1)
            if self.has_autocorr:
                if key == 'I':
                    return HH + VV
                elif key == 'Q':
                    return cos2pa * (VV - HH) - sin2pa * 2 * ReHV
                elif key == 'U':
                    return sin2pa * (VV - HH) + cos2pa * 2 * ReHV
                elif key == 'V':
                    return -2 * ImHV
            else:
                if key == 'I':
                    return HH + VV
                elif key == 'Q':
                    return cos2pa * (VV - HH) - sin2pa * (VH + HV)
                elif key == 'U':
                    return sin2pa * (VV - HH) + cos2pa * (VH + HV)
                elif key == 'V':
                    return 1j * (HV - VH)
        raise KeyError("Polarisation key should be one of %s" % list(set(scape_pol_sd + scape_pol_if + sky_coh + stokes)),)

    def swap_h_and_v(self, antenna=0):
        """Swap around H and V polarisations (feeds) in the correlation data.

        Parameters
        ----------
        antenna : {0, 1, 2}
            The antenna whose feeds are to be swapped (0 means both antennas)

        """
        # Single-dish only has one antenna, so "both" antennas must swap their feeds around
        if self.has_autocorr:
            hh_vv = [scape_pol_sd.index('HH'), scape_pol_sd.index('VV')]
            self.data[:, :, hh_vv] = self.data[:, :, np.flipud(hh_vv)]
            # The imaginary part of HV changes sign, as HV => VH (its conjugate)
            self.data[:, :, scape_pol_sd.index('ImHV')] *= -1.0
        else:
            orig = [scape_pol_if.index(p) for p in ('HH', 'VV', 'HV', 'VH')]
            if antenna == 1:
                self.data[:, :, [scape_pol_if.index(p) for p in ('VH', 'HV', 'VV', 'HH')]] = self.data[:, :, orig]
            elif antenna == 2:
                self.data[:, :, [scape_pol_if.index(p) for p in ('HV', 'VH', 'HH', 'VV')]] = self.data[:, :, orig]
            else:
                self.data[:, :, [scape_pol_if.index(p) for p in ('VV', 'HH', 'VH', 'HV')]] = self.data[:, :, orig]
        return self

    def select(self, timekeep=None, freqkeep=None, copy=False):
        """Select a subset of time and frequency indices in data matrix.

        This creates a new :class:`Scan` object that contains a subset of the
        rows and columns of the data matrix. This allows time samples and/or
        frequency channels/bands to be discarded. If *copy* is False, the data
        is selected via a masked array or view, and the returned object is a
        view on the original data. If *copy* is True, the data matrix and all
        associated coordinate vectors are reduced to a smaller size and copied.

        Parameters
        ----------
        timekeep : sequence of bools or ints or None, optional
            Sequence of indicators of which time samples to keep (either integer
            indices or booleans that are True for the values to be kept). The
            default is None, which keeps everything.
        freqkeep : sequence of bools or ints or None, optional
            Sequence of indicators of which frequency channels/bands to keep
            (either integer indices or booleans that are True for the values to
            be kept). The default is None, which keeps everything.
        copy : {False, True}, optional
            True if the new scan is a copy, False if it is a view

        Returns
        -------
        scan : :class:`Scan` object
            Scan with reduced data matrix (either masked array or smaller copy)

        Raises
        ------
        IndexError
            If time or frequency selection is out of range

        """
        # Check time selection against data array size
        if timekeep is not None:
            if np.asarray(timekeep).dtype == 'bool':
                if len(timekeep) != self.data.shape[0]:
                    raise IndexError('Length of time selection mask (%d) differs from data shape (%d)' %
                                     (len(timekeep), self.data.shape[0]))
            else:
                if (np.asarray(timekeep).min() < 0) or (np.asarray(timekeep).max() >= self.data.shape[0]):
                    raise IndexError('Selected time indices out of range (should be 0..%d, but are %d..%d)' %
                                     (self.data.shape[0] - 1, np.asarray(timekeep).min(), np.asarray(timekeep).max()))
        # Check frequency selection against data array size
        if freqkeep is not None:
            if np.asarray(freqkeep).dtype == 'bool':
                if len(freqkeep) != self.data.shape[1]:
                    raise IndexError('Length of frequency selection mask (%d) differs from data shape (%d)' %
                                     (len(freqkeep), self.data.shape[1]))
            else:
                if (np.asarray(freqkeep).min() < 0) or (np.asarray(freqkeep).max() >= self.data.shape[1]):
                    raise IndexError('Selected frequency channels out of range (should be 0..%d, but are %d..%d)' %
                                     (self.data.shape[1] - 1, np.asarray(freqkeep).min(), np.asarray(freqkeep).max()))
        # Use advanced indexing to create a smaller copy of the data matrix
        if copy:
            # If data matrix is kept intact, make a straight copy - probably faster
            if (timekeep is None) and (freqkeep is None):
                selected_data = self.data.copy()
                timekeep = np.arange(self.data.shape[0])
                freqkeep = np.arange(self.data.shape[1])
            else:
                # Convert boolean selection vectors (and None) to indices
                if timekeep is None:
                    timekeep = np.arange(self.data.shape[0])
                elif np.asarray(timekeep).dtype == 'bool':
                    timekeep = np.asarray(timekeep).nonzero()[0]
                if freqkeep is None:
                    freqkeep = np.arange(self.data.shape[1])
                elif np.asarray(freqkeep).dtype == 'bool':
                    freqkeep = np.asarray(freqkeep).nonzero()[0]
                selected_data = self.data[np.atleast_2d(timekeep).transpose(), np.atleast_2d(freqkeep), :]
            target_coords = self.target_coords[:, timekeep] if self.target_coords is not None else None
            parangle = self.parangle[timekeep] if self.parangle is not None else None
            return Scan(selected_data, self.timestamps[timekeep], self.pointing[timekeep], self.flags[timekeep],
                        self.label, self.path, target_coords, parangle, self.baseline, self.compscan)
        # Create a shallow view of data matrix via a masked array or view
        else:
            # If data matrix is kept intact, rather just return a view instead of masked array
            if (timekeep is None) and (freqkeep is None):
                selected_data = self.data
            else:
                # Normalise the selection vectors to select elements via bools instead of indices
                if timekeep is None:
                    timekeep1d = np.tile(True, self.data.shape[0])
                else:
                    timekeep1d = np.tile(False, self.data.shape[0])
                    timekeep1d[timekeep] = True
                if freqkeep is None:
                    freqkeep1d = np.tile(True, self.data.shape[1])
                else:
                    freqkeep1d = np.tile(False, self.data.shape[1])
                    freqkeep1d[freqkeep] = True
                # Create 3-D mask matrix of same shape as data, with rows and columns masked
                timekeep3d = np.atleast_3d(timekeep1d).transpose((1, 0, 2))
                freqkeep3d = np.atleast_3d(freqkeep1d).transpose((0, 1, 2))
                polkeep3d = np.atleast_3d([True, True, True, True]).transpose((0, 2, 1))
                keep3d = np.kron(timekeep3d, np.kron(freqkeep3d, polkeep3d))
                selected_data = np.ma.array(self.data, mask=~keep3d)
            return Scan(selected_data, self.timestamps, self.pointing, self.flags, self.label,
                        self.path, self.target_coords, self.parangle, self.baseline, self.compscan)
