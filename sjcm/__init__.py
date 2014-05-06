"""The Single-dish Continuum Analysis PackagE."""

import logging
import sys

# Setup library logger, and suppress spurious logger messages via a null handler
class _NullHandler(logging.Handler):
    def emit(self, record):
        pass
_logger = logging.getLogger("scape")
_logger.addHandler(_NullHandler())
# Set up basic logging if it hasn't been done yet, in order to display error messages and such
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format="%(levelname)s: %(message)s")

# Most operations are directed through the data set
from .dataset import DataSet
from .compoundscan import CorrelatorConfig, CompoundScan
from .scan import Scan

def _module_found(name, message):
    """Check whether module *name* imports, otherwise log a warning *message*."""
    try:
        __import__(name)
        return True
    except ImportError:
        _logger.warn(message)
        return False

# Check if matplotlib is present, otherwise skip plotting routines
if _module_found('matplotlib', 'Matplotlib was not found - plotting will be disabled'):
    from .plots_canned import plot_xyz, extract_xyz_data, extract_scan_data, \
                              plot_spectrum, plot_waterfall, plot_spectrogram, plot_fringes, \
                              plot_compound_scan_in_time, plot_compound_scan_on_target, \
                              plot_data_set_in_mount_space, plot_measured_beam_pattern

# Check if pyfits is present, otherwise skip FITS creation routines
if _module_found('pyfits', 'PyFITS was not found - FITS creation will be disabled'):
    from .plots_basic import save_fits_image

def _extract_version():
    """Extract module version from pkg_resources structure."""
    try:
        import pkg_resources
        dist = pkg_resources.get_distribution("scape")
        # ver needs to be a list since tuples in Python <= 2.5 don't have a .index method
        ver = list(dist.parsed_version)
        return "r%d" % int(ver[ver.index("*r") + 1])
    except (ImportError, pkg_resources.DistributionNotFound, ValueError, IndexError, TypeError):
        return "unknown"
__version__ = _extract_version()

# Clean up module namespace to make it easier to spot the useful parts
del logging, sys, _NullHandler, _module_found, _extract_version
