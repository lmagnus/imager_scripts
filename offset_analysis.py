#! /usr/bin/python
# Script that loads and process the data produced by analyse_point_source_scan.py.
#
# First run the analyse_point_source_scans.py script to generate the data file
# that serves as input to this script.
#
# Ludwig Schwardt
# 26 August 2009
# Lindsay Magnus
# Nov 2011

import sys
import optparse
import logging
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.projections import PolarAxes
from matplotlib.dates import date2num,HourLocator, DateFormatter, drange,epoch2num
import datetime
import katpoint
from katpoint import rad2deg, deg2rad

def angle_wrap(angle, period=2.0 * np.pi):
    """Wrap angle into the interval -*period* / 2 ... *period* / 2."""
    return (angle + 0.5 * period) % period - 0.5 * period

# These fields contain strings, while the rest of the fields are assumed to contain floats
string_fields = ['dataset', 'target', 'timestamp_ut', 'data_unit']
# Create a date/time string for current time
now = time.strftime('%Y-%m-%d_%Hh%M')

# Parse command-line options and arguments
parser = optparse.OptionParser(usage="%prog [options] <data file>",
                               description="Loads and processes pointing data")
parser.add_option('-n', '--no-stats', dest='use_stats', action='store_false', default=True,
                  help="Ignore uncertainties of data points during fitting")
# Minimum pointing uncertainty is arbitrarily set to 1e-12 degrees, which corresponds to a maximum error
# of about 10 nano-arcseconds, as the least-squares solver does not like zero uncertainty
parser.add_option('-m', '--min-rms', type='float', default=np.sqrt(2) * 60. * 1e-12,
                  help="Minimum uncertainty of data points, expressed as the sky RMS in arcminutes")
(opts, args) = parser.parse_args()


# Set up logging: logging everything (DEBUG & above)
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format="%(levelname)s: %(message)s")
logger = logging.root
logger.setLevel(logging.DEBUG)

observation_sources = katpoint.Catalogue()
observation_sources.add(file('/home/kat/comm/catalogues/sources_pnt.csv'))
all_data = []
p_models = []
all_antennas = []
all_xranges = []
all_dates = []
all_mean = []
all_std = []
fig = plt.figure()
for i in range(7):
	old_model = file('./pointing-models/ant'+str(i+1)+'.pm.csv').readline().strip()
	p_models.append(katpoint.PointingModel(old_model, strict=False))
	filename = './ant'+str(i+1)+'.csv'
	# Load data file in one shot as an array of strings
	data = np.loadtxt(filename, dtype='string', comments='#', delimiter=', ')
# Interpret first non-comment line as header
	fields = data[0].tolist()
#remove all the other field names from the concatenated csv files
	data = data[data[:,0] != 'dataset',:]
# By default, all fields are assumed to contain floats
	formats = np.tile(np.float, len(fields))
# The string_fields are assumed to contain strings - use data's string type, as it is of sufficient length
	formats[[fields.index(name) for name in string_fields if name in fields]] = data.dtype
# Convert to heterogeneous record array
	data = np.rec.fromarrays(data[0:].transpose(), dtype=zip(fields, formats))
# Load antenna description string from first line of file and construct antenna object from it
	all_data.append(data)
	all_antennas.append(katpoint.Antenna(file(filename).readline().strip().partition('=')[2]))
# Process the data below
	all_xranges.append([(date2num(datetime.datetime.strptime(t[0:19],'%Y-%m-%d %H:%M:%S')),30.0/(24*60*60)) for t in all_data[i]['timestamp_ut']])
	all_dates.append([datetime.datetime.strptime(t[0:19],'%Y-%m-%d %H:%M:%S').toordinal() for t in all_data[i]['timestamp_ut']])
#now select the two date reqions and create the pointing models
	ind_last = np.array(all_dates[i]) > datetime.datetime(2011, 10,1,0,0,0).toordinal()
	ind_first =( np.array(all_dates[i]) > datetime.datetime(2011, 6,1,0,0,0).toordinal()) *( np.array(all_dates[i]) < datetime.datetime(2011, 10,1,0,0,0).toordinal() ) 
	ind_all = np.array(all_dates[i]) > datetime.datetime(2011, 6,1,0,0,0).toordinal()
	#all_data[i] = all_data[i][ind_all]
	all_mean_azi = []
	all_mean_el = []
	all_std_azi = []
	all_std_el = []
	for w in range(10):
		ind = (all_data[i]['wind_speed']<(w+1)) * (all_data[i]['wind_speed']>(w))
		d_azi = np.array(all_data[i]['delta_azimuth'][ind])
		d_el = np.array(all_data[i]['delta_elevation'][ind])
		print(i,'mean',d_azi.mean(),d_el.mean())
		print(i,'std',d_azi.std(),d_el.std())
		all_mean_azi.append(d_azi.mean())
		all_mean_el.append(d_el.mean())
		all_std_azi.append(d_azi.std())
		all_std_el.append(d_el.std())
	fig.add_subplot(7,2,(i*2)+1)
	plt.errorbar(range(10),all_mean_azi,np.sqrt(all_std_azi))
	plt.errorbar(range(10),all_mean_el,np.sqrt(all_std_el))
	fig.add_subplot(7,2,(i*2)+2)
	plt.plot(all_std_azi)
	plt.plot(all_std_el)
