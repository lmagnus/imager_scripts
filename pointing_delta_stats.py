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
	antenna = katpoint.Antenna(file(filename).readline().strip().partition('=')[2])
# Process the data below
	xranges = [(date2num(datetime.datetime.strptime(t[0:19],'%Y-%m-%d %H:%M:%S')),30.0/(24*60*60)) for t in all_data[i]['timestamp_ut']]
	#fig = plt.figure(0)
	#ax = fig.add_axes([0,0,1,1],projection='polar')
	#ax.scatter(np.pi/2. - np.deg2rad(all_data[i]['azimuth']), np.pi/2. - np.deg2rad(all_data[i]['elevation']))
	#ax.broken_barh(xranges, (0, 1), facecolors='blue')
	#ax.plot(xranges,all_data[i]['beam_width_I'],'o')
	#ax.xaxis.set_major_formatter(DateFormatter('%d/%m/%Y\n%H:%M:%S'))
	#ax.set_xlim(date2num(datetime.date(2010,06,30)),date2num(datetime.date(2011,12,30)))

#pointing model analysis
#pointing model as a function of wind speed
	for w in range(1):
		ind = all_data[i]['wind_speed']<(2*w)
		az = np.deg2rad(all_data[i]['azimuth'][ind])
		d_az = np.deg2rad(all_data[i]['delta_azimuth'][ind])
		el = np.deg2rad(all_data[i]['elevation'][ind])
		d_el = np.deg2rad(all_data[i]['delta_elevation'][ind])
		pm = katpoint.PointingModel()
		pm.fit(az,el,d_az,d_el)
		model_delta_az, model_delta_el = pm.offset(az, el)
		residual_az = d_az - model_delta_az
		residual_el = d_el - model_delta_el
		residual_xel = residual_az * np.cos(el)
		abs_sky_error = rad2deg(np.sqrt(residual_xel ** 2 + residual_el ** 2)) * 60. #in arcseconds
		sky_rms = np.sqrt(np.mean(abs_sky_error ** 2))
		#print(i+1,2*w,sky_rms)
		print(pm.description)

	for t in set(all_data[i]['target']):
		kat_target = observation_sources[t]
		if kat_target == None: continue
		ind = (all_data[i]['target'] == t) * (all_data[i]['wind_speed']<2)
		N = len(all_data[i]['target'][ind])
		az = np.deg2rad(all_data[i]['azimuth'][ind])
		el = np.deg2rad(all_data[i]['elevation'][ind])
		d_az = all_data[i]['delta_azimuth'][ind]
		d_el = all_data[i]['delta_elevation'][ind]
		#if N > 20: print(i,t,N,np.std(d_az)*3600.,np.std(d_el)*3600.)
#plt.show()


