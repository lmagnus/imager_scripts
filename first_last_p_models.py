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
	pm = katpoint.PointingModel()
	az = np.deg2rad(all_data[i]['azimuth'][ind_first])
	d_az = np.deg2rad(all_data[i]['delta_azimuth'][ind_first])
	el = np.deg2rad(all_data[i]['elevation'][ind_first])
	d_el = np.deg2rad(all_data[i]['delta_elevation'][ind_first])
	pm.fit(az,el,d_az,d_el,enabled_params =[1,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22])
#now last 
	pm2 = katpoint.PointingModel()
        az2 = np.deg2rad(all_data[i]['azimuth'][ind_last])
        d_az2 = np.deg2rad(all_data[i]['delta_azimuth'][ind_last])
        el2 = np.deg2rad(all_data[i]['elevation'][ind_last])
        d_el2 = np.deg2rad(all_data[i]['delta_elevation'][ind_last])
        pm2.fit(az2,el2,d_az2,d_el2,enabled_params =[1,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22])
#now all
	pm3 = katpoint.PointingModel()
	az3 = np.deg2rad(all_data[i]['azimuth'][ind_all])
	d_az3 = np.deg2rad(all_data[i]['delta_azimuth'][ind_all])
	el3 = np.deg2rad(all_data[i]['elevation'][ind_all])
	d_el3 = np.deg2rad(all_data[i]['delta_elevation'][ind_all])
	pm3.fit(az3,el3,d_az3,d_el3,enabled_params = [1,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22])
# now calculate the residuals
	model_delta_az, model_delta_el = pm.offset(az2, el2)
	residual_az = d_az2 - model_delta_az
	residual_el = d_el2 - model_delta_el
	residual_xel = residual_az * np.cos(el2)
	abs_sky_error = rad2deg(np.sqrt(residual_xel ** 2 + residual_el ** 2)) * 60. #in arcseconds
	sky_rms = np.sqrt(np.mean(abs_sky_error ** 2))
	print(p_models[i].description)
	print(pm.description)
	print(pm2.description)
	print(pm3.description)
	print(sky_rms)
	print((pm.params - p_models[i].params)/p_models[i].params * 100)
	print((pm2.params - pm.params)/pm.params * 100)
