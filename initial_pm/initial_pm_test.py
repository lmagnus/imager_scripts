#! /usr/bin/python
# test script for antenna 3
#
# Script to read the last file observed (assuming a raster scan for pointing)
# Then determine the two term pointing model [P1, P7].
#
# Ludwig Schwardt
# 26 August 2009
# Lindsay Magnus
# Nov 2011
# Lindsay Magnus
# Jan 2013


import numpy as np
import katpoint
from katpoint import rad2deg, deg2rad
import katarchive
import os 
def angle_wrap(angle, period=2.0 * np.pi):
    """Wrap angle into the interval -*period* / 2 ... *period* / 2."""
    return (angle + 0.5 * period) % period - 0.5 * period

# get the last observed file in the archive
list = katarchive.search_archive()
f = list[0]
fn = f.path_to_file

# process the file using APSS
rng = '200,800'
a = 3
args = " -a A%iA%i -b -f '%s'  -o ant3  %s" % (a,a,rng,fn,)
print args
os.system('/home/kat/scripts/reduction/analyse_point_source_scans.py'+args)

# Read in the data
# These fields contain strings, while the rest of the fields are assumed to contain floats
string_fields = ['dataset', 'target', 'timestamp_ut', 'data_unit']

filename = 'ant3.csv'
# Load data file in one shot as an array of strings
data = np.loadtxt(filename, dtype='string', comments='#',skiprows=1, delimiter=', ')
# Interpret first non-comment line as header
fields = data[0].tolist()
data = data[1:]
#remove all the other field names from the concatenated csv files
#data = data[data[:,0] != 'dataset',:]
# By default, all fields are assumed to contain floats
formats = np.tile(np.float, len(fields))
# The string_fields are assumed to contain strings - use data's string type, as it is of sufficient length
formats[[fields.index(name) for name in string_fields if name in fields]] = data.dtype
# Convert to heterogeneous record array
data = np.rec.fromarrays(data[0:].transpose(), dtype=zip(fields, formats))

# Create a blank PM and then fit
pm = katpoint.PointingModel()
az = np.deg2rad(data['azimuth'])
d_az = np.deg2rad(data['delta_azimuth'])
el = np.deg2rad(data['elevation'])
d_el = np.deg2rad(data['delta_elevation'])
pm.fit(az,el,d_az,d_el,enabled_params = [1,7])

# Print the text for the PM file
print 'Insert the next line into the PM file'
print pm.description

