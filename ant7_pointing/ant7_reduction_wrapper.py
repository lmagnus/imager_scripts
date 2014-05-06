#!/usr/bin/python

import katarchive
import katfile
import os
import numpy as np

f = katarchive.search_archive(description='point',startdate='01/06/2012')
a = np.array([True if i.metadata.Antennas.count('ant7')>0 else False for i in f])
files = np.array(f)[a]
for file in files:
	f = file.path_to_file
	dataset_name = os.path.splitext(os.path.basename(f))[0]
	args = ' -a A7A7 -b -f 200,400  -o ant7'+dataset_name+' '+f
	os.system('/home/kat/scripts/reduction/analyse_point_source_scans.py'+args)


