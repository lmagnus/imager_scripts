#!/usr/bin/python

import katarchive
import katfile
import os

#files = katarchive.list_katfiles(start='30/7/2011',end='1/8/2011',description='point source scan')
files = katarchive.list_katfiles(description="point source scan")
for file in files:
	h5 = katfile.open(file)
	ants = h5.ants
	if h5.version.startswith('1'):
		rng = '90,424'
	else:
		rng = '200,400'
	h5.file.close()
	dataset_name = os.path.splitext(os.path.basename(file))[0]
	for ant in ants:
		a = ant.name[3:]
		args = ' -a A'+a+'A'+a+' -b -f '+rng+'  -o '+ant.name+'_'+dataset_name+' '+file
		os.system('./analyse_point_source_scans.py'+args)


