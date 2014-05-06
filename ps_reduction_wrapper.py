#!/usr/bin/python

import katarchive
import katfile
import os
import re

f = katarchive.search_archive(startdate='1/1/2013',description='point')
files = [t.metadata.FileLocation+'/'+t.metadata.Filename for t in f if re.search('(hour point)|(Point.*scan)',t.metadata.Description) != None]

for fi in files:
	h5 = katfile.open(fi)
	ants = h5.ants
	if h5.version.startswith('1'):
		rng = '90,424'
	else:
		rng = '200,400'
	h5.file.close()
	dataset_name = os.path.splitext(os.path.basename(fi))[0]
	for ant in ants:
		a = ant.name[3:]
		args = ' -a A'+a+'A'+a+' -b -f '+rng+'  -o '+ant.name+'_'+dataset_name+' '+fi
		os.system('/home/kat/scripts/reduction/analyse_point_source_scans.py'+args)


