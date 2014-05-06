#!/usr/bin/python

import katarchive
import os

f = katarchive.get_archived_products('1355547913.h5')

for ant in range(1,8):
	a = str(ant)
	dataset_name = os.path.splitext(os.path.basename(f[0]))[0]
	args = ' -a "A'+a+'A'+a+'" --keep="ant'+a+'_keep.csv" --katfile  -m 25 -b -f "200,300"  -o "ant'+a+'_'+dataset_name+'" '+f[0]
	os.system('/home/kat/scripts/reduction/analyse_point_source_scans.py'+args)


