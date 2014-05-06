#!/usr/bin/python
#
#Script to plot the track of a source

import katpoint
import matplotlib.pyplot as plt
import time
import numpy as np

#You will need to start with an antenna object ... the simplest way I have found to get the most
#accurate version of an antenna object for one of the KAT7 antennas is simply to extract it out 
#of a data file created by the antenna in question. For example if you have a data file <filename> for 
#antenna 7 data you could use

#import scape 
#d = scape.DataSet(<filename>,baseline='A7A7')
#ant = d.antenna


#If you do not have a file you can create an antenna object using katpoint.
#Parameters for the antennas are from code/conf/trunk/antennas.

#for the "center" of the array
cent = 'kat7,-30:43:17.34, 21:24:38.46, 1038,12.0' # name,lat,lon,elev,diameter

ant = katpoint.Antenna(cent)


#source
source_arr = []
#3C138
source_arr.append(katpoint.construct_radec_target('05:21:09.887', '+16:38:22.06'))

observation_sources = katpoint.Catalogue(antenna=ant)

observation_sources.add(source_arr)

for source in observation_sources.targets:
	#set the timestamp array
	timestamp = []
	source_az = []
	source_el = []
	for i in range(0,24): 
		timestamp.append(katpoint.Timestamp(time.time()+i*3600))
		source_az.append(np.rad2deg(source.azel(antenna=ant,timestamp=timestamp[i])[0]))
		source_el.append(np.rad2deg(source.azel(antenna=ant,timestamp=timestamp[i])[1]))

	plt.figure(1)

	plt.scatter(source_az,source_el,c=np.array(range(0,24)),cmap='jet',s = 100)
plt.colorbar()
