import matplotlib.pyplot as plt
import katpoint
import numpy as np
from matplotlib.dates import HourLocator, DateFormatter, drange,epoch2num

cent = 'kat7,-30:43:17.34, 21:24:38.46, 1038,12.0' # name,lat,lon,elev,diameter
ant = katpoint.Antenna(cent)

dbe,name,label,flux,RA,DEC,time_stamp,source_az,source_el = [],[],[],[],[],[],[],[],[]

file = open('point_source_scan_log_test.txt','r')
for line in file:
	e = line.split(',')
	if not e[0].startswith('2'):
		continue
	if len(e) < 6:
		continue
	i = 0
	if len(e) == 7:
		i = 1
	dbe.append(e[0])
	name.append(e[1].split('|')[0]) 
	label.append(e[2])
	flux.append(e[5*i])
	RA.append(e[3])
	DEC.append(e[4])
	time_stamp.append(float(e[5+i]))
	source = katpoint.construct_radec_target(e[3], e[4])
	source_az.append(source.azel(antenna=ant,timestamp=time_stamp[-1])[0])
	source_el.append(source.azel(antenna=ant,timestamp=time_stamp[-1])[1])

file.close()
source_az = np.array(source_az)
source_el = np.array(source_el)

xranges = [(epoch2num(t),30.0/(24*60*60)) for t in time_stamp]
fig = plt.figure(0)
ax = fig.add_subplot(111)
ax.broken_barh(xranges, (0, 1), facecolors='blue')
ax.xaxis.set_major_formatter(DateFormatter('%d/%m/%Y\n%H:%M:%S'))

plt.figure(1)
ax = plt.subplot(111,polar = True)
ax.scatter(source_az,np.pi/2 - source_el)

plt.show()


