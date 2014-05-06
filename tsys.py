import katarchive
import katfile
import scape

import matplotlib.pyplot as plt
import numpy as np

def overlap_range(x, y):
    """
        @param x: an ordered list e.g. [10 11 12 13 14 15 16 17]
        @param y: another ordered list e.g. [13 14 15]
        @return: (start, stop) indices in x where x and y overlap, e.g. (3, 5)
    """
    # No overlap or not on x
    if (y[0] > x[-1]) or (y[1] < x[0]):
        return None
    start = 0
    while x[start] < y[0]:
       start += 1
    stop = start + len(y) - 1
    while (stop < len(x)-1) and (x[stop] < y[-1]):
        stop += 1
    return (start, stop)

files = katarchive.list_katfiles(start = '5/7/2011',end = '5/7/2011',description ='7')[1:4]
hot = katfile.open(files[1])
cold = katfile.open(files[2])

hot.select(channels=range(100,400))
cold.select(channels=range(100,400))
cold.select(ants = 'ant7',corrprods = 'auto',pol = 'h')
hot.select(ants = 'ant7',corrprods = 'auto',pol = 'h')

nd = scape.gaincal.NoiseDiodeModel('ant7.coupler.h.csv')
#temp_nd = nd.temperature(center_freqs / 1e6)
cold_data = [] 
cold_coup_data = []
hot_data = []
hot_coup_data = []
cold_freq = []
hot_freq = []
cold_nd_temp = []

for i,l,t in cold.scans():
    if i in range(0,18,3):
        cold_data.append(cold.vis[:,:,:])
	f_c = float(cold.sensor['Observation/label'][0][:8]) - 4200
	cold_freq.append((f_c * 1e6 + 200e6) - (cold.channels * cold.channel_width))
	cold_nd_temp.append(nd.temperature(cold_freq[-1] / 1e6))
    if i in range(1,18,3):
        cold_coup_data.append(cold.vis[:,:,:])

for i,l,t in hot.scans():
    if i in range(0,18,3):
        hot_data.append(hot.vis[:,:,:])
	f_c = float(hot.sensor['Observation/label'][0][:8]) - 4200
	hot_freq.append((f_c * 1e6 + 200e6) - (hot.channels * hot.channel_width))
    if i in range(1,18,3):
        hot_coup_data.append(hot.vis[:,:,:])

plt.figure()
for i in range(0,6):
    cold_spec = np.median(cold_data[i].real,0)[:,0]
    coup_spec = np.median(cold_coup_data[i].real,0)[:,0]
    jump = coup_spec - cold_spec
    gain = jump / np.array(cold_nd_temp[i])
    tsys = cold_spec / gain
    plt.plot(cold_freq[i],tsys,'b')

#plt.figure()
for i in range(0,6):
    cold_spec = np.median(cold_data[i].real,0)[:,0]
    hot_spec = np.median(hot_data[i].real,0)[:,0]
    Y = hot_spec / cold_spec
    tsys = (300-17.1-3*(cold_freq[i]/1e9))/Y
    plt.plot(cold_freq[i],tsys,'r')

print(overlap_range(cold_freq[0],cold_freq[1]))

plt.show()
