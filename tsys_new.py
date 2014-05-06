import katarchive
import katfile
import scape

import matplotlib.pyplot as plt
import numpy as np

files = katarchive.list_katfiles(filename = '1320314989.h5')
cold = katfile.open(files[0])

cold.select(channels=range(200,800))
cold.select(ants = 'ant7',corrprods = 'auto',pol = 'h')

nd = scape.gaincal.NoiseDiodeModel('ant7.coupler.h.csv')

cold_data = [] 
cold_coup_data = []
cold_freq = []
cold_nd_temp = []

for i,l,t in cold.scans():
    if cold.sensor['Observation/label'][0] == 'cold' and l == 'track':
        cold_data.append(cold.vis[:,:,:])
	f_c = float(cold.sensor['RFE/rfe7.lo1.frequency'][0]) - 4200e6
	cold_freq.append((f_c + 200e6) - (cold.channels * cold.channel_width))
	cold_nd_temp.append(nd.temperature(cold_freq[-1] / 1e6))
    if cold.sensor['Observation/label'][0] == 'cold+coupler' and l == 'track':
        cold_coup_data.append(cold.vis[:,:,:])

plt.figure()
for i in range(0,6):
    cold_spec = np.median(cold_data[i].real,0)[:,0]
    coup_spec = np.median(cold_coup_data[i].real,0)[:,0]
    jump = coup_spec - cold_spec
    gain = jump / np.array(cold_nd_temp[i])
    tsys = cold_spec / gain
    plt.plot(cold_freq[i],tsys)
plt.xlabel('freq')
plt.ylabel('Tsys/[K]')
plt.title('1320314989.h5')
plt.show()
