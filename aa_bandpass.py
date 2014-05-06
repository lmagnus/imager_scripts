# script to scan through all point source scans and create a target list.
import katfile
import katarchive
import numpy as np
from scipy import signal 
import matplotlib.pyplot as plt

def dB(x):
	return 10*np.log10(x)

def find_delta(x, win_length):
	"""
	@return: peak-to-peak difference over sliding windows on x.
	"""
	N = min([len(x), int(win_length+0.1)])
	deltas = []
	for i in range(len(x)-N+1): # Look at blocks of length N
		x_block = x[i:i+N+1]
		deltas.append(max(x_block)-min(x_block))
	return max(deltas)

all_spec = []

for i in range(14):
	all_spec.append([])

#locate the files 
n = '1362604276.h5'
#fi = katarchive.get_archived_products(n)
fi = katarchive.search_archive(description='Auto atten')
for f in fi:
	h5 = katfile.open(f.path_to_file)
	if h5.channels.shape[0] != 1024:
		continue
	n_spw = len(h5.spectral_windows)
	ants = h5.ants
	for sp in range(n_spw):
		if h5.spectral_windows[sp].centre_freq != 1822000000.0:
			continue
		for ant in ants:
			ant_num = int(ant.name[-1])
			for pol in ('H','V'):
				p = ant_num * 2 if pol == 'V' else ant_num * 2 -1
				plt.figure(1)
				plt.subplot(7,2,p)
				h5.select(ants=ant,pol=pol,spw=sp)#,channels=range(200,800))
				flags = h5.flags()[:]
				data = np.ma.array(h5.vis[:].real,mask=flags)
				spec = 10* np.log10(data[-1,:,0])
				fft_spec = signal.ifft(spec)
				f_data = fft_spec[0:20]
				all_spec[p-1].append(f_data)
				full_spec = np.concatenate((f_data,np.zeros(spec.shape[0]-39),np.conjugate(f_data[-1:0:-1])))
				#full_spec[0] = 0
				pl_data = np.real(signal.fft(full_spec))
				plt.plot(h5.channel_freqs,spec,'b')
				plt.plot(h5.channel_freqs,pl_data,'r')

