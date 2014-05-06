import katarchive
import katfile
import scape

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig 


pols = ['v','h']
diodes = ['pin','coupler'] 



first_cold = ['1360834610.h5','1360830130.h5','1360838606.h5','1360844527.h5','1360913982.h5','1360919003.h5','1360923262.h5']
second_cold = ['1360837183.h5','1360832758.h5','1360841328.h5','1360847111.h5','1360916700.h5','1360921706.h5','1360926295.h5']


colour = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
for pol in pols:
	for a,b,col in zip(first_cold,second_cold,colour):	
		fi = katarchive.get_archived_products(a)
		cold1 = katfile.open(fi[0])
		fi = katarchive.get_archived_products(b)
		cold2 = katfile.open(fi[0])
		
		ant = cold1.ants[0].name
		ant_num = int(ant[3])
		cold2_data = [] 
		cold1_data = []
		freq = []
		cold2_off = []
		cold1_off = []
		
		for cn,s in enumerate(cold2.spectral_windows):
			if cn == 0 and s.centre_freq != 1264e6:
				continue
			f_c = s.centre_freq
			cold2.select(spw=cn,pol=pol,freqrange=(f_c - 128e6, f_c + 128e6))
			freq.append(cold2.channel_freqs)
			cold2_data.append(np.ma.array(cold2.vis[:,:,:],mask=cold2.flags()[:],fill_value=np.nan))
			cold2_off.append(~np.logical_or(cold2.sensor['Antennas/'+ant+'/nd_pin'],cold2.sensor['Antennas/'+ant+'/nd_coupler']))
		
		for cn,s in enumerate(cold1.spectral_windows):
			if cn == 0 and s.centre_freq != 1264e6:
				continue
			f_c = s.centre_freq
			cold1.select(spw=cn,pol=pol,freqrange=(f_c - 128e6, f_c + 128e6))
			cold1_data.append(np.ma.array(cold1.vis[:,:,:],mask=cold1.flags()[:],fill_value=np.nan))
			cold1_off.append(~np.logical_or(cold1.sensor['Antennas/'+ant+'/nd_pin'],cold1.sensor['Antennas/'+ant+'/nd_coupler']))
		
		
		for i in range(6):
			cold2_spec = cold2_data[i][150,:,0].real#np.median(cold2_data[i][cold2_off[i],:,0].real,0)
			cold1_spec = cold1_data[i][150,:,0].real#np.median(cold1_data[i][cold1_off[i],:,0].real,0)
			plt.figure(1)
			p = ant_num * 2 if pol == 'v' else ant_num * 2 - 1
			plt.subplot(7,2,p)
			plt.plot(freq[i],cold1_spec,'b')
			plt.plot(freq[i],cold2_spec,'r')
			plt.ylim(0,50)
			plt.figure(2)
			plt.subplot(7,2,p)
			plt.plot(freq[i],100* (cold1_spec - cold2_spec)/cold1_spec,'b')
			plt.ylim(-5,5)


plt.figure(1)
plt.subplot(7,2,1)
plt.title('Spectrum: H pol')
plt.subplot(7,2,2)
plt.title('Spectrum: V pol')
plt.figure(2)
plt.subplot(7,2,1)
plt.title('Residual/[%]: H pol')
plt.subplot(7,2,2)
plt.title('Residual/[%]: V pol')
plt.show()


