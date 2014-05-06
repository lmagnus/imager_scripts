import katarchive
import katfile
import scape

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig 
from stitch import stitch

try:
	from my_roach_hsys import make_van_vleck_inverse_function
	inv_van_vleck = make_van_vleck_inverse_function()
except Exception, e:
	print("Proceeding without van Vleck correction: %s" % e)
	inv_van_vleck = lambda fn, x, chan=None: x

pols = ['v','h']
diodes = ['pin','coupler'] 

dbe7_hot = ['1360835888.h5','1360831379.h5','1360839945.h5','1360845700.h5','1360915318.h5','1360920291.h5','1360924347.h5']
dbe7_cold = ['1360834610.h5','1360830130.h5','1360838606.h5','1360844527.h5','1360913982.h5','1360919003.h5','1360923262.h5']


colour = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
fnum = -1
for diode in diodes:
	fnum += 1
	for pol in pols:
		#fnum += 1
		for a,b,col in zip(dbe7_hot,dbe7_cold,colour):	
			fi = katarchive.get_archived_products(a)
			hot = katfile.open(fi[0])
			fi = katarchive.get_archived_products(b)
			cold = katfile.open(fi[0])
			
			ant = hot.ants[0].name
			ant_num = int(ant[3])
			air_temp = np.mean(hot.sensor['Enviro/asc.air.temperature'])
			nd_model = hot.file['MetaData/Configuration/Antennas/'+ant+'/'+pol+'_'+diode+'_noise_diode_model'].value
			nd = scape.gaincal.NoiseDiodeModel(freq = nd_model[:,0]/1e6,temp = nd_model[:,1])
			cold_data = [] 
			hot_data = []
			freq = []
			nd_temp = []
			cold_off = []
			cold_on = []
			hot_off = []
			hot_on = []
			
			for cn,s in enumerate(cold.spectral_windows):
				if cn == 0 and s.centre_freq != 1264e6:
					continue
				f_c = s.centre_freq
				cold.select(spw=cn,pol=pol,freqrange=(f_c - 128e6, f_c + 128e6))
				freq.append(cold.channel_freqs)
				nd_temp.append(nd.temperature(freq[-1] / 1e6))
				cold_data.append(np.ma.array(cold.vis[:,:,:],mask=cold.flags()[:],fill_value=np.nan))
				cold_off.append(~np.logical_or(cold.sensor['Antennas/'+ant+'/nd_pin'],cold.sensor['Antennas/'+ant+'/nd_coupler']))
				cold_on.append(cold.sensor['Antennas/'+ant+'/nd_'+diode])
			
			for cn,s in enumerate(hot.spectral_windows):
				if cn == 0 and s.centre_freq != 1264e6:
					continue
				f_c = s.centre_freq
				hot.select(spw=cn,pol=pol,freqrange=(f_c - 128e6, f_c + 128e6))
				hot_data.append(np.ma.array(hot.vis[:,:,:],mask=hot.flags()[:],fill_value=np.nan))
				hot_off.append(~np.logical_or(hot.sensor['Antennas/'+ant+'/nd_pin'],hot.sensor['Antennas/'+ant+'/nd_coupler']))
				hot_on.append(hot.sensor['Antennas/'+ant+'/nd_'+diode])
			
			
			s_tdiode = []
			s_tsys = []
			for i in range(6):
				cold_spec = np.mean(cold_data[i][cold_off[i],:,0].real,0)
				hot_spec = np.mean(hot_data[i][hot_off[i],:,0].real,0)
				Y = hot_spec / cold_spec
				Tcold = 17.1-3*(freq[i]/1e9)
				Thot = 273.15+air_temp
				Trx = (Thot-Y*Tcold)/(Y-1)
				Tsys = Tcold + Trx
				cold_nd_spec = np.mean(cold_data[i][cold_on[i],:,0].real,0)
				hot_nd_spec = np.mean(hot_data[i][hot_on[i],:,0].real,0)
				Ydiode = hot_nd_spec / cold_nd_spec
				Trx_nd = (Thot - Ydiode*Tcold)/(Ydiode-1)
				tdiode = Trx_nd - Trx
				tdiode.set_fill_value(np.nan)
				s_tdiode.append(tdiode.filled())
				Tsys.set_fill_value(np.nan)
				s_tsys.append(Tsys.filled())

			plt.figure(1)
			if pol == 'v' and diode == 'coupler': p = ant_num * 4 
			if pol == 'h' and diode == 'coupler': p = ant_num * 4-1 
			if pol == 'v' and diode == 'pin': p = ant_num * 4-2
			if pol == 'h' and diode == 'pin': p = ant_num * 4-3 
			plt.subplot(7,4,p)
			plt.ylim(0,6 if diode == 'coupler' else 90)
			if p ==ant_num * 4-3: plt.ylabel(ant)
			f,d = stitch(s_tdiode,freq,n_channels=len(s_tdiode),df=hot.channel_width,smooth=True)
			#plt.plot(f,d,col,label=ant)
			fs,ds = f,sig.medfilt(d,21)
			plt.plot(f,ds,'b')
			outfile = file('%s.%s.%s.csv' % (ant, diode, pol.lower()), 'w')
			outfile.write('#\n# Frequency [Hz], Temperature [K]\n')
			# Write CSV part of file
			outfile.write(''.join(['%s, %s\n' % (entry[0], entry[1]) for entry in zip(f[((fs>1.2e9) & (fs < 1.95e9))],d[((fs>1.2e9) & (fs < 1.95e9))])]))
			outfile.close()
			for f,d in zip(freq,nd_temp):
				plt.plot(f,d,'k')
			plt.figure(2)
			p = ant_num * 2 if pol == 'v' else ant_num * 2 -1
			plt.subplot(7,2,p)
			plt.ylim(20,40)
			if p == ant_num * 2 -1: plt.ylabel(ant)
			f,t = stitch(s_tsys,freq,n_channels=len(s_tsys),df=hot.channel_width,smooth=True)
			if diode == 'pin': plt.plot(f,t,'b')
			plt.grid()
	
plt.figure(1)
plt.subplot(7,4,1)
plt.title('Pin Diode: H pol')
plt.subplot(7,4,2)
plt.title('Pin Diode: V pol')
plt.subplot(7,4,3)
plt.title('Coupler Diode: H pol')
plt.subplot(7,4,4)
plt.title('Coupler Diode: V pol')

plt.figure(2)
plt.subplot(7,2,1)
plt.title('Tsys: H pol')
plt.subplot(7,2,2)
plt.title('Tsys: V pol')

plt.show()

