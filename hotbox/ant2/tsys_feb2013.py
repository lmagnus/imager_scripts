import katarchive
import katfile
import scape

import matplotlib.pyplot as plt
import numpy as np

fi = katarchive.get_archived_products('1360830130.h5')
cold = katfile.open(fi[0])
fi = katarchive.get_archived_products('1360831379.h5')
hot = katfile.open(fi[0])

for diode in ['coupler','pin']:
	for pol in ['h','v']:
		f_num = 1 if diode == 'pin' else 3 
		ant = hot.ants[0].name
		air_temp = np.mean(hot.sensor['Enviro/asc.air.temperature'])
		nd_sc_no = 0 if diode == 'coupler' else 1

		nd = scape.gaincal.NoiseDiodeModel('../nd_models/'+ant+'.'+diode+'.'+pol+'.csv')
#temp_nd = nd.temperature(center_freqs / 1e6)
		cold_data = []
		cold_flags = []
		hot_data = []
		hot_flags = []
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
			cold_data.append(np.ma.array(cold.vis[:,:,:],mask=cold.flags()[:]))
			cold_flags.append(cold.flags()[:])
			cold_off.append(~np.logical_or(cold.sensor['Antennas/'+ant+'/nd_pin'],cold.sensor['Antennas/'+ant+'/nd_coupler']))
			cold_on.append(cold.sensor['Antennas/'+ant+'/nd_'+diode])

		for cn,s in enumerate(hot.spectral_windows):
			if cn == 0 and s.centre_freq != 1264e6:
				continue
			f_c = s.centre_freq
			hot.select(spw=cn,pol=pol,freqrange=(f_c - 128e6, f_c + 128e6))
			hot_data.append(np.ma.array(hot.vis[:,:,:],mask=hot.flags()[:]))
			hot_flags.append(hot.flags()[:])
			hot_off.append(~np.logical_or(hot.sensor['Antennas/'+ant+'/nd_pin'],hot.sensor['Antennas/'+ant+'/nd_coupler']))
			hot_on.append(hot.sensor['Antennas/'+ant+'/nd_'+diode])

		plt.figure(f_num)
		for i in range(6):
		    cold_spec = np.median(cold_data[i][cold_off[i],:,0].real,0)
		    nd_spec = np.median(cold_data[i][cold_on[i],:,0].real,0)
		    jump = nd_spec - cold_spec
		    gain = jump / np.array(nd_temp[i])
		    tsys = cold_spec / gain
		    plt.plot(freq[i],tsys,'b',label='nd_'+diode)

#plt.figure()
		for i in range(6):
		    cold_spec = np.median(cold_data[i][cold_off[i],:,0].real,0)
		    hot_spec = np.median(hot_data[i][hot_off[i],:,0].real,0)
		    Y = hot_spec / cold_spec
		    #tsys = (273.15+air_temp-17.1-3*(freq[i/3]/1e9))/(Y-1)
		    Tcold = 17.1-3*(freq[i]/1e9)
		    Thot = 273.15+air_temp
		    Trx = (Thot-Y*Tcold)/(Y-1)
		    tsys = Tcold + Trx 
		    plt.figure(f_num)
		    plt.plot(freq[i],tsys,'r',label='Y method')
		    cold_nd_spec = np.median(cold_data[i][cold_on[i],:,0].real,0)
		    hot_nd_spec = np.median(hot_data[i][hot_on[i],:,0].real,0)
		    Ydiode = hot_nd_spec / cold_nd_spec
		    Trx_nd = (Thot - Ydiode*Tcold)/(Ydiode-1)
		    tdiode = Trx_nd - Trx
		    plt.figure(f_num+1)
		    plt.plot(freq[i],tdiode,'b')
		    plt.plot(freq[i],np.array(nd_temp[i]),'r')

		plt.figure(f_num)
		plt.ylim(0,50)
		plt.ylabel('Tsys/[K]')
		plt.xlabel('Freq/[Hz]')
		plt.title(ant+pol+': Tsys at: '+cold.start_time.local())
		plt.grid()
		plt.figure(f_num+1)
		plt.ylim(0,5 if diode == 'coupler' else 80)
		plt.ylabel('Tdiode/[K]')
		plt.xlabel('Freq/[Hz]')
		plt.title(ant+pol+': Tdiode_'+diode+' at: '+cold.start_time.local())
		plt.grid()

plt.show()
