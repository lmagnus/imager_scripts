import katarchive
import katfile
import scape

import matplotlib.pyplot as plt
import numpy as np

fi = katarchive.get_archived_product('1343128008.h5')
hot = katfile.open(fi)
fi = katarchive.get_archived_product('1343130223.h5')
cold = katfile.open(fi)

fi = katarchive.get_archived_product('1317896927.h5')
ff_hot = katfile.open(fi)
fi = katarchive.get_archived_product('1317895550.h5')
ff_cold = katfile.open(fi)

diode = 'coupler'
pol = 'h'
ant = hot.ants[0].name
air_temp = np.mean(hot.sensor['Enviro/asc.air.temperature'])
nd_sc_no = 1 if diode == 'coupler' else 2

ff_hot.select(scans='scan',ants=ant,pol=pol,channels=range(100,400))
ff_cold.select(scans='scan',ants=ant,pol=pol,channels=range(100,400))

nd = scape.gaincal.NoiseDiodeModel('../nd_models/'+ant+'.'+diode+'.'+pol+'.csv')
#temp_nd = nd.temperature(center_freqs / 1e6)
cold_data = [] 
hot_data = []
freq = []
nd_temp = []

ff_cold_data = [] 
ff_hot_data = []
ff_freq = []
ff_nd_temp = []
ff_f_c = [1264,1392,1520,1648,1776,1904]

for f in ff_f_c:
	ff_freq.append((f*1e6 + 200e6) - np.arange(100,400) * 400e6/512)
	ff_nd_temp.append(nd.temperature(ff_freq[-1] / 1e6))
			        
for i,l,t in ff_cold.scans():
	ff_cold_data.append(ff_cold.vis[3:,:,:])
						 
for i,l,t in ff_hot.scans():
	ff_hot_data.append(ff_hot.vis[3:,:,:])

for cn,s in enumerate(cold.spectral_windows):
	if cn == 0 and s.centre_freq != 1264e6:
		continue
	f_c = s.centre_freq
	hot.select(spw=cn,pol=pol,freqrange=(f_c - 128e6, f_c + 128e6))
	cold.select(spw=cn,pol=pol,freqrange=(f_c - 128e6, f_c + 128e6))
	freq.append(cold.channel_freqs)
	nd_temp.append(nd.temperature(freq[-1] / 1e6))
	for i,l,t in cold.scans():
		cold_data.append(cold.vis[3:,:,:])
	for i,l,t in hot.scans():
		hot_data.append(hot.vis[3:,:,:])


plt.figure(1)
for i in range(0,18,3):
    cold_spec = np.median(cold_data[i].real,0)[:,0]
    coup_spec = np.median(cold_data[i+nd_sc_no].real,0)[:,0]
    jump = coup_spec - cold_spec
    gain = jump / np.array(nd_temp[i/3])
    tsys = cold_spec / gain
#    plt.plot(freq[i/3],tsys,'b',label='nd_'+diode)

#plt.figure()
for i in range(0,18,3):
    cold_spec = np.median(cold_data[i].real,0)[:,0]
    hot_spec = np.median(hot_data[i].real,0)[:,0]
    Y = hot_spec / cold_spec
    #tsys = (273.15+air_temp-17.1-3*(freq[i/3]/1e9))/(Y-1)
    Tcold = 17.1-3*(freq[i/3]/1e9)
    Thot = 273.15+air_temp
    Trx = (Thot-Y*Tcold)/(Y-1)
    tsys = Tcold + Trx 
    plt.figure(1)
    plt.plot(freq[i/3],tsys,'r',label='Y method')
    ff_cold_spec = np.median(ff_cold_data[i].real,0)[:,0]
    ff_hot_spec = np.median(ff_hot_data[i].real,0)[:,0]
    ff_Y = ff_hot_spec / ff_cold_spec
    #tsys = (273.15+air_temp-17.1-3*(freq[i/3]/1e9))/(Y-1)
    ff_Tcold = 17.1-3*(ff_freq[i/3]/1e9)
    Thot = 273.15+air_temp
    ff_Trx = (Thot-ff_Y*ff_Tcold)/(ff_Y-1)
    ff_tsys = ff_Tcold + ff_Trx 
    plt.figure(1)
    plt.plot(ff_freq[i/3],ff_tsys,'g',label='Y method')

    cold_nd_spec = np.median(cold_data[i+nd_sc_no].real,0)[:,0]
    hot_nd_spec = np.median(hot_data[i+nd_sc_no].real,0)[:,0]
    Ydiode = hot_nd_spec / cold_nd_spec
    Trx_nd = (Thot - Ydiode*Tcold)/(Ydiode-1)
    tdiode = Trx_nd - Trx
    plt.figure(2)
    plt.plot(freq[i/3],tdiode,'b')
    plt.plot(freq[i/3],np.array(nd_temp[i/3]),'r')

for i in range(0,18,3):
	cold_spec = np.mean(hot_data[i].real[3:,:,0],0)
	ff_cold_spec = np.mean(ff_hot_data[i].real[3:,:,0],0)
	cold_std = np.std(hot_data[i].real[3:,:,0],0,dtype=np.float64)*1.0/(np.sqrt(hot.channel_width))
	ff_cold_std = np.std(ff_hot_data[i].real[3:,:,0],0,dtype=np.float64)*1.0/(np.sqrt(ff_hot.channel_width))
	plt.figure(3)
	plt.plot(freq[i/3],cold_std/cold_spec,'b')
	plt.plot(ff_freq[i/3],ff_cold_std/ff_cold_spec,'r')
plt.figure(3)
plt.ylim(0,1e-5)
plt.ylabel('Coefficient of variation (sigma/mu)')
plt.xlabel('Freq/[Hz]')
plt.title(ant+pol+':DBE7 (blue) FF (red) at: '+cold.start_time.local())
plt.figure(1)
plt.ylim(0,50)
plt.ylabel('Tsys/[K]')
plt.xlabel('Freq/[Hz]')
plt.title(ant+pol+': Tsys at: '+cold.start_time.local())
plt.grid()
plt.figure(2)
plt.ylim(0,5 if diode == 'coupler' else 80)
plt.ylabel('Tdiode/[K]')
plt.xlabel('Freq/[Hz]')
plt.title(ant+pol+': Tdiode_'+diode+' at: '+cold.start_time.local())
plt.grid()

plt.show()

