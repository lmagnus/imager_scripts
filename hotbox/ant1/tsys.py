import katarchive
import katfile
import scape

import matplotlib.pyplot as plt
import numpy as np
from stitch import stitch

fi = katarchive.get_archived_product('1344329413.h5')
hot = katfile.open(fi)
fi = katarchive.get_archived_product('1344332256.h5')
cold = katfile.open(fi)

fi = katarchive.get_archived_product('1316000070.h5')
ff_hot = katfile.open(fi)
fi = katarchive.get_archived_product('1316002096.h5')
ff_cold = katfile.open(fi)



diode = 'coupler'
pol = 'h'
ant = hot.ants[0].name
air_temp = np.mean(hot.sensor['Enviro/asc.air.temperature'])
ff_air_temp = np.mean(ff_hot.sensor['Antennas/'+ant+'/enviro_air_temperature'])
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
	ff_cold_data.append(ff_cold.vis[3:-3,:,:])
						 
for i,l,t in ff_hot.scans():
	ff_hot_data.append(ff_hot.vis[3:-3,:,:])

for cn,s in enumerate(cold.spectral_windows):
	if cn == 0 and s.centre_freq != 1264e6:
		continue
	f_c = s.centre_freq
	cold.select(spw=cn,pol=pol,freqrange=(f_c - 128e6, f_c + 128e6))
	freq.append(cold.channel_freqs)
	nd_temp.append(nd.temperature(freq[-1] / 1e6))
	for i,l,t in cold.scans():
		cold_data.append(cold.vis[3:-3,:,:])

for cn,s in enumerate(hot.spectral_windows):
	if cn == 0 and s.centre_freq != 1264e6:
		continue
	f_c = s.centre_freq
	hot.select(spw=cn,pol=pol,freqrange=(f_c - 128e6, f_c + 128e6))
	for i,l,t in hot.scans():
		hot_data.append(hot.vis[3:-3,:,:])



s_tsys = []
s_ff_tsys = []
s_tdiode = []
s_nd_temp = []
#plt.figure()
for i in range(0,18,3):
    cold_spec = np.mean(cold_data[i].real,0)[:,0]
    hot_spec = np.mean(hot_data[i].real,0)[:,0]
    Y = hot_spec / cold_spec
    Tcold = 17.1-3*(freq[i/3]/1e9)
    Thot = 273.15+air_temp
    Trx = (Thot-Y*Tcold)/(Y-1)
    tsys = Tcold + Trx
    s_tsys.append(tsys)
    
    ff_cold_spec = np.mean(ff_cold_data[i].real,0)[:,0]
    ff_hot_spec = np.mean(ff_hot_data[i].real,0)[:,0]
    ff_Y = ff_hot_spec / ff_cold_spec
    ff_Tcold = 17.1-3*(ff_freq[i/3]/1e9)
    ff_Thot = 273.15+ff_air_temp
    ff_Trx = (ff_Thot-ff_Y*ff_Tcold)/(ff_Y-1)
    ff_tsys = ff_Tcold + ff_Trx
    s_ff_tsys.append(ff_tsys)
    plt.figure(1)

    cold_nd_spec = np.mean(cold_data[i+nd_sc_no].real,0)[:,0]
    hot_nd_spec = np.mean(hot_data[i+nd_sc_no].real,0)[:,0]
    Ydiode = hot_nd_spec / cold_nd_spec
    Trx_nd = (Thot - Ydiode*Tcold)/(Ydiode-1)
    tdiode = Trx_nd - Trx
    s_tdiode.append(tdiode)

plt.figure(1)
f,d = stitch(s_tsys,freq,n_channels=len(s_tsys),df=hot.channel_width,smooth=True)
plt.plot(f,d,'b')

f,d = stitch(s_ff_tsys,ff_freq,n_channels=len(s_ff_tsys),df=ff_hot.channel_width,smooth=True)
plt.plot(f,d,'k')

plt.figure(2)
f,d = stitch(s_tdiode,freq,n_channels=len(s_tdiode),df=hot.channel_width,smooth=True)
plt.plot(f,d,'b')

f,d = stitch(nd_temp,freq,n_channels=len(s_nd_temp),df=hot.channel_width,smooth=True)
plt.plot(f,d,'r')

for i in range(0,18,3):
	cold_spec = np.mean(cold_data[i].real[:,:,0],0)/cold.channel_width
	ff_cold_spec = np.mean(ff_cold_data[i].real[:,:,0],0)/ff_cold.channel_width
	cold_std = np.std(cold_data[i].real[:,:,0],0,dtype=np.float64)/(np.sqrt(cold.channel_width))
	ff_cold_std = np.std(ff_cold_data[i].real[:,:,0],0,dtype=np.float64)/(np.sqrt(ff_cold.channel_width))
	plt.figure(3)
	plt.plot(freq[i/3],cold_std/cold_spec,'b')
	plt.plot(ff_freq[i/3],ff_cold_std/ff_cold_spec,'r')
plt.figure(3)
plt.ylim(0,10)
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

