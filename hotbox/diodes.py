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

dbe7_hot = ['1344329413.h5','1344250298.h5','1344420352.h5','1344429189.h5','1344939597.h5','1343128008.h5','1344947543.h5']
dbe7_cold = ['1344332256.h5','1344253392.h5','1344423054.h5','1344432633.h5','1344941903.h5','1343130223.h5','1344950290.h5']

ff_hot = ['1316000070.h5','1309772946.h5','1309853604.h5','1318240578.h5','1317297097.h5','1317896927.h5','1309859500.h5']
ff_cold = ['1316002096.h5','1309775420.h5','1309855663.h5','1318242675.h5','1317298249.h5','1317899419.h5','1309861113.h5']

colour = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
pol = pols[0]
diode = diodes[1]
for a,b,c,d,col in zip(dbe7_hot,dbe7_cold,ff_hot,ff_cold,colour):    
    fi = katarchive.get_archived_products(a)
    hot = katfile.open(fi[0])
    fi = katarchive.get_archived_products(b)
    cold = katfile.open(fi[0])
    
    fi = katarchive.get_archived_products(c)
    ff_hot = katfile.open(fi[0])
    fi = katarchive.get_archived_products(d)
    ff_cold = katfile.open(fi[0])
    
    ant = hot.ants[0].name
    air_temp = np.mean(hot.sensor['Enviro/asc.air.temperature'])
    ff_air_temp = np.mean(ff_hot.sensor['Antennas/'+ant+'/enviro_air_temperature'])
    nd = scape.gaincal.NoiseDiodeModel('./nd_models/'+ant+'.'+diode+'.'+pol+'.csv')
    nd_sc_no = 1 if diode == 'coupler' else 2

    ff_hot.select(scans='scan',ants=ant,pol=pol,channels=range(100,400))
    ff_cold.select(scans='scan',ants=ant,pol=pol,channels=range(100,400))
    
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
	ff_cold_data.append(inv_van_vleck(d,ff_cold.vis[3:-3,:,:],chan=None))
			     
    for i,l,t in ff_hot.scans():
	ff_hot_data.append(inv_van_vleck(c,ff_hot.vis[3:-3,:,:],chan=None))
    
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
    
    
    
    s_tdiode = []
    s_ff_tdiode = []
    for i in range(0,18,3):
	cold_spec = np.mean(cold_data[i].real,0)[:,0]
	hot_spec = np.mean(hot_data[i].real,0)[:,0]
	Y = hot_spec / cold_spec
	Tcold = 17.1-3*(freq[i/3]/1e9)
	Thot = 273.15+air_temp
	Trx = (Thot-Y*Tcold)/(Y-1)
	cold_nd_spec = np.mean(cold_data[i+nd_sc_no].real,0)[:,0]
	hot_nd_spec = np.mean(hot_data[i+nd_sc_no].real,0)[:,0]
	Ydiode = hot_nd_spec / cold_nd_spec
	Trx_nd = (Thot - Ydiode*Tcold)/(Ydiode-1)
	tdiode = Trx_nd - Trx
	s_tdiode.append(tdiode)

	ff_cold_spec = np.mean(ff_cold_data[i].real,0)[:,0]
	ff_hot_spec = np.mean(ff_hot_data[i].real,0)[:,0]
	ff_Y = ff_hot_spec / ff_cold_spec
	ff_Tcold = 17.1-3*(ff_freq[i/3]/1e9)
	Thot = 273.15+air_temp
	ff_Trx = (Thot-ff_Y*ff_Tcold)/(ff_Y-1)
	ff_cold_nd_spec = np.mean(ff_cold_data[i+nd_sc_no].real,0)[:,0]
	ff_hot_nd_spec = np.mean(ff_hot_data[i+nd_sc_no].real,0)[:,0]
	ff_Ydiode = ff_hot_nd_spec / ff_cold_nd_spec
	ff_Trx_nd = (Thot - ff_Ydiode*ff_Tcold)/(ff_Ydiode-1)
	ff_tdiode = ff_Trx_nd - ff_Trx
	s_ff_tdiode.append(ff_tdiode)
    
    
    plt.figure(1)
    f,d = stitch(s_tdiode,freq,n_channels=len(s_tdiode),df=hot.channel_width,smooth=True)
    plt.plot(f,d,color=col,label=ant)
    fs,ds = f,sig.medfilt(d,21)
    plt.plot(f,fs,'b.')
    outfile = file('%s.%s.%s.csv' % (ant, diode, pol.lower()), 'w')
    outfile.write('#\n# Frequency [Hz], Temperature [K]\n')
    # Write CSV part of file
    outfile.write(''.join(['%s, %s\n' % (entry[0], entry[1]) for entry in zip(f[((fs>1.2e9) & (fs < 1.95e9))],d[((fs>1.2e9) & (fs < 1.95e9))])]))
    outfile.close()
    plt.figure(2) 
    for a,f,d in zip(range(7),freq,nd_temp):
	if a == 0:
	    plt.plot(f,d,color=col,label=ant)
	else:
	    plt.plot(f,d,color=col)
    ds = scape.gaincal.NoiseDiodeModel(ant+'.'+diode+'.'+pol+'.csv').temperature(fs/1e6)
    plt.plot(fs,ds,'.',color=col)
    plt.figure(3) 
    f,d = stitch(s_ff_tdiode,ff_freq,n_channels=len(s_ff_tdiode),df=ff_hot.channel_width,smooth=True)
    plt.plot(f,d,color=col,label=ant)
    
    
plt.figure(1)
plt.ylim(0,6 if diode == 'coupler' else 80)
plt.ylabel('Tnd/[K]')
plt.xlabel('Freq/[Hz]')
plt.title('Diode: '+diode+' Pol: '+pol+': Tnd for DBE7')
plt.legend()
plt.grid()

plt.figure(2)
plt.ylim(0,6 if diode == 'coupler' else 80) 
plt.ylabel('Tnd/[K]')
plt.xlabel('Freq/[Hz]')
plt.title('Diode: '+diode+' Pol: '+pol+': Model Tnd')
plt.legend()
plt.grid()

plt.figure(3)
plt.ylim(0,6 if diode == 'coupler' else 80)
plt.ylabel('Tnd/[K]')
plt.xlabel('Freq/[Hz]')
plt.title('Diode: '+diode+' Pol: '+pol+': Tnd for FF')
plt.legend()
plt.grid()


plt.show()

