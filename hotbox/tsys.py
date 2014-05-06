import katarchive
import katfile
import scape

import matplotlib.pyplot as plt
import numpy as np
from stitch import stitch

try:
    from my_roach_hsys import make_van_vleck_inverse_function
    inv_van_vleck = make_van_vleck_inverse_function()
except Exception, e:
    print("Proceeding without van Vleck correction: %s" % e)
    inv_van_vleck = lambda fn, x, chan=None: x

pol = 'v'

dbe7_hot = ['1344329413.h5','1344250298.h5','1344420352.h5','1344429189.h5','1344939597.h5','1343128008.h5','1344947543.h5']
dbe7_cold = ['1344332256.h5','1344253392.h5','1344423054.h5','1344432633.h5','1344941903.h5','1343130223.h5','1344950290.h5']

ff_hot = ['1316000070.h5','1309772946.h5','1309853604.h5','1318240578.h5','1317297097.h5','1317896927.h5','1309859500.h5']
ff_cold = ['1316002096.h5','1309775420.h5','1309855663.h5','1318242675.h5','1317298249.h5','1317899419.h5','1309861113.h5']

colour = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
for a,b,c,d,col in zip(dbe7_hot,dbe7_cold,ff_hot,ff_cold,colour):    
    fi = katarchive.get_archived_product(a)
    hot = katfile.open(fi)
    fi = katarchive.get_archived_product(b)
    cold = katfile.open(fi)
    
    fi = katarchive.get_archived_product(c)
    ff_hot = katfile.open(fi)
    fi = katarchive.get_archived_product(d)
    ff_cold = katfile.open(fi)
    
    ant = hot.ants[0].name
    air_temp = np.mean(hot.sensor['Enviro/asc.air.temperature'])
    ff_air_temp = np.mean(ff_hot.sensor['Antennas/'+ant+'/enviro_air_temperature'])
    
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
    f,d = stitch(s_tsys,freq,n_channels=len(s_tsys),df=hot.channel_width,smooth=True)
    plt.plot(f,d,color=col,label=ant)
    plt.figure(2) 
    f,d = stitch(s_ff_tsys,ff_freq,n_channels=len(s_ff_tsys),df=ff_hot.channel_width,smooth=True)
    plt.plot(f,d,color=col,label=ant)
    
plt.figure(1)
plt.ylim(20,70)
plt.ylabel('Tsys/[K]')
plt.xlabel('Freq/[Hz]')
plt.title('Pol: '+pol+' : Tsys for DBE7')
plt.legend()
plt.grid()

plt.figure(2)
plt.ylim(20,70)
plt.ylabel('Tsys/[K]')
plt.xlabel('Freq/[Hz]')
plt.title('Pol: '+pol+' : Tsys for FF')
plt.legend()
plt.grid()

plt.show()

