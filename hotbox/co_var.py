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
    
    
    s_dbe7 = []
    s_ff = []
    for i in range(0,18,3):
        cold_spec = np.mean(cold_data[i].real[:,:,0],0)/cold.channel_width
        ff_cold_spec = np.mean(ff_cold_data[i].real[:,:,0],0)/ff_cold.channel_width
        cold_std = np.std(cold_data[i].real[:,:,0],0,dtype=np.float64)/(np.sqrt(cold.channel_width))
        ff_cold_std = np.std(ff_cold_data[i].real[:,:,0],0,dtype=np.float64)/(np.sqrt(ff_cold.channel_width))
        s_dbe7.append(cold_std/cold_spec)
        s_ff.append(ff_cold_std/ff_cold_spec)
    
    plt.figure(1)
    f,d = stitch(s_dbe7,freq,n_channels=len(s_dbe7),df=hot.channel_width,smooth=True)
    plt.plot(f,d,color=col,label=ant)
    plt.figure(2) 
    f,d = stitch(s_ff,ff_freq,n_channels=len(s_ff),df=ff_hot.channel_width,smooth=True)
    plt.plot(f,d,color=col,label=ant)
    
plt.figure(1)
plt.ylim(0,4)
plt.ylabel('CoV')
plt.xlabel('Freq/[Hz]')
plt.title('Pol: '+pol+' : DBE7 Coeff of Variation')
plt.legend()
plt.grid()

plt.figure(2)
plt.ylim(0,4)
plt.ylabel('CoV')
plt.xlabel('Freq/[Hz]')
plt.title('Pol: '+pol+' : FF Coeff of Variation')
plt.legend()
plt.grid()

plt.show()

