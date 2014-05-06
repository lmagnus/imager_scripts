import katfile
import katarchive
import matplotlib.pyplot as plt
import numpy as np
import scape



f = katarchive.get_archived_products('1365193129.h5,1365188668.h5,1365184129.h5,1365179749.h5')
i=1
for fi in f:
    h5 = katfile.open(fi)
    ant = 'ant4'
    h5.select(ants=ant,corrprods='auto',scans='track')
    data = h5.vis[:].real
    on = h5.sensor['Antennas/'+ant+'/nd_coupler']
    off = ~h5.sensor['Antennas/'+ant+'/nd_coupler']
    spec = np.mean(data[off,:,0].real)
    nd_spec = np.mean(data[on,:,0].real)
    jump = nd_spec - spec
    nd_model = h5.file['MetaData/Configuration/Antennas/'+ant+'/h_coupler_noise_diode_model'].value
    nd = scape.gaincal.NoiseDiodeModel(freq = nd_model[:,0]/1e6,temp = nd_model[:,1])
    nd_temp = np.mean(nd.temperature(h5.channel_freqs / 1e6))
    gain = (nd_temp/jump) * 1.3806488e-23
    plt.subplot(4,2,i)
    i +=1
    plt.plot(h5.channel_freqs,10*np.log10(np.mean(data[:,:,0],0)*gain*h5.channel_width))
    plt.xlim(1.8335e9,1.8345e9)
    #plt.ylim(5.5,12.5)
    plt.subplot(4,2,i)
    i +=1
    plt.plot(h5.channel_freqs,10*np.log10(data[200,:,0]*gain*h5.channel_width))
    plt.xlim(1.8335e9,1.8345e9)
    #plt.ylim(5.5,12.5)
plt.show()
