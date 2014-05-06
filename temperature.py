
def calc_gain(h5,data):
    ant = h5.ants[0].name
    on = h5.sensor['Antennas/'+ant+'/nd_coupler']
    off = ~h5.sensor['Antennas/'+ant+'/nd_coupler']
    spec = np.mean(data[off,:,0].real,0)
    nd_spec = np.mean(data[on,:,0].real,0)
    jump = nd_spec - spec
    nd_model = h5.file['MetaData/Configuration/Antennas/'+ant+'/h_coupler_noise_diode_model'].value
    nd = scape.gaincal.NoiseDiodeModel(freq = nd_model[:,0]/1e6,temp = nd_model[:,1])
    nd_temp = nd.temperature(h5.channel_freqs / 1e6)
    return nd_temp/jump



