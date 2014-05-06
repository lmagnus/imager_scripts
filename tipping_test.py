import numpy as np
import katfile
import matplotlib.pyplot as plt
import scape

f = katfile.open('/data/nadeem/Tipping/1326690796.h5')
f.select(channels=slice(300,400))
no_pol = np.tile(False,len(f.corr_products))
no_pol[0:-1:4] = True
no_pol[1:-1:4] = True
ants = f.ants
try:
	for ant in ants:
		plt.figure()
		for pol in ['h','v']:
			f.select(corrprods=[[ant.name+pol,ant.name+pol]])
			for scan_index, state, target in f.scans():
				ts = f.timestamps[:] # to determine if the scans are of zero length
				if state != 'track' or len(ts) < 2:
					continue
				#setup
				nd = np.asarray(f.sensor['Antennas/'+ant.name+'/rfe3.rfe15.noise.coupler.on'],dtype = np.int16)
				num_times = len(f.timestamps)
				#Assume all data false
				on_flag = np.tile(False, num_times)
				off_flag = np.tile(False, num_times)
				#find the jumps
				jumps = (np.diff(nd).nonzero()[0] + 1).tolist()
				#set the flags
				on_flag[ jumps[0]+1:jumps[1] ] = True
				off_flag[0 : jumps[0]] = True
				off_flag[jumps[1]+1:num_times ] = True
				#get the data
				data = f.vis[:,:,:].real
				nd_file = '/var/kat/katconfig/user/noise-diode-models/'+ant.name+'.coupler.'+pol+'.csv'
				nd_model = scape.gaincal.NoiseDiodeModel(nd_file)
				nd_temp = nd_model.temperature(f.channel_freqs / 1e6)
				on_spec = np.median(data[on_flag,:,0],0)
				off_spec = np.median(data[off_flag,:,0],0)
				jump = on_spec - off_spec
				gain = jump / np.array(nd_temp)
				tsys = off_spec / gain
				pnt = 'ob' if pol == 'h' else 'or'
				plt.plot(f.sensor['Antennas/'+ant.name+'/pos.request-scan-elev'][0],np.median(tsys),pnt)
except KeyError , error:
	print 'Key error'
	pass

