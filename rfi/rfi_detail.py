import katarchive
import katfile
import numpy as np
import matplotlib.pyplot as plt

files =  katarchive.search_archive(startdate='8/10/2012',description='RFI')#'Basic')
for j,f in enumerate(files):
	plt.figure()
	f._get_archived_product()
	h5 = katfile.open(f.path_to_file)
	ants = h5.ants
	N = len(ants)
	for i,s in enumerate(h5.spectral_windows):
		for an,a in enumerate(ants):
			h5.select(spw=i)
			h5.select(corrprods='auto',ants = a.name)
			flags = h5.flags()[:,1:,:]
			data = np.mean(flags,2)
			data = np.mean(data,0)
			plt.subplot(N,1,an+1)
			plt.plot(h5.channel_freqs[1:],data,'b')
			plt.text(1.7e9,0.6,a.name+": "+h5.start_time.local())
			ax = plt.gca()
			if not an == N-1:
				ax.set_xticklabels([])
			ax.set_xlim([1.1e9,2.1e9])
plt.show()
