import katarchive
import katfile
import numpy as np
import matplotlib.pyplot as plt

files =  katarchive.search_archive(startdate='28/11/2012',description='RFI')#'Basic')
N = len(files)
corr = ['auto','cross']
for co in corr:
	plt.figure()
	for j,f in enumerate(files):
		f._get_archived_product()
		h5 = katfile.open(f.path_to_file)
		#if len(h5.spectral_windows) < 3:
			#continue
		for i,s in enumerate(h5.spectral_windows):
			h5.select(spw=i)
			h5.select(corrprods=co)
			flags = h5.flags()[:,1:,:]
			data = np.mean(flags,2)
			data = np.mean(data,0)
			plt.subplot(len(files),1,j+1)
			plt.plot(h5.channel_freqs[1:],data,'b')
			plt.text(1.7e9,0.6,h5.start_time.local())
			ax = plt.gca()
			if not j == N-1:
				ax.set_xticklabels([])
			ax.set_xlim([1.1e9,2.1e9])
			ax.set_ylim([0,1])
			if  j == 0:
				plt.title(co)
plt.show()
