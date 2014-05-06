import katarchive
import katfile
import numpy as np
import matplotlib.pyplot as plt

files =  katarchive.search_archive(startdate='6/8/2012',description='RFI')

plt.figure()

for j,f in enumerate(files):
	f._get_archived_product()
	h5 = katfile.open(f.path_to_file)
	if len(h5.spectral_windows) != 3:
		continue
	for i,s in enumerate(h5.spectral_windows):
		h5.select(spw=i,corrprods='cross')
		flags = h5.flags()[:,:,:]
		data = np.mean(flags,2)
		data = np.mean(data,0)
		plt.subplot(len(files),1,j+1)
		plt.plot(h5.channel_freqs,data,'b')

