import katarchive
import katfile
import numpy as np
import matplotlib.pyplot as plt

files =  katarchive.search_archive(startdate='1/9/2012',enddate='24/10/2012',description='RFI')#'Basic')

plt.figure()
fil=[]
for j,f in enumerate(files):
	if f.metadata.Antennas.find('7') < 0:
		continue
	fil.append(f)

for j,f in enumerate(fil):
	f._get_archived_product()
	h5 = katfile.open(f.path_to_file)
	for i,s in enumerate(h5.spectral_windows):
		h5.select(spw = i,freqrange=[1.25e9,1.35e9],ants = 'ant7',corrprods='auto')
		data = h5.flags()[:,1:,:]
		data = h5.vis[:,1:,:].real
		#data = np.mean(data,2)
		data = np.mean(data,0)
		plt.subplot(len(fil),1,j+1)
		plt.plot(h5.channel_freqs[1:],data,'b')
		plt.text(1.3e9,0.6,h5.start_time.local())
		plt.xlim(1.2e9,1.4e9)
