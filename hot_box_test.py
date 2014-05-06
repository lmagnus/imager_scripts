import katarchive
import katfile
import numpy as np
import matplotlib as plt

files = katarchive.list_katfiles(start='27/3/2012')

for fl in files:
	h = katfile.open(fl)
	print h
	plot(np.mean(h.vis[:,:,0].real,1))



