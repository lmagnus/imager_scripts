import katarchive
import matplotlib.pyplot as plt
import katfile
import numpy as np

plt.figure()

data = []
ts = []
files = katarchive.list_katfiles(start = '04/01/2012', end = '06/01/2012')
for file in files:
        h5 = katfile.open(file)
	h5.select(ants = 'ant2',corrprods = 'auto',pol='v')
	data.append(np.mean(h5.vis[:,200:800,0].real,1))
	ts.append(h5.timestamps[:])

data = np.vstack(np.array(data))[:,0]
ts = np.hstack(ts)

plt.plot(ts,data)

plt.show
