import katfile
import matplotlib.pyplot as plt
from matplotlib.dates import HourLocator, DateFormatter, drange,epoch2num
import numpy as np

h5_3 = katfile.open('/data/local_archive/data/comm/2012/01/22/1327209599.h5')
h5 = katfile.open('/data/local_archive/data/comm/2012/01/21/1327151141.h5')
h5_2 = katfile.open('/data/local_archive/data/comm/2012/01/21/1327180529.h5')
times = epoch2num(np.hstack([h5.timestamps(),h5_2.timestamps(),h5_3.timestamps()]))
data = h5.vis(('ant7V','ant7H'))
data2 = h5_2.vis(('ant7V','ant7H'))
data3 = h5_3.vis(('ant7V','ant7H')) 
#line_data = hstack([mean(10*log(data[:,500:600]),1),mean(10*log(data2[:,500:600]),1),mean(10*log(data3[:,500:600]),1)])
line_data = np.hstack([np.angle(data[:,550]),np.angle(data2[:,550]),np.angle(data3[:,550])])
plt.plot(times,line_data)
a = plt.gca()
a.xaxis.set_major_locator(HourLocator(range(0,24,2)))
a.xaxis.set_major_formatter(DateFormatter('%d/%m/%Y\n%H:%M:%S'))
plt.show()
