import katfile
import katarchive
import matplotlib.pyplot as plt
from matplotlib.dates import HourLocator, DateFormatter, drange,epoch2num
import numpy as np

f = katarchive.get_archived_products('1327151141.h5,1327180529.h5,1327209599.h5')

h5 = katfile.open(f)

h5.select(channels = range(500,600), scans='track',ants='ant7',pol='HH')
HH = h5.vis[:]
h5.select(channels = range(500,600), scans='track',ants='ant7',pol='VV')
VV = h5.vis[:]
h5.select(channels = range(500,600), scans='track',ants='ant7',pol='VH')
VH = h5.vis[:]

I = VV + HH
Q = VV - HH
U = 2 * np.real(VH)
V = 2 * np.imag(VH)

chi = 0.5 * np.arctan(V/np.sqrt(np.power(U,2)+np.power(Q,2))) 
psi = 0.5 * np.arctan(U/Q)

times = epoch2num(h5.timestamps[:])
plt.plot(times,I[:,80,0],label='I')
plt.plot(times,Q[:,80,0],label='Q')
plt.plot(times,U[:,80,0],label='U')
plt.plot(times,V[:,80,0],label='V')
plt.legend()
plt.figure()
plt.plot(times,chi[:,80,0],label='chi')
plt.plot(times,psi[:,80,0],label='psi')
plt.legend()
a = plt.gca()
a.xaxis.set_major_locator(HourLocator(range(0,24,2)))
a.xaxis.set_major_formatter(DateFormatter('%d/%m/%Y\n%H:%M:%S'))
plt.show()

