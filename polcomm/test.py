import katarchive
import katfile
import scipy
import numpy as np

f = katarchive.get_archived_products('1322589857.h5')
h5 = katfile.open(f[0])
h5.select(scans='track',ants='ant1')
data = h5.vis[:]
nd_flags = h5.sensor['Antennas/ant1/nd_pin']
nd_data = data[nd_flags,:,:]
nd_ts = h5.timestamps[nd_flags]
sky_data = data[~nd_flags,:,:]
sky_ts = h5.timestamps[~nd_flags]
sky=[]
ts=[]
f=[]
pa = []
I=[]
Q=[]
U=[]
V=[]
for i in range(5):
	h5.select()
	h5.select(scans='track',targets=i,ants='ant2')
	nd_flags = h5.sensor['Antennas/ant1/nd_pin']
	data = h5.vis[~nd_flags,:,:]
	sky.append(data[:,:,0] + data[:,:,1])
	sky.append(data[:,:,0] - data[:,:,1])
	sky.append(2*np.real(data[:,:,2]))
	sky.append(2*np.imag(data[:,:,2]))
	ts.append(h5.timestamps[~nd_flags])
	pa.append(h5.parangle[~nd_flags])
for i in [0,2,3,4]:
	for j in range(4):
		f.append(scipy.interpolate.interp1d(ts[i],sky[i*4+j][:,200]))

ex = 100
x = [1,1,0,2]
y = [2,0,1,1]
d = []
for i in range(4): # the 4 pol params
	for t,s in zip(ts[1][ex:-ex],sky[1*4+i][ex:-ex,200]):
		top = f[0*4+i](t)
		bot = f[1*4+i](t)
		lef = f[2*4+i](t)
		rig = f[3*4+i](t)
		f2 = scipy.interpolate.interp2d(x,y,np.abs([top,bot,lef,rig]))
		d.append(s-f2(1,1))

n = len(d)/4

I = d[0:n]
Q = d[n:n*2]
U = d[n*2:n*3]
V = d[n*3:n*4]

