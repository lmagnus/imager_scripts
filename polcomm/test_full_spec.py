import katarchive
import katfile
import scipy

f = katarchive.get_archived_products('1354993862.h5')
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
for i in range(5):
	h5.select()
	h5.select(scans='track',targets=i,ants='ant2')
	nd_flags = h5.sensor['Antennas/ant1/nd_pin']
	sky.append(h5.vis[~nd_flags,:,:])
	ts.append(h5.timestamps[~nd_flags])
	pa.append(h5.parangle[~nd_flags])
for i in [0,2,3,4]:
	for j in [0,1,2]:
		f.append(scipy.interpolate.interp1d(ts[i],sky[i][:,:,j],axis=0))
c = sky[0].shape[1]
ex = 20
tr = sky[1].shape[0]-2*ex
x = [1,1,0,2]
y = [2,0,1,1]
d = np.zeros((3,tr,c),dtype=np.complex64)
for ch in range(c):
	for i in [0,1,2]:
		for j,t,s in zip(range(tr),ts[1][ex:-ex],sky[1][ex:-ex,ch,i]):
			top = f[0*3+i](t)[ch]
			bot = f[1*3+i](t)[ch]
			lef = f[2*3+i](t)[ch]
			rig = f[3*3+i](t)[ch]
			f2 = scipy.interpolate.interp2d(x,y,np.abs([top,bot,lef,rig]))
			d[i,j,ch] = (s-f2(1,1))[0] 
p = pa[1][ex:-ex]
n = len(d)/3

HH = d[0,:,:]
VV = d[1,:,:]
HV = d[2,:,:]

I = VV + HH
Q = VV - HH
U = 2 * np.real(HV)
V = 2 * np.imag(HV)


