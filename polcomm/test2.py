import katarchive
import katfile
import scipy
import numpy as np

#f = katarchive.get_archived_products('1322589857.h5')
f = katarchive.get_archived_products('1354993862.h5')

h5 = katfile.open(f[0])


sky_pa = [[],[],[],[],[]]
nd_ts=[[],[],[],[],[]]
nd_I=[[],[],[],[],[]]
nd_Q=[[],[],[],[],[]]
nd_U=[[],[],[],[],[]]
nd_V=[[],[],[],[],[]]

sky_ts=[[],[],[],[],[]]
sky_I=[[],[],[],[],[]]
sky_Q=[[],[],[],[],[]]
sky_U=[[],[],[],[],[]]
sky_V=[[],[],[],[],[]]

ant = 'ant2'

#gather data
for i in range(5):
	h5.select()
	h5.select(scans='track',targets=i,ants=ant)
	for s in h5.scans():
		data = h5.vis[:]
		nd_flags = h5.sensor['Antennas/'+ant+'/nd_pin']
		HH = data[:,:,0]
		VV = data[:,:,1]
		HV = data[:,:,2]
		if any(nd_flags):
			nd_ts[i].append(np.mean(h5.timestamps[nd_flags]))
			nd_I[i].append(np.median((HH[nd_flags,:] + VV[nd_flags,:]),0))
			nd_Q[i].append(np.median((HH[nd_flags,:] - VV[nd_flags,:]),0))
			nd_U[i].append(np.median((2*np.real(HV[nd_flags,:])),0))
			nd_V[i].append(np.median((2*np.imag(HV[nd_flags,:])),0))
		sky_ts[i].append(np.mean(h5.timestamps[~nd_flags]))
		sky_I[i].append(np.median((HH[~nd_flags,:] + VV[~nd_flags,:]),0))
		sky_Q[i].append(np.median((HH[~nd_flags,:] - VV[~nd_flags,:]),0))
		sky_U[i].append(np.median((2*np.real(HV[~nd_flags,:])),0))
		sky_V[i].append(np.median((2*np.imag(HV[~nd_flags,:])),0))
		sky_pa[i].append(np.mean(h5.parangle[~nd_flags]))
	nd_ts[i] = np.array(nd_ts[i])
	nd_I[i] = np.array(nd_I[i])
	nd_Q[i] = np.array(nd_Q[i])
	nd_U[i] = np.array(nd_U[i])
	nd_V[i] = np.array(nd_V[i])

	sky_ts[i] = np.array(sky_ts[i])
	sky_I[i] = np.array(sky_I[i])
	sky_Q[i] = np.array(sky_Q[i])
	sky_U[i] = np.array(sky_U[i])
	sky_V[i] = np.array(sky_V[i])

	
nd_f_I = []
nd_f_Q = []
nd_f_U = []
nd_f_V = []

#create the interpolation functions for the ND firings
for i in range(5):
	nd_f_I.append(scipy.interpolate.interp1d(nd_ts[i],nd_I[i],axis=0))
	nd_f_Q.append(scipy.interpolate.interp1d(nd_ts[i],nd_Q[i],axis=0))
	nd_f_U.append(scipy.interpolate.interp1d(nd_ts[i],nd_U[i],axis=0))
	nd_f_V.append(scipy.interpolate.interp1d(nd_ts[i],nd_V[i],axis=0))
cal_pa=[[],[],[],[],[]]
cal_ts=[[],[],[],[],[]]
cal_I=[[],[],[],[],[]]
cal_Q=[[],[],[],[],[]]
cal_U=[[],[],[],[],[]]
cal_V=[[],[],[],[],[]]

# calibrate the data


for i in range(5):
	for j,t in enumerate(sky_ts[i]):
		try:
			jump_I = nd_f_I[i](t)[:] - sky_I[i][j,:]
			jump_Q = nd_f_Q[i](t)[:] - sky_Q[i][j,:]
			jump_U = nd_f_U[i](t)[:] - sky_U[i][j,:]
			jump_V = nd_f_V[i](t)[:] - sky_V[i][j,:]
			cal_I[i].append(sky_I[i][j,:]/jump_I)
			cal_Q[i].append(sky_Q[i][j,:]/jump_Q)
			cal_U[i].append(sky_U[i][j,:]/jump_U)
			cal_V[i].append(sky_V[i][j,:]/jump_V)
			cal_ts[i].append(t)
			cal_pa[i].append(sky_pa[i][j])
			print i,j ,t,'first'
		except:
			pass
	
	cal_ts[i] = np.array(cal_ts[i])
	cal_I[i] = np.array(cal_I[i])
	cal_Q[i] = np.array(cal_Q[i])
	cal_U[i] = np.array(cal_U[i])
	cal_V[i] = np.array(cal_V[i])

#cal_ts = sky_ts
#cal_I = sky_I
#cal_Q = sky_Q
#cal_U = sky_U
#cal_V = sky_V
#cal_pa = sky_pa


f_I = []
f_Q = []
f_U = []
f_V = []

#create the interpolation functions for off-source data
for i in [0,2,3,4]:
	f_I.append(scipy.interpolate.interp1d(cal_ts[i],cal_I[i],axis=0))
	f_Q.append(scipy.interpolate.interp1d(cal_ts[i],cal_Q[i],axis=0))
	f_U.append(scipy.interpolate.interp1d(cal_ts[i],cal_U[i],axis=0))
	f_V.append(scipy.interpolate.interp1d(cal_ts[i],cal_V[i],axis=0))

#remove sky + ground contribution
x = [1,1,0,2]
y = [2,0,1,1]
d = []
p = []
ch = 850
ts = []
for i,t in enumerate(cal_ts[1]):
	try:
		m_I = np.mean([f_I[0](t)[ch],f_I[1](t)[ch],f_I[2](t)[ch],f_I[3](t)[ch]])
		m_Q = np.mean([f_Q[0](t)[ch],f_Q[1](t)[ch],f_Q[2](t)[ch],f_Q[3](t)[ch]])
		m_U = np.mean([f_U[0](t)[ch],f_U[1](t)[ch],f_U[2](t)[ch],f_U[3](t)[ch]])
		m_V = np.mean([f_V[0](t)[ch],f_V[1](t)[ch],f_V[2](t)[ch],f_V[3](t)[ch]])
		d.append(cal_I[1][i,ch]-m_I)
		d.append(cal_Q[1][i,ch]-m_Q)
		d.append(cal_U[1][i,ch]-m_U)
		d.append(cal_V[1][i,ch]-m_V)
		p.append(cal_pa[1][i])
		ts.append(t)
	except:
		pass

n = len(d)/4

I = d[0::4]
Q = d[1::4]
U = d[2::4]
V = d[3::4]

