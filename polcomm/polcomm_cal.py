import katarchive
import katfile
import scipy
import numpy as np
from scikits.fitting import NonLinearLeastSquaresFit
import matplotlib.pyplot as plt

#n = '1322589857.h5' # fonax A
n = '1354993862.h5' # Tau A Dec 2012
#n = '1322682474.h5' # Tau A Noc 2011
#n = '1360707164.h5' # PKS 1903-802 second obs
f = katarchive.get_archived_products(n)

h5 = katfile.open(f[0])


sky_pa = [[],[],[],[],[]]
nd_ts=[[],[],[],[],[]]
nd_HH=[[],[],[],[],[]]
nd_VV=[[],[],[],[],[]]
nd_VH=[[],[],[],[],[]]

sky_ts=[[],[],[],[],[]]
sky_HH=[[],[],[],[],[]]
sky_VV=[[],[],[],[],[]]
sky_VH=[[],[],[],[],[]]

ant = 'ant4'

#gather data
for i in range(5):
	h5.select()
	h5.select(scans='track',targets=i,ants=ant)
	for s in h5.scans():
		data = h5.vis[:]
		flags = h5.flags()[:]
		nd_flags = h5.sensor['Antennas/'+ant+'/nd_pin']
		HH = np.ma.array(data[:,:,0],mask = flags[:,:,0],fill_value=np.nan)
		VV = np.ma.array(data[:,:,1],mask = flags[:,:,1],fill_value=np.nan)
		VH = np.ma.array(data[:,:,3],mask = flags[:,:,3],fill_value=np.nan)
		if any(nd_flags):
			nd_ts[i].append(np.mean(h5.timestamps[nd_flags]))
			nd_HH[i].append(np.median(HH.filled()[nd_flags,:],0))
			nd_VV[i].append(np.median(VV.filled()[nd_flags,:],0))
			nd_VH[i].append(np.median(VH.filled()[nd_flags,:],0))
		sky_ts[i].append(np.mean(h5.timestamps[~nd_flags]))
		sky_HH[i].append(np.median(HH.filled()[~nd_flags,:],0))
		sky_VV[i].append(np.median(VV.filled()[~nd_flags,:],0))
		sky_VH[i].append(np.median(VH.filled()[~nd_flags,:],0))
		sky_pa[i].append(np.mean(h5.parangle[~nd_flags]))
	nd_ts[i] = np.array(nd_ts[i])
	nd_HH[i] = np.array(nd_HH[i])
	nd_VV[i] = np.array(nd_VV[i])
	nd_VH[i] = np.array(nd_VH[i])

	sky_ts[i] = np.array(sky_ts[i])
	sky_HH[i] = np.array(sky_HH[i])
	sky_VV[i] = np.array(sky_VV[i])
	sky_VH[i] = np.array(sky_VH[i])

	
nd_f_HH = []
nd_f_VV = []
nd_f_VH = []
nd_f_V = []

#create the interpolation functions for the ND firings
for i in range(5):
	nd_f_HH.append(scipy.interpolate.interp1d(nd_ts[i],nd_HH[i],axis=0))
	nd_f_VV.append(scipy.interpolate.interp1d(nd_ts[i],nd_VV[i],axis=0))
	nd_f_VH.append(scipy.interpolate.interp1d(nd_ts[i],nd_VH[i],axis=0))
cal_pa=[[],[],[],[],[]]
cal_ts=[[],[],[],[],[]]
cal_HH=[[],[],[],[],[]]
cal_VV=[[],[],[],[],[]]
cal_VH=[[],[],[],[],[]]

# calibrate the data


for i in range(5):
	for j,t in enumerate(sky_ts[i]):
		try:
			jump_HH = nd_f_HH[i](t)[:] - sky_HH[i][j,:]
			jump_VV = nd_f_VV[i](t)[:] - sky_VV[i][j,:]
			jump_VH = nd_f_VH[i](t)[:] - sky_VH[i][j,:]
			#if any(jump_VV ==0) or any(jump_HH == 0) or any(jump_VH == 0): 
			#	continue
			cal_HH[i].append(sky_HH[i][j,:]/jump_HH)
			cal_VV[i].append(sky_VV[i][j,:]/jump_VV)
			cal_VH[i].append(sky_VH[i][j,:]/jump_VH)
			cal_ts[i].append(t)
			cal_pa[i].append(sky_pa[i][j])
		except:
			pass
	
	cal_ts[i] = np.array(cal_ts[i])
	cal_HH[i] = np.array(cal_HH[i])
	cal_VV[i] = np.array(cal_VV[i])
	cal_VH[i] = np.array(cal_VH[i])


f_HH = []
f_VV = []
f_VH = []

#create the interpolation functions for off-source data
for i in [0,2,3,4]:
	f_HH.append(scipy.interpolate.interp1d(cal_ts[i],cal_HH[i],axis=0))
	f_VV.append(scipy.interpolate.interp1d(cal_ts[i],cal_VV[i],axis=0))
	f_VH.append(scipy.interpolate.interp1d(cal_ts[i],cal_VH[i],axis=0))

#remove sky + ground contribution
d = []
p = []
ts = []
for i,t in enumerate(cal_ts[1]):
	try:
		m_HH = np.mean(np.vstack((f_HH[0](t),f_HH[1](t),f_HH[2](t),f_HH[3](t))),0)
		m_VV = np.mean(np.vstack((f_VV[0](t),f_VV[1](t),f_VV[2](t),f_VV[3](t))),0)
		m_VH = np.mean(np.vstack((f_VH[0](t),f_VH[1](t),f_VH[2](t),f_VH[3](t))),0)
		d.append(cal_HH[1][i,:]-m_HH)
		d.append(cal_VV[1][i,:]-m_VV)
		d.append(cal_VH[1][i,:]-m_VH)
		p.append(cal_pa[1][i])
		ts.append(t)
	except:
		pass


pa = np.array(p)

#freq in GHz
def IQUV_TAUA(freq):
	mI=0
	cI=1
	mQ=((-0.0144)-(-0.0133))/(1.81-1.97)
	cQ=(-0.0144)-1.81*mQ
	mU=((-0.0188)-(-0.0238))/(1.81-1.97)
	cU=(-0.0188)-1.81*mU
	mV=0
	cV=0
	I=mI*freq+cI
	Q=mQ*freq+cQ
	U=mU*freq+cU
	V=mV*freq+cV    
	rotby=0.0*np.pi/180.0;#TESTINGHERE
#    rotby=-7.5*np.pi/180.0;#TESTINGHERE
	rP=np.sqrt(Q*Q+U*U)
	tP=np.arctan2(U,Q)
	Q=rP*np.cos(tP+rotby)
	U=rP*np.sin(tP+rotby)
	return [I,Q,U,V]
def calc_stokes(p,pa,S):
	pa = np.array(pa)*(np.pi/180)
	p = np.array(p)
	S = np.array(S)
	'''
	VV = 0.5 * (p[0]+1j*p[1])*(p[0]-1j*p[1])*(S[0]+S[1]*np.cos(2*pa)+S[2]*np.sin(2*pa))
	VH = 0.5 * (p[0]+1j*p[1])*(p[2]-1j*p[3])*((p[4]+1j*p[5]-p[6]+1j*p[7])*S[0]-S[1]*np.sin(2*pa)+S[2]*np.cos(2*pa)+1j*S[3])
	HV = 0.5 * (p[2]+1j*p[3])*(p[0]-1j*p[1])*((p[4]-1j*p[5]-p[6]-1j*p[7])*S[0]-S[1]*np.sin(2*pa)+S[2]*np.cos(2*pa)-1j*S[3])
	HH = 0.5 * (p[2]+1j*p[3])*(p[2]-1j*p[3])*(S[0]-S[1]*np.cos(2*pa)-S[2]*np.sin(2*pa))
	'''
	gp = np.complex(p[0],0)#,p[1])
	gq = np.complex(p[2],0)#,p[3])
	dp = np.complex(p[4],p[5])
	dq = np.complex(p[6],p[7])
	VV = 0.5 * gp * gp.conjugate() *(S[0]+S[1]*np.cos(2*pa)+S[2]*np.sin(2*pa))
	VH = 0.5 * gp * gq.conjugate() *((dp - dq.conjugate())*S[0]-S[1]*np.sin(2*pa)+S[2]*np.cos(2*pa)+1j*S[3])
	HV = 0.5 * gq * gp.conjugate() *((dp.conjugate() - dq)*S[0]-S[1]*np.sin(2*pa)+S[2]*np.cos(2*pa)-1j*S[3])
	HH = 0.5 * gq * gq.conjugate() *(S[0]-S[1]*np.cos(2*pa)-S[2]*np.sin(2*pa))
	I = HH + VV
	Q = VV - HH
	U = VH + HV
	V = -1j*(VH - HV)
	return np.vstack((I.real,Q.real,U.real,V.real))

def calc_stokes2(p,pa):
	pa = np.array(pa)*(np.pi/180)
	p = np.array(p)
	Q = p[0] * np.sin(2 * (pa + p[1]))+p[2] 
	U = p[3] * np.sin(2 * (pa + p[4]))+p[5]
	V = p[6] * np.sin(2 * (pa + p[7]))+p[8]
	return np.vstack((Q.real,U.real,V.real))

D = []
D2 = []
f = []
for ch in range(200,419):
	freq = h5.channel_freqs[ch]/1e9
	s = IQUV_TAUA(freq)
	#s = [905.66612757174755, 1605.432643331761, 283.08109072949952,0]
	HH = np.array(d[0::3])[:,ch]
	VV = np.array(d[1::3])[:,ch]
	VH = np.array(d[2::3])[:,ch]
	ind = np.logical_or(np.logical_or(np.isnan(HH),np.isnan(VV)),np.isnan(VH))
	if np.all(ind) or pa[~ind].shape[0] < 10:
		continue
	I = HH[~ind]+VV[~ind]
	Q = VV[~ind] - HH[~ind]
	U = 2 * np.real(VH[~ind])
	V = 2 * np.imag(VH[~ind])

	fitter = NonLinearLeastSquaresFit(lambda p,pa:calc_stokes(p,pa,s),[1,0,1,0,0,0,0,0])
	fitter.fit(pa[~ind],np.vstack((I.real,Q.real,U.real,V.real)))
	fitter2 = NonLinearLeastSquaresFit(lambda p,pa:calc_stokes2(p,pa),[1,0,0,1,np.pi/4,0,1,0,0])
	fitter2.fit(pa[~ind],np.vstack((Q.real,U.real,V.real)))
	p = np.array(fitter.params)
	p2 = np.array(fitter2.params)
	if p2[0]<0:
		p2[0] = p2[0]*-1
		p2[1] = p2[1] + np.pi/2
	if p2[3]<0:
		p2[3] = p2[3]*-1
		p2[4] = p2[4] + np.pi/2
	if p2[6]<0:
		p2[6] = p2[6]*-1
		p2[7] = p2[7] + np.pi/2
	D.append(p)
	D2.append(p2)
	f.append(h5.channel_freqs[ch])
temp = calc_stokes(np.array(p),pa,s)
temp2 = calc_stokes2(np.array(p2),pa)
d = np.vstack(D)
d2 = np.vstack(D2)
'''
plt.plot(pa,HH+VV,'ro')
plt.plot(pa,VV - HH,'bo')
plt.plot(pa,2*np.real(VH),'yo')
plt.plot(pa,2*np.imag(VH),'go')
plt.plot(pa,temp[0,:],'r+')
plt.plot(pa,temp[1,:],'b+')
plt.plot(pa,temp[2,:],'y+')
plt.plot(pa,temp[3,:],'g+')
plt.plot(pa,temp2[0,:],'b*')
plt.plot(pa,temp2[1,:],'y*')
plt.plot(pa,temp2[2,:],'g*')
'''
