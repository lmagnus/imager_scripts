import katarchive
import katfile
import scipy
import numpy as np
from scikits.fitting import NonLinearLeastSquaresFit
import matplotlib.pyplot as plt

#n = '1322589857.h5' # fonax A
#n = '1354993862.h5' # Tau A Dec 2012
#n = '1322682474.h5' # Tau A Noc 2011
n = '1360707164.h5' # PKS 1903-802 second obs
f = katarchive.get_archived_products(n)

h5 = katfile.open(f[0])


sky_pa = [[],[],[],[],[]]

sky_ts=[[],[],[],[],[]]
sky_HH=[[],[],[],[],[]]
sky_VV=[[],[],[],[],[]]
sky_VH=[[],[],[],[],[]]

ant = 'ant3,ant4'

#gather data
for i in range(5):
	h5.select()
	h5.select(corrprods='cross',scans='track',targets=i,ants=ant)
	for s in h5.scans():
		data = h5.vis[:]
		HH = data[:,:,0]
		VV = data[:,:,1]
		VH = data[:,:,3]
		sky_ts[i].append(np.mean(h5.timestamps[~nd_flags]))
		sky_HH[i].append(np.median(HH[~nd_flags,:],0))
		sky_VV[i].append(np.median(VV[~nd_flags,:],0))
		sky_VH[i].append(np.median(VH[~nd_flags,:],0))
		sky_pa[i].append(np.mean(h5.parangle[~nd_flags]))

	sky_ts[i] = np.array(sky_ts[i])
	sky_HH[i] = np.array(sky_HH[i])
	sky_VV[i] = np.array(sky_VV[i])
	sky_VH[i] = np.array(sky_VH[i])



f_HH = []
f_VV = []
f_VH = []

#create the interpolation functions for off-source data
for i in [0,2,3,4]:
	f_HH.append(scipy.interpolate.interp1d(sky_ts[i],sky_HH[i],axis=0))
	f_VV.append(scipy.interpolate.interp1d(sky_ts[i],sky_VV[i],axis=0))
	f_VH.append(scipy.interpolate.interp1d(sky_ts[i],sky_VH[i],axis=0))

#remove sky + ground contribution
d = []
p = []
ts = []
for i,t in enumerate(sky_ts[1]):
	try:
		m_HH = np.mean(np.vstack((f_HH[0](t),f_HH[1](t),f_HH[2](t),f_HH[3](t))),0)
		m_VV = np.mean(np.vstack((f_VV[0](t),f_VV[1](t),f_VV[2](t),f_VV[3](t))),0)
		m_VH = np.mean(np.vstack((f_VH[0](t),f_VH[1](t),f_VH[2](t),f_VH[3](t))),0)
		d.append(sky_HH[1][i,:]-m_HH)
		d.append(sky_VV[1][i,:]-m_VV)
		d.append(sky_VH[1][i,:]-m_VH)
		p.append(sky_pa[1][i])
		ts.append(t)
	except:
		pass


pa = p

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
	VV = 0.5 * (p[0]+1j*p[1])*(p[0]-1j*p[1])*(S[0]+S[1]*np.cos(2*pa)+S[2]*np.sin(2*pa))
	VH = 0.5 * (p[0]+1j*p[1])*(p[2]-1j*p[3])*((p[4]+1j*p[5]-p[6]+1j*p[7])*S[0]-S[1]*np.sin(2*pa)+S[2]*np.cos(2*pa)+1j*S[3])
	HH = 0.5 * (p[2]+1j*p[3])*(p[2]-1j*p[3])*(S[0]-S[1]*np.cos(2*pa)-S[2]*np.sin(2*pa))
	I = HH + VV
	Q = VV - HH
	U = 2 * np.real(VH)
	V = 2 * np.imag(VH)
	return np.vstack((I.real,Q.real,U.real,V.real))

def calc_stokes2(p,pa):
	pa = np.array(pa)*(np.pi/180)
	p = np.array(p)
	Q = p[0] * np.sin(2 * pa + p[1])+p[2] 
	U = p[3] * np.sin(2 * pa + p[4])+p[5]
	V = p[6] * np.sin(2 * pa + p[7])+p[8]
	return np.vstack((Q.real,U.real,V.real))

D = []
D2 = []
for ch in [300]:#range(1024):
	freq = h5.channel_freqs[ch]/1e9
	s = IQUV_TAUA(freq)

	HH = np.array(d[0::3])[:,ch]
	VV = np.array(d[1::3])[:,ch]
	VH = np.array(d[2::3])[:,ch]
	I = HH+VV
	Q = VV - HH
	U = 2 * np.real(VH)
	V = 2 * np.imag(VH)

	fitter = NonLinearLeastSquaresFit(lambda p,pa:calc_stokes(p,pa,s),[1,0,1,0,0,0,0,0])
	fitter.fit(pa,np.vstack((I.real,Q.real,U.real,V.real)))
	fitter2 = NonLinearLeastSquaresFit(lambda p,pa:calc_stokes2(p,pa),[1,0,0,1,np.pi/4,0,1,0,0])
	fitter2.fit(pa,np.vstack((Q.real,U.real,V.real)))
	p = np.array(fitter.params)
	p2 = np.array(fitter2.params)
	D.append(np.sqrt(pow(p[4],2)+pow(p[5],2)))
	D.append(np.sqrt(pow(p[6],2)+pow(p[7],2)))
	D.append(np.arctan(p[5]/p[4])*180/np.pi)
	D.append(np.arctan(p[7]/p[6])*180/np.pi)
	D2.append(p2)
temp = calc_stokes(np.array(p),pa,s)
temp2 = calc_stokes2(np.array(p2),pa)
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
