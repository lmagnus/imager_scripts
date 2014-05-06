import katarchive
import katfile
import scipy
import numpy as np
from scikits.fitting import NonLinearLeastSquaresFit
import matplotlib.pyplot as plt

n = '1360845700.h5'
f = katarchive.get_archived_products(n)

h5 = katfile.open(f[0])


ant = 'ant4'

#gather data
h5.select(ants=ant,spw=4)

pa = 0

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


D = []
p = []
for s in h5.scans():
	data = h5.vis[:]
	HH_full = data[:,:,0]
	VV_full = data[:,:,1]
	VH_full = data[:,:,3]

	for ch in range(1024):
		freq = h5.channel_freqs[ch]/1e9
		s = [1,0,0,0]

		HH = HH_full[:,ch]
		VV = VV_full[:,ch]
		VH = VH_full[:,ch]
		I = HH+VV
		Q = VV - HH
		U = 2 * np.real(VH)
		V = 2 * np.imag(VH)
		GD = [1,0,1,0,0,0,0,0] # 
		fitter = NonLinearLeastSquaresFit(lambda p,pa:calc_stokes(p,pa,s),[1,0,1,0,0,0,0,0])
		fitter.fit(pa,np.vstack((I.real,Q.real,U.real,V.real)))
		p.append(np.array(fitter.params))
		D.append(np.sqrt(pow(p[-1][4],2)+pow(p[-1][5],2))/np.sqrt(pow(p[-1][0],2)+pow(p[-1][1],2)))
		D.append(np.sqrt(pow(p[-1][6],2)+pow(p[-1][7],2))/np.sqrt(pow(p[-1][2],2)+pow(p[-1][3],2)))
		D.append(np.arctan(p[-1][5]/p[-1][4])*180/np.pi)
		D.append(np.arctan(p[-1][7]/p[-1][6])*180/np.pi)
temp = calc_stokes(np.array(p[-1]),pa,s)
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
