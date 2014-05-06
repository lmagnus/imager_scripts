#! /usr/bin/python
# Analysis of the bandpass shape from the
# Heath checks of the system

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import os,traceback,re
import time
import csv
import numpy as np
from scipy import signal

import katfile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

pp = PdfPages('bandpass_summary.pdf')

#plt.savefig(pp,format='pdf')
fl = open('test_spec.csv','r')
r = csv.reader(fl)
for row in r:
        centre_freq = float(row[2])
        start_freq = centre_freq - 128e6 if centre_freq - 128e6 > 1200e6 else 1200e6
        end_freq = centre_freq + 128e6 if centre_freq + 128e6 < 1900e6 else 1900e6
        df = (end_freq - start_freq)/512.0
        f_ind = end_freq - np.arange(512)*df
#debug_here()
        spec = row[3][1:-1].split(',')
        spec = np.asarray([np.complex(rw) for rw in spec])
        full_spec = np.concatenate((spec,np.zeros(len(f_ind)-39),np.conjugate(spec[-1:0:-1])))
        #full_spec[0] = 0
	time_arr.append(row[0])
	ant_arr.append(row[1])
	cent_arr.append(centre_freq)
	spec_arr.append(spec)
#plt.savefig(pp,format='pdf')
fl.close()
#pp.close()
plt.figure(figsize=(80,40))
time_arr = np.asarray(time_arr)
ant_arr = np.asarray(ant_arr)
cent_arr = np.asarray(cent_arr)
spec_arr = np.asarray(spec_arr)
for a in range(7):
	for p_i,p in enumerate(('h','v')):
		for f_i,f in enumerate(centre_f):
			start_freq = f - 128e6 if f - 128e6 > 1200e6 else 1200e6
			end_freq = f + 128e6 if f + 128e6 < 1900e6 else 1900e6
			df = (end_freq - start_freq)/512.0
			f_ind = end_freq - np.arange(512)*df
			index = 1 + (a * 12)+ f_i + p_i * 6
			plt.subplot(7,12,index)
			curr_ant = np.asarray(ants)[np.asarray([str(a+1)+p in an for an in ants])]
			curr_data = spec_arr[np.logical_and(cent_arr == f , ant_arr == curr_ant)]
			for data in curr_data:
				full_spec = np.concatenate((data,np.zeros(len(f_ind)-39),np.conjugate(data[-1:0:-1])))
				full_spec[0] = 0
				pl_data = np.real(signal.fft(full_spec))
				col = 'k' if pl_data.max() - pl_data.min() < 3 else 'r'
				plt.plot(f_ind,pl_data,color = col)
				plt.title(str(a+1)+p+p)
plt.savefig(pp,format='pdf')
pp.close()
