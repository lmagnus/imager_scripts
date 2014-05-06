#! /usr/bin/python
# Heath checks of the system

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import os,traceback,re
import time
import numpy as np
from scipy import signal
import optparse

import katpoint
import katfile
import katarchive
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

n = '1362893705.h5'
f = katarchive.get_archived_products(n)
h5 = katfile.open(f[0])
pp = PdfPages('spectra.pdf')


h5.select()
cor = h5.corr_products
ants = h5.ants
num_ants = len(ants)
bp = np.array([t.tags.count('target') for t in h5.catalogue.targets]) == 1
bp = np.arange(len(bp))[bp]
#code to plot the cross phase ... fringes
fig = plt.figure(figsize=(24,20))
try:
    j=0
#plt.figure()
    n_channels = len(h5.channels)
    for pol in ('h','v'):
        h5.select(targets=bp,corrprods = 'cross',pol=pol)
        crosscorr = [(h5.inputs.index(inpA), h5.inputs.index(inpB)) for inpA, inpB in h5.corr_products]
        #extract the fringes
	power = 10 * np.log10(np.abs((h5.vis[:,:,:])))
        #For plotting the fringes
        fig.subplots_adjust(wspace=0., hspace=0.)
        #debug_here()
        for n, (indexA, indexB) in enumerate(crosscorr):
            subplot_index = (num_ants * indexA + indexB + 1) if pol == 'h' else (indexA + num_ants * indexB + 1)
            ax = fig.add_subplot(num_ants, num_ants, subplot_index)
            ax.plot(h5.channel_freqs,np.mean(power[:,:,n],0))
	    ax.set_xticks([])
            ax.set_yticks([])
            if pol == 'h':
                if indexA == 0:
                    ax.xaxis.set_label_position('top')
                    ax.set_xlabel(h5.inputs[indexB][3:],size='xx-large')
                if indexB == len(h5.ants) - 1:
                    ax.yaxis.set_label_position('right')
                    ax.set_ylabel(h5.inputs[indexA][3:], rotation='horizontal',size = 'xx-large')
            else:
                if indexA == 0:
                    ax.set_ylabel(h5.inputs[indexB][3:], rotation='horizontal',size='xx-large')
                if indexB == len(h5.ants) - 1:
                    ax.set_xlabel(h5.inputs[indexA][3:],size='xx-large')
    #plt.savefig(pp,format='pdf')
except KeyError , error:
    print 'Failed to read scans from File: ',fn,' with Key Error:',error
except ValueError , error:
    print 'Failed to read scans from File: ',fn,' with Value Error:',error
plt.savefig(pp,format='pdf')
plt.close('all')
pp.close()
