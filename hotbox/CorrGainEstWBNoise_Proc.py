# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 13:57:40 2011

@author: paulh
"""

# ---------------
# IMPORT PACKAGES
# ---------------

import numpy as np
from os import path
import matplotlib.pyplot as plt
from scipy import interpolate, log10, savez, signal, stats
import roach_hsys
from plot_tools import plot_to_subplots
# ----------------
# CUSTOM FUNCTIONS
# ----------------

def savexy(x_dB,y_dB,fname = 'CorrGainEst'):
    savez(fname,x_dB=x_dB,y_dB=y_dB)
    
def fitxy(x,y,delta_x):
    x_fit_min = np.amax(x[0,:,:])
    x_fit_max = np.amin(x[-1,:,:])
    x_fit = np.arange(x_fit_min,x_fit_max,delta_x)
    y_fit = np.zeros((len(x_fit),np.shape(x)[1],np.shape(x)[2]))
    for idx_ch in range(np.shape(x)[2]):
        for idx_l in range(np.shape(x)[1]):
            y_fit[:,idx_l,idx_ch] = np.interp(x_fit,x[:,idx_l,idx_ch], \
                                    y[:,idx_l,idx_ch])
    
    return x_fit,y_fit
    
def quant_norm_pmf(levels, mean=0.0, std=1.0):
    """Probability mass function of quantised normal variable."""
    levels = np.asarray(levels)
    edges = np.r_[-np.inf, levels[:-1] + np.diff(levels) / 2., np.inf]
    return stats.norm.cdf(edges[1:], loc=mean, scale=std) - \
        stats.norm.cdf(edges[:-1], loc=mean, scale=std)

def quant_norm_var(levels, mean=0.0, std=1.0):
    """Variance of quantised normal variable."""
    pmf = quant_norm_pmf(levels, mean, std)
    quant_mean = np.dot(levels, pmf)
    return np.dot((levels - quant_mean) ** 2, pmf)

    
collapse2d = lambda x : x.ravel().reshape((np.prod(x.shape[:-1]),x.shape[-1]))    

# -----------------
# LOAD ALL THE DATA
# -----------------
VNA_FileDir = 'C:\\CurrentProjects\\DataAnalalysis\\FFCorrGainEst\\vna'
VNA_FileName = ['Splitter_Ch1DivCh8.csv','Splitter_Ch2DivCh8.csv', \
                'Splitter_Ch3DivCh8.csv','Splitter_Ch4DivCh8.csv']

path_data = 'C://svnSystemEngineering//KAT-7//Telescope//6 Integration & verification//Analysis//Measurements//' + \
            'FF_SA_ADCPFB_XEng'
fname_npy = ['EstCorrGain_NOISE_20110303_1426.npy', \
             'EstCorrGain_NOISE_20110301_1559.npy', \
             'EstCorrGain_NOISE_20110302_1246.npy']

# Step 1: Load recorded dataset
for idx_f,f_npy in enumerate(fname_npy):
    if idx_f == 0:
        dataset = np.load(path_data + path.sep + f_npy)
        data = dataset[0]['data']
    else:
        nextdataset = np.load(path_data + path.sep + f_npy)
        data = np.hstack((data,nextdataset[0]['data']))

if idx_f > 0:
    nextdataset = 0
        
if np.rank(data['SA_ChP_dBm']) == 1:
    Src_NrLevels = 1
else:
    Src_NrLevels = np.shape(data['SA_ChP_dBm'])[0]

FEng_NrCh = int(dataset[0]['FEng']['NrCh'])
FEng_f_Hz = dataset[0]['FEng']['f_Hz']
Sys_IpChId = dataset[0]['Sys']['IpChId']
Sys_NrIpCh = len(Sys_IpChId)
do_correlator = dataset[0]['Proc']['do_correlator']
do_snapshot = dataset[0]['Proc']['do_snapshot']

# Step 2: Extract the spectrum analyzer channel power for all input power levels
if np.rank(data['SA_ChP_dBm']) == 1:
    SA_ChP_Base_dBm = data['SA_ChP_dBm'][np.newaxis,:,np.newaxis].repeat(Sys_NrIpCh,2)
else:
    SA_ChP_Base_dBm = data['SA_ChP_dBm'][:,:,np.newaxis].repeat(Sys_NrIpCh,2)

SA_GainOffset_dB = np.zeros((FEng_NrCh,Sys_NrIpCh))
for idx_f,fname in enumerate(VNA_FileName):
    vna_data = np.loadtxt(VNA_FileDir+path.sep+fname,skiprows = 3)
    tck = interpolate.splrep(vna_data[:,0],vna_data[:,1],s=0) # Get cubic spline t,c,k tuple representing the knot-points,cofficients and order (taken from http://docs.scipy.org/doc/scipy-0.8.x/reference/tutorial/interpolate.html)
    SA_GainOffset_dB[:,idx_f] = interpolate.splev(FEng_f_Hz,tck)

SA_GainOffset_dB = SA_GainOffset_dB[:,:,np.newaxis].repeat(Src_NrLevels,2).transpose(2,0,1)

# Step 3: Extract snapshot data
if do_snapshot == 1:
    # 3.1: ADC snapshot
    ADC_power_dBL = 20*log10(data['ADC_std'])
    ADC_mean = data['ADC_mean']
    if np.rank(data['SA_ChP_dBm']) == 1:
        ADC_power_dBL = ADC_power_dBL[np.newaxis,:]
        ADC_mean = ADC_mean[np.newaxis,:]
        ADC_pfb_dBL = 20*log10(data['ADC_pfb'][np.newaxis,:,:])
    else:
        ADC_pfb_dBL = 20*log10(data['ADC_pfb'])
    
    # 3.2: Requantizer I channel snapshot data
    if np.rank(data['SA_ChP_dBm']) == 1:
        ReQ_P_dBL = 10*log10((data['ReQ_I_std']**2 + data['ReQ_Q_std']**2)[np.newaxis,:,:])
        ReQ_I_mean = data['ReQ_I_mean'][np.newaxis,:,:]
        ReQ_Q_mean = data['ReQ_Q_mean'][np.newaxis,:,:]
    else:
        ReQ_P_dBL = 10*log10((data['ReQ_I_std']**2 + data['ReQ_Q_std']**2))
        ReQ_I_mean = data['ReQ_I_mean']
        ReQ_Q_mean = data['ReQ_Q_mean']
    ReQ_P_dBL[np.isinf(ReQ_P_dBL)] = ReQ_P_dBL[np.isfinite(ReQ_P_dBL)].min()
        
# Step 4: Load correlator data
if do_correlator == 1:
    XEng_fname_out = data['XEng_fname_out']
    if np.rank(data['SA_ChP_dBm']) == 1:
        XEng_fname_out = list([XEng_fname_out])
        
    XEng_P_dBL = np.zeros((len(XEng_fname_out),FEng_NrCh,len(Sys_IpChId)))
    for idx_f,fname_h5 in enumerate(XEng_fname_out):
        for idx_s,corr_str in enumerate(Sys_IpChId):
            print('Working on',fname_h5,corr_str)
            XEng_raw_data = roach_hsys.read_dbe_correlator_h5(path_data + \
            path.sep + fname_h5 + '.h5',corr_str+corr_str)[0]*(1./512) # corr_str + corr_str 
            # yields autocorrelation (integrated power). The k7w already corrects for XEng 
            # accumulation
            XEng_P_dBL[idx_f,:,idx_s] = 10*np.log10(np.median(np.real(XEng_raw_data),0)) 
            
    XEng_raw_data = 0
    
# ---------------------
# PROCESSING PARAMETERS
# ---------------------

# Define censoring
idx_ip_ch = range(0,4)
idx_feng_ch = range(50,350)
# Define lpf for smoothing of recorded data
lpf_fn_cutoff = 0.05
lpf_order = 21
# fitting parameters
delta_x_dB = 0.1
# Define correction filename
fname_vanvleck = 'CorrGainEst_SA_Tot.npz'
# Define what processing to do
do_vna_correction = True
do_vanvleck = False
do_adc_correction = False
do_filter_sa = True
do_filter_pfb = True
do_filter_req = True
do_filter_xeng = True
do_filter_gain = False
do_save_file = True
# Define if new figures are to be drawn
new_figure = True    

# ----------
# PROCESSING
# ----------

if new_figure:
    next_fig_num = 0
else:
    next_fig_num = max(plt.get_fignums()) + 1

# Step 0: Define lpf
lpf_groupdelay = int((lpf_order-1)/2)
lpf_b = signal.firwin(lpf_order,lpf_fn_cutoff,window='hamming')
lpf = lambda x , axis_proc : np.roll(signal.lfilter(lpf_b,1,x,axis_proc),-lpf_groupdelay,axis_proc)
# Lambda function that applies a low-pass filter with the specified cut-off and order and reverses
# the filter group delay. Data will be corrupt at the beginning and end for a length equal to group_delay
#np.roll(signal.lfilter(lpf,1,x,axis),-lpf_groupdelay,axis)

# Step 1: Do vna correction, if required
if do_vna_correction:
    SA_ChP_dBm = SA_ChP_Base_dBm + SA_GainOffset_dB
else:
    SA_ChP_dBm = SA_ChP_Base_dBm
    
# Step 2: Calculate total input channel power and find the sorted index to ensure that data is represented
# for increasing channel power
SA_TotP_dBm = 10*log10(np.sum(10**(0.1*SA_ChP_dBm),1))
idx_sort = np.argsort(SA_TotP_dBm[:,0])

ADC_mean = ADC_mean[idx_sort,:]
ADC_pfb_dBL = ADC_pfb_dBL[idx_sort,:,:]
ADC_power_dBL = ADC_power_dBL[idx_sort,:]

ReQ_I_mean = ReQ_I_mean[idx_sort,:,:]
ReQ_P_dBL = ReQ_P_dBL[idx_sort,:,:]
ReQ_Q_mean = ReQ_Q_mean[idx_sort,:,:]

SA_ChP_Base_dBm = SA_ChP_Base_dBm[idx_sort,:,:]
SA_GainOffset_dB = SA_GainOffset_dB[idx_sort,:,:]
SA_ChP_dBm = SA_ChP_dBm[idx_sort,:,:]
SA_TotP_dBm = SA_TotP_dBm[idx_sort,:]

XEng_fname_out = XEng_fname_out[idx_sort]
XEng_P_dBL = XEng_P_dBL[idx_sort,:,:]
data = data[idx_sort]

# Filter input data if so required
if do_filter_sa:
    SA_ChP_dBm = lpf(SA_ChP_dBm,1)
    
if do_filter_pfb:
    ADC_pfb_dBL = lpf(ADC_pfb_dBL,1)

if do_filter_req:
    ReQ_P_dBL = lpf(ReQ_P_dBL,1)
    ReQ_I_mean = lpf(ReQ_I_mean,1)
    ReQ_Q_mean = lpf(ReQ_Q_mean,1)
    
if do_filter_xeng:
    XEng_P_dBL = lpf(XEng_P_dBL,1)

# Step 3: Characterize ADC response
SA_ADC_gain_dB = ADC_power_dBL - SA_TotP_dBm

next_fig_num += 1
plot_to_subplots([SA_TotP_dBm[:,idx_ip_ch]]*3, \
                 [ADC_power_dBL[:,idx_ip_ch],ADC_mean[:,idx_ip_ch],SA_ADC_gain_dB[:,idx_ip_ch]], \
                 marker = '.',title_txt = ['ADC response'], \
                 xlabel_txt = [None,None,'Total input power [dBm]'], \
                 ylabel_txt = ['Variance [dBL]','Mean [levels]','Gain [dB]'], \
                 legend_txt = [[list(Sys_IpChId[idx_ip_ch])],None,None], \
                 fig = next_fig_num)

if do_adc_correction:
    SA_ChP_dBm += SA_ADC_gain_dB[:,:,np.newaxis].repeat(FEng_NrCh,2).transpose([0,2,1])

SA_PFB_gain_dB = ADC_pfb_dBL - SA_ChP_dBm
if do_filter_gain:
    SA_PFB_gain_dB = lpf(SA_PFB_gain_dB,1)   
              
next_fig_num += 1
plot_to_subplots([SA_ChP_dBm[:,idx_feng_ch,idx_ip].T for idx_ip in idx_ip_ch], \
                 [SA_PFB_gain_dB[:,idx_feng_ch,idx_ip].T for idx_ip in idx_ip_ch], \
                 linestyle = '-',marker = '.', \
                 xlabel_txt = 'Input channel power [dBm]', \
                 ylabel_txt = list(Sys_IpChId[idx_ip_ch]), \
                 title_txt = ['ADC frequency/amplitude transfer function'], \
                 fig = next_fig_num,sp_cols = 2)

next_fig_num += 1
plot_to_subplots([FEng_f_Hz[idx_feng_ch]*1e-6]*len(idx_ip_ch), \
                 [SA_PFB_gain_dB[:,idx_feng_ch,idx_ip].T for idx_ip in idx_ip_ch], \
                 linestyle = '-',marker = '.', \
                 xlabel_txt = 'Channel frequency [MHz]', \
                 ylabel_txt = list(Sys_IpChId[idx_ip_ch]), \
                 title_txt = ['ADC frequency/amplitude transfer function'], \
                 fig = next_fig_num,sp_cols = 2)
                 
# Step 3: Characterize requantizer response
SA_ReQ_gain_dB = ReQ_P_dBL - SA_ChP_dBm
PFB_ReQ_gain_dB = ReQ_P_dBL - ADC_pfb_dBL
if do_filter_gain:
    SA_ReQ_gain_dB = lpf(SA_ReQ_gain_dB,1)
    PFB_ReQ_gain_dB = lpf(PFB_ReQ_gain_dB,1)
    
next_fig_num += 1
plot_to_subplots([SA_ChP_dBm[:,idx_feng_ch,idx_ip].T for idx_ip in idx_ip_ch], \
                 [SA_ReQ_gain_dB[:,idx_feng_ch,idx_ip].T for idx_ip in idx_ip_ch], \
                 linestyle = '-',marker = '.', \
                 xlabel_txt = 'Input channel power [dBm]', \
                 ylabel_txt = list(Sys_IpChId[idx_ip_ch]), \
                 title_txt = ['ReQ - SA transfer function'], \
                 fig = next_fig_num,sp_cols = 2)

next_fig_num += 1
plot_to_subplots([ADC_pfb_dBL[:,idx_feng_ch,idx_ip].T for idx_ip in idx_ip_ch], \
                 [PFB_ReQ_gain_dB[:,idx_feng_ch,idx_ip].T for idx_ip in idx_ip_ch], \
                 linestyle = '-',marker = '.', \
                 xlabel_txt = 'Input channel power [dBm]', \
                 ylabel_txt = list(Sys_IpChId[idx_ip_ch]), \
                 title_txt = ['ReQ -PFB function'], \
                 fig = next_fig_num,sp_cols = 2)

# Step 3: Characterize correlator response
SA_XEng_gain_dB = XEng_P_dBL - SA_ChP_dBm
PFB_XEng_gain_dB = XEng_P_dBL - ADC_pfb_dBL
ReQ_XEng_gain_dB = XEng_P_dBL - ReQ_P_dBL
if do_filter_gain:
    SA_ReQ_gain_dB = lpf(SA_ReQ_gain_dB,1)
    PFB_ReQ_gain_dB = lpf(PFB_ReQ_gain_dB,1)
    ReQ_XEng_gain_dB = lpf(ReQ_XEng_gain_dB,1)
    
SA_ChP_fit_dBm,XEng_SA_P_fit_dBL = fitxy(SA_ChP_dBm[:,idx_feng_ch,:][:,:,idx_ip_ch], \
                                   XEng_P_dBL[:,idx_feng_ch,:][:,:,idx_ip_ch],delta_x_dB)
XEng_SA_ChMedian_dBL = np.median(XEng_SA_P_fit_dBL,1)
XEng_SA_TotMedian_dBL = np.median(collapse2d(XEng_SA_P_fit_dBL.transpose([1,2,0])),0)

ADC_pfb_fit_dBL,XEng_PFB_P_fit_dBL = fitxy(ADC_pfb_dBL[:,idx_feng_ch,:][:,:,idx_ip_ch], \
                                     XEng_P_dBL[:,idx_feng_ch,:][:,:,idx_ip_ch],delta_x_dB)
XEng_PFB_ChMedian_dBL = np.median(XEng_PFB_P_fit_dBL,1)
XEng_PFB_TotMedian_dBL = np.median(collapse2d(XEng_PFB_P_fit_dBL.transpose([1,2,0])),0)
                  
levels = np.arange(-7., 8.)
delta = levels[1] - levels[0]
sigma = np.arange(0.2, 10.05, 0.05)
theory_x_dB = 20*log10(sigma)
theory_y_dB = 10*log10(2 * np.array([quant_norm_var(levels, std=s) for s in sigma]))                   

y_normSA_dB = min([max(theory_y_dB),max(XEng_SA_TotMedian_dBL)])
theory_xSA_dB = theory_x_dB + \
                SA_ChP_fit_dBm[np.argmin(abs(XEng_SA_TotMedian_dBL - y_normSA_dB))] - \
                theory_x_dB[np.argmin(abs(theory_y_dB - y_normSA_dB))]

y_normPFB_dB = min([max(theory_y_dB),max(XEng_PFB_TotMedian_dBL)])
theory_xPFB_dB = theory_x_dB + \
                ADC_pfb_fit_dBL[np.argmin(abs(XEng_PFB_TotMedian_dBL - y_normPFB_dB))] - \
                theory_x_dB[np.argmin(abs(theory_y_dB - y_normPFB_dB))]

next_fig_num += 1
plot_to_subplots([SA_ChP_dBm[:,idx_feng_ch,idx_ip].T for idx_ip in idx_ip_ch], \
                 [SA_XEng_gain_dB[:,idx_feng_ch,idx_ip].T for idx_ip in idx_ip_ch], \
                 linestyle = '-',marker = '.', \
                 xlabel_txt = 'Input channel power [dBm]', \
                 ylabel_txt = list(Sys_IpChId[idx_ip_ch]), \
                 title_txt = ['XEng - SA transfer function'], \
                 fig = next_fig_num,sp_cols = 2)

next_fig_num += 1
plot_to_subplots([ADC_pfb_dBL[:,idx_feng_ch,idx_ip].T for idx_ip in idx_ip_ch], \
                 [PFB_XEng_gain_dB[:,idx_feng_ch,idx_ip].T for idx_ip in idx_ip_ch], \
                 linestyle = '-',marker = '.', \
                 xlabel_txt = 'Input channel power [dBL]', \
                 ylabel_txt = list(Sys_IpChId[idx_ip_ch]), \
                 title_txt = ['XEng - PFB transfer function'], \
                 fig = next_fig_num,sp_cols = 2)

next_fig_num += 1
plot_to_subplots([ReQ_P_dBL[:,idx_feng_ch,idx_ip].T for idx_ip in idx_ip_ch], \
                 [ReQ_XEng_gain_dB[:,idx_feng_ch,idx_ip].T for idx_ip in idx_ip_ch], \
                 linestyle = '-',marker = '.', \
                 xlabel_txt = 'Input channel power [dBL]', \
                 ylabel_txt = list(Sys_IpChId[idx_ip_ch]), \
                 title_txt = ['XEng - ReQ transfer function'], \
                 fig = next_fig_num,sp_cols = 2)

next_fig_num += 1
plot_to_subplots([[SA_ChP_fit_dBm,SA_ChP_fit_dBm,theory_xSA_dB], \
                 [ADC_pfb_fit_dBL,ADC_pfb_fit_dBL,theory_xPFB_dB], \
                 [SA_ChP_fit_dBm,SA_ChP_fit_dBm,theory_xSA_dB], \
                 [ADC_pfb_fit_dBL,ADC_pfb_fit_dBL,theory_xPFB_dB]], \
                 [[XEng_SA_ChMedian_dBL,XEng_SA_TotMedian_dBL,theory_y_dB], \
                 [XEng_PFB_ChMedian_dBL,XEng_PFB_TotMedian_dBL,theory_y_dB], \
                 [XEng_SA_ChMedian_dBL-SA_ChP_fit_dBm[:,np.newaxis].repeat(len(idx_ip_ch),1), \
                 XEng_SA_TotMedian_dBL-SA_ChP_fit_dBm,theory_y_dB-theory_xSA_dB], \
                 [XEng_PFB_ChMedian_dBL-ADC_pfb_fit_dBL[:,np.newaxis].repeat(len(idx_ip_ch),1), \
                 XEng_PFB_TotMedian_dBL-ADC_pfb_fit_dBL,theory_y_dB-theory_xPFB_dB]], \
                 marker = [[None,'.','x'],[None,'.','x'],[None,'.','x'],[None,'.','x']], \
                 xlabel_txt = [None,None,'SA input channel power [dBm]','ADC PFB channel power [dBL]'], \
                 ylabel_txt = ['Correlator power [dBL]',None,'Gain [dB]',None], \
                 title_txt  = ['Correlator vs SA','Correlator vs ADC PFB'], \
                 legend_txt = [[list(Sys_IpChId[idx_ip_ch])],[None,'Total','Theory']], \
                 fig = next_fig_num,sp_cols = 2)

# -------------------------
# SAVE CHARACTERISTIC CURVE
# -------------------------

if do_save_file:
    for idx_ip in idx_ip_ch:
        savestr = 'CorrGainEst_SA_Ch%s'%Sys_IpChId[idx_ip]
        savexy(SA_ChP_fit_dBm,XEng_SA_ChMedian_dBL[:,idx_ip],savestr)
        
    savexy(SA_ChP_fit_dBm,XEng_SA_TotMedian_dBL,'CorrGainEst_SA_Tot')
    savexy(ADC_pfb_fit_dBL,XEng_PFB_TotMedian_dBL,'CorrGainEst_PFB_Tot')
    
# --------------
# TEST LINEARITY
# --------------
XEng_test_dBL = np.zeros((len(XEng_fname_out),FEng_NrCh,len(Sys_IpChId)))
if do_vanvleck:
    x_inv_dB,y_inv_dB = roach_hsys.get_van_vleck_inverse(fname_vanvleck)

for idx_f,fname_h5 in enumerate(XEng_fname_out):
    for idx_s,corr_str in enumerate(Sys_IpChId):
        print('Working on',fname_h5,corr_str)
        XEng_raw_data = roach_hsys.read_dbe_correlator_h5(path_data + \
        path.sep + fname_h5 + '.h5',corr_str+corr_str)[0] # corr_str + corr_str 
        # yields autocorrelation (integrated power). The k7w already corrects for XEng 
        # accumulation, whilst read_dbe_correlator_h5 corrects for the  128 accumulation 
        # that occurs on the FEng
        if do_vanvleck:
            XEng_raw_data = roach_hsys.apply_van_fleck_inverse(XEng_raw_data,x_inv_dB,y_inv_dB)
            
        XEng_test_dBL[idx_f,:,idx_s] = 10*np.log10(np.median(np.real(XEng_raw_data),0)) 

XEng_raw_data = 0

next_fig_num += 1
plot_to_subplots([SA_ChP_dBm[:,idx_feng_ch,idx_ip].T for idx_ip in idx_ip_ch], \
                 [(XEng_test_dBL - SA_ChP_dBm)[:,idx_feng_ch,idx_ip].T for idx_ip in idx_ip_ch], \
                 linestyle = '-',marker = '.', \
                 xlabel_txt = 'SA Input channel power [dBm]', \
                 ylabel_txt = list(Sys_IpChId[idx_ip_ch]), \
                 title_txt = ['Linearized XEng-SA response'], \
                 fig = next_fig_num,sp_cols = 2)

plt.show()
