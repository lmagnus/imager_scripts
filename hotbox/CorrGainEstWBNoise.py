# -*- coding: utf-8 -*-
"""
Module that estimates the correlator gain for wideband noise on the fringe finder correlator
Created on Wed Feb 09 11:41:11 2011

@author: paulh
"""
# ------------------------
# IMPORT REQUIRED PACKAGES
# ------------------------
from SpectrumAnalyzerSocket import sa_sock
from roach_hsys import get_snapshot, calc_pfb_out, set_feng, set_xeng
import numpy as np
from scipy import interpolate, log10
import sys
import time
from os import path, makedirs, getcwd

# ----------------
# BASIC PARAMETERS
# ----------------

# Define measurement setup
Sys_IpChId = ('0x','0y','1x','1y')
Sys_NrIpCh = len(Sys_IpChId)         # Number of system inputs
Out_Sys_dtype = np.dtype([('IpChId','a16',Sys_NrIpCh)])

# Define channelizer setup
FEng_NrCh = 512          # F engine number of channels
FEng_ChBW = 781.25e3     # F engine channel bandwidth
FEng_FStart_Hz = 0       # Spectrum analyzer start frequency
FEng_POCOGain = 4000     # Gain before quantization
FEng_FFTShiftSchedule = 1023 # Shift schedule for the fft stages MSB = first stage, 10 stages 

Out_FEng_dtype = np.dtype([('NrCh','f4',1),('ChBW_Hz','f4',1), \
                 ('FStart_Hz','f4',1),('f_Hz','f4',FEng_NrCh), \
                 ('POCOGain','f4',1),('FFTShiftSchedule','f4',1)])
                 # Definition of datatype for structured array output

# Define correlator settings
XEng_TCorr_ms = 512          # Correlation time in ms
XEng_F0_LO_Hz = 1.5e9        # LO frequency in MHz
XEng_TCapture_s = 10         # Define total capture time in seconds
Out_XEng_dtype = np.dtype([('TCorr_ms','f4',1),('F0_LO_Hz','f4',1), \
                 ('TCapture_s','f4',1)])

# Define spectrum analyzer settings
SA_RBW_Hz = 20e3       # Spectrum analyzer resolution bandwidth in Hz
SA_NrSweeps = 10       # Spectrum analyzer number of sweeps to average
SA_IP = '192.168.10.145' # Spectrum analyzer IP address (currently auto assigned)
SA_Port = 5025          # Spectrum analyzer port (arbitrary, taken from Agilent)
SA_Timeout_s = 1        # Spectrum analyzer long timeout
Out_SA_dtype = np.dtype([('RBW_Hz','f4',1),('NrSweeps','f4',1)])
               
# Define adc snapshot settings
ADC_MinNrSamples = 2**20 # Minimum number of samples for sufficient ADC statistics, i.e. mean and variance
ADC_pfb_P = 4
ADC_pfb_N = 1024
ADC_pfb_win = 'hamming'
Out_ADC_dtype = np.dtype([('MinNrSamples','f4',1),('pfb_P','f4',1), \
                ('pfb_N','f4',1),('pfb_win','a16',1)])

# Define requantizer snatshot settings
ReQ_MinNrSamples = 100*FEng_NrCh # Minimum number of samples for sufficient requantizer statistics PER CHANNEL, real and imaginary.
Out_ReQ_dtype = np.dtype([('MinNrSamples','f4',1)])

# Define source settings
Src_type = 'noise'
Out_Src_dtype = np.dtype([('Type','a16',1)])

# Define the output data type
Out_data_dtype = np.dtype([('Src_NoiseAtt_dB','f4'), \
                 ('Src_CWP_dBm','f4',1),('Src_CW_f0_Hz','f4',1), \
                 ('PM_PIn_dBm','f4',1),('SA_ChP_dBm','f4',FEng_NrCh), \
                 ('ADC_mean','f4',Sys_NrIpCh),('ADC_std','f4',Sys_NrIpCh), \
                 ('ADC_pfb','f4',(FEng_NrCh,Sys_NrIpCh)), \
                 ('ReQ_I_mean','f4',(FEng_NrCh,Sys_NrIpCh)),('ReQ_I_std','f4',(FEng_NrCh,Sys_NrIpCh)), \
                 ('ReQ_Q_mean','f4',(FEng_NrCh,Sys_NrIpCh)),('ReQ_Q_std','f4',(FEng_NrCh,Sys_NrIpCh)), \
                 ('XEng_fname_out','a32',1)])

# Set up processing required
do_sa_captures = False
do_snapshot = False
do_correlator = False
do_powermeter = False
Out_Proc_dtype = np.dtype([('do_sa_captures','b',1),('do_snapshot','b',1), \
                 ('do_correlator','b',1),('do_powermeter','b',1)])

# ----------
# PROCESSING
# ----------

FEng_f_Hz = np.arange(FEng_NrCh)*FEng_ChBW + (FEng_ChBW/2)

# STEP 1: Basic configuration of the spectrum analyzer
if do_sa_captures:
    sa = sa_sock()
    sa.sa_connect((SA_IP,SA_Port))
    start_f_set,delta_f_set,stop_f_set,nr_points_set,nr_sweeps_set = \
    sa.sa_sweep_startinc(FEng_ChBW/2,FEng_ChBW,FEng_NrCh+100,SA_NrSweeps)
    SA_f_Hz = np.arange(nr_points_set)*delta_f_set + start_f_set
    rbw_auto_set,rbw_set,vbw_auto_set,vbw_set,det_mode_set,trace_mode_set = \
    sa.sa_detect('off',SA_RBW_Hz,'on',0,'RMS','AVER')
    SA_RBW_set_Hz = rbw_set
    ref_level_dBm_set = sa.sa_amplitude(0)
    
# STEP 2: Configure the quantizer
if do_snapshot or do_correlator:
    set_feng(kat,FEng_FFTShiftSchedule,FEng_POCOGain)

# STEP 3: Configure correlator and k7w
if do_correlator:
    set_xeng(kat,XEng_TCorr_ms,XEng_F0_LO_Hz,(1,3),0,1)

# STEP 4: Iterate through the different attenuator settings
data = np.zeros(0,dtype = Out_data_dtype)
while True:
    new_data = np.zeros(1,dtype = Out_data_dtype)
    
    # Get the new source settings
    new_data[0]['Src_NoiseAtt_dB'] = np.double(raw_input('Please enter noise source attenuation [dB]: '))
    if Src_type.upper() == 'CW':
        new_data[0]['Src_CWP_dBm'] = np.double(raw_input('Please enter cw source input power [dBm]: '))
        new_data[0]['Src_CW_f0_Hz'] = np.double(raw_input('Please enter cw source centre frequency [Hz]: '))
    else:
        new_data[0]['Src_CWP_dBm'] = 0
        new_data[0]['Src_CW_f0_Hz'] = 0
        
    # Get the power meter settings
    if do_powermeter:
        new_data[0]['PM_PIn_dBm'] = np.double(raw_input('Please enter measured total input power [dBm]: '))
    else:
        new_data[0]['PM_PIn_dBm'] = 0
        
    # Get the Spectrum analyzer data
    if do_sa_captures:
        SA_Yraw_dBm = sa.sa_getsweepdata()
        tck = interpolate.splrep(SA_f_Hz,SA_Yraw_dBm,s=0)
        SA_Y_dBm = interpolate.splev(FEng_f_Hz,tck)
        SA_Y_dBm = SA_Y_dBm + 10*log10(FEng_ChBW/SA_RBW_set_Hz) + 2.51 
        # Scaling to get the channel power based on the instrument RBW. Also corrects
        # for the underestimation of Gaussian noise variance when averaging on log-scale    
        new_data[0]['SA_ChP_dBm'][:] = SA_Y_dBm

    # Get the ADC and ReQ snapshot data
    if do_snapshot: 
        # ADC samples
        x_adc = get_snapshot(kat,'adc',ADC_MinNrSamples,Sys_IpChId)
        if np.rank(x_adc[0])==1:
            axis_proc = 0
        else:
            axis_proc = 1
        new_data[0]['ADC_mean'][:] = np.mean(x_adc[0],axis_proc)
        new_data[0]['ADC_std'][:] = np.std(x_adc[0],axis_proc)
        new_data[0]['ADC_pfb'][:,:] = calc_pfb_out(x_adc[0],ADC_pfb_N, \
        ADC_pfb_P,True,ADC_pfb_win)[:,0:(ADC_pfb_N/2)].T
        # ReQ Q samples
        x_q = get_snapshot(kat,'quanti',ReQ_MinNrSamples, \
        Sys_IpChId)[0].reshape((Sys_NrIpCh,-1,FEng_NrCh))
        new_data[0]['ReQ_I_mean'][:,:] = np.mean(np.real(x_q),1).T
        new_data[0]['ReQ_I_std'][:,:] = np.std(np.real(x_q),1).T
        new_data[0]['ReQ_Q_mean'][:,:] = np.mean(np.imag(x_q),1).T
        new_data[0]['ReQ_Q_std'][:,:] = np.std(np.imag(x_q),1).T
    
    # Get the XEng correlator data filename
    if do_correlator:
        rx4 = kat.dbe.req.capture_start()
        print(rx4)
        sys.stdout.flush()
        if not rx4.succeeded:
            raise RuntimeError('Could not start capture')
        
        time.sleep(XEng_TCapture_s)
        rx5 = kat.dbe.req.capture_stop()
        print(rx5)
        sys.stdout.flush()
        if not rx5.succeeded:
            raise RuntimeError('Could not stop capture')
            
        rx6 =  kat.dbe.req.k7w_get_current_file()
        print(rx6)
        sys.stdout.flush()
        if not rx6.succeeded:
            raise RuntimeError('Could not return filename')
        
        new_data[0]['XEng_fname_out'] = path.split(str(rx6.messages[1].arguments)[2:-2])[1].split('.')[0]

    # Concatenate the new data to the old data
    data = np.concatenate((data,new_data))
    # Test if more data is to be recorded
    more_data = raw_input('Do you want to capture more attenuation settings data (y/n): ')
    if more_data.upper().startswith('Y'):
        print('More data to be captured')
    elif more_data.upper().startswith('N'):
        print('Done with captures')
        break
    else:
        print('Assume you want to continue with more data captures')
        
# Prepare for saving of the data
Out_dtype = np.dtype([('Sys',Out_Sys_dtype,1),('FEng',Out_FEng_dtype,1), \
            ('XEng',Out_XEng_dtype,1),('SA',Out_SA_dtype,1), \
            ('ADC',Out_ADC_dtype,1),('ReQ',Out_ReQ_dtype,1), \
            ('Src',Out_Src_dtype,1),('Proc',Out_Proc_dtype,1), \
            ('data',Out_data_dtype,len(data))])
            
dataset = np.zeros(1,dtype = Out_dtype)
dataset[0]['Sys']['IpChId'][:] = np.asarray(Sys_IpChId)
dataset[0]['FEng']['NrCh'] = FEng_NrCh
dataset[0]['FEng']['ChBW_Hz'] = FEng_ChBW
dataset[0]['FEng']['FStart_Hz'] = FEng_FStart_Hz
dataset[0]['FEng']['f_Hz'][:] = FEng_f_Hz
dataset[0]['FEng']['FFTShiftSchedule'] = FEng_FFTShiftSchedule
dataset[0]['FEng']['POCOGain'] = FEng_POCOGain
dataset[0]['XEng']['TCorr_ms'] = XEng_TCorr_ms
dataset[0]['XEng']['F0_LO_Hz'] = XEng_F0_LO_Hz
dataset[0]['XEng']['TCapture_s'] = XEng_TCapture_s
dataset[0]['SA']['RBW_Hz'] = SA_RBW_Hz
dataset[0]['SA']['NrSweeps'] = SA_NrSweeps
dataset[0]['ADC']['MinNrSamples'] = ADC_MinNrSamples               
dataset[0]['ReQ']['MinNrSamples'] = ReQ_MinNrSamples               
dataset[0]['Src']['Type'] = Src_type
dataset[0]['Proc']['do_sa_captures'] = do_sa_captures
dataset[0]['Proc']['do_snapshot'] = do_snapshot
dataset[0]['Proc']['do_correlator'] = do_correlator
dataset[0]['Proc']['do_powermeter'] = do_powermeter
if len(data) > 1:
    dataset[0]['data'][:] = data
else:
    dataset[0]['data'] = data

# Save data to directory
save_path = path.join(getcwd(),'results','data')
if not path.exists(save_path):
    makedirs(save_path)

save_fname = save_path + path.sep + 'EstCorrGain_' + Src_type.upper() + time.strftime('_%Y%m%d_%H%M')

np.save(save_fname,dataset)
        
# Finishing off
if do_sa_captures:
    sa.close()
    
if do_correlator:
    rx2 = kat.dbe.req.k7w_write_hdf5(0)
    print(rx2)
    if not rx2.succeeded:
        raise RuntimeError('Could not configure dbe')







