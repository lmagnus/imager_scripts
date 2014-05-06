# -*- coding: utf-8 -*-
"""
Module that estimates estimates the roach FX correlator
Created on Fr Feb 25 11:42:00 2011

@author: paulh
"""
# ------------------------
# IMPORT REQUIRED PACKAGES
# ------------------------
import h5py
import numpy as np
from scipy import log10, signal, load
import sys
import os

# -----------------------------------
# DEFINE FUNCTIONS
# -----------------------------------
def set_feng(kat,fft_shift_schedule = 1023,poco_gain = 4000,debug_level = 1):
    ''' Sets the F engine of the FX correlator with the required critical parameters
    
    This function is used to configure the F engine critical parameters:
    
    a) fft shift schedule - For a 2**N length FFT, the current implementation comprises
    of N stages. At each stage the data may be right shifted (divided by 2) to control 
    bitgrowth. The shift schedule define the shifting at each of the stages. This may be
    represented by N bits. In the example of a 2**10 = 1024 fft length, the shift schedule 
    is represented by 10 bits. The MSB sets the shifting for the first stage, going down
    the stages until the LSB sets the last stage. The most conservative scheme is that where
    all stages are shifted and may be reprensted by the integer 2**N -1 (1023 in the example)
    More optimal shifting schemes may be exploited for signals with known statistics, 
    e.g. white noise.
    
    b) poco_gain - used to choose which bits will be sliced as input to 4-bit correlator. 
    The current default value of 4000 have been used for the fringe finder for some time now.
    The output of the fft is multiplied with the poco_gain before being casted as a 4-bit signed
    fractional, with 3 fractional bits.
    
    Parameters
    ----------
    kat : :class:'katuilib.utility.KATHost' object
        kat object used to address the roach
    fft_shift_schedule : integer
    poco_gain : float
    debug_level : integer
    
    Returns
    -------
    Nothing
    
    Notes
    -----
    None
    
    '''
    rx1 = kat.dbe.req.dbe_fft_shift(fft_shift_schedule)
    rx2 = kat.dbe.req.dbe_poco_gain(poco_gain)
    if debug_level == 1:
        print(rx1)
        print(rx2)
    if not (rx1.succeeded and rx2.succeeded):
        raise RuntimeError('Could not configure quantizer')

def set_xeng(kat,t_corr_ms =  512,f0_lo_Hz = 1.5e9,baseline_mask = (1,3), \
    do_capture = 0,write_hdf5 = 0,debug_level = 1):
    '''
    Function that sets the X engine critical parameters:
    a) Correlation time [ms]
    b) LO frequency [Hz]
    Also sets whether hdf5 files will be written and whether to capture data (mutually exclusive)
    '''
    rx1 = kat.dbe.req.capture_stop()
    rx2 = kat.dbe.req.k7w_write_hdf5(0)
    rx3 = kat.dbe.req.capture_setup(t_corr_ms,f0_lo_Hz)
    rx4 = eval('kat.dbe.req.k7w_baseline_mask%s'%str(baseline_mask))
	
    if debug_level == 1:
        print(rx1)
        print(rx2)
        print(rx3)
        print(rx4)
        
    if not (rx1.succeeded and rx2.succeeded and rx3.succeeded and rx4.succeeded):
        raise RuntimeError('Could not configure dbe')
	
    if do_capture:
        rx5 = kat.dbe.req.capture_start()
    elif write_hdf5:
        rx5 = kat.dbe.req.k7w_write_hdf5(1)
		
    if do_capture or write_hdf5:
        if debug_level == 1:
            print(rx5)
        if not rx5.succeeded:
            raise RuntimeError('Could not configure dbe')

def get_snapshot(kat,data_source = 'adc',nr_samples=8192,ch_list=['0x']):
    '''
    Function that acquires snapshot data of a defined lenght from listed channels
    @kat: katuilib.utility.KATHost object
    @nr_samples: Number of samples to return per channel
    @ch_id: List of all the channels to capture
    
    Return:
    -------
    data - numpy.ndarray containing all the data samples
    snapshot_length - intenger indicating the basic snapshot length
    '''
    if type(ch_list) == str:
        ch_list = list([ch_list])
    if data_source == 'quanti':
        data = np.zeros((len(ch_list),nr_samples),dtype = 'complex64')
    else:
        data = np.zeros((len(ch_list),nr_samples))
    first_snapshot = True
    for idx_ch,ch_id in enumerate(ch_list):
        snapshot_data = list()
        while True:
            snapshot_data.extend(kat.dh.get_snapshot(data_source,ch_id))
            if first_snapshot:
                snapshot_length = len(snapshot_data)
                first_snapshot = False
            if (len(snapshot_data) >= nr_samples):
                break
            else:
                print('Collecting more %s samples for %s: %s of %s'%(data_source,ch_id,len(snapshot_data),nr_samples))
                sys.stdout.flush()
        data[idx_ch,:] = np.asarray(snapshot_data[:nr_samples])
    return data.squeeze(),snapshot_length
    
def calc_pfb_out(x_in,pfb_N = 1024, pfb_P = 4, pfb_ave = True, pfb_win = 'hamming'):
    ''' Function that calculates the polyphase filter bank (PFB) response from the input data. It is critical that 
    The (fft length * prefilter length) be an integer fraction of the snapshot length, if not, the concatenation of 
    the data will lead to degraded filter response estimation
    INPUT PARAMETERS:
    @ x_in: Input numpy.ndarray array. If more than 1-D, assumes individual datasets are stored in the last dimension
    @ pfb_N: fft length
    @ pfb_P: Pre-filter length
    @ pfb_ave: Whether to take the average of the pfb output sequence or not
    @ pfb_win: Window applied to the pdf
    RETURN:
    xout:   Output of the pfb. If pfb_ave == True, this will be the same dimension as input, 
            alternatively this will have 1 more dimension than the output. The fft have been normalized by N. 
            Both possitive and negative frequencies are still reprenseted to cater for complex data.
    '''    
    pfb_h0 = signal.firwin(pfb_P*pfb_N,1. / pfb_N,window = pfb_win)
    #pfb_p = pfb_N*np.flipud(pfb_h0.reshape((pfb_P,pfb_N)))
    pfb_p = pfb_N*(pfb_h0.reshape((pfb_P,pfb_N)))
    
    if np.rank(x_in) == 1:
        x_in_preshape = tuple([1]);
    else:
        x_in_preshape = x_in.shape[:-1]
    pre_filt_shape = tuple(np.hstack((x_in_preshape,[-1,pfb_P,pfb_N])))
    axis_proc = len(pre_filt_shape)-2
    pfb_out = np.fft.fft((x_in.reshape(pre_filt_shape) * pfb_p).sum(axis = axis_proc),axis = axis_proc) / np.float(pfb_N)
    if pfb_ave:
        return np.mean(abs(pfb_out),axis = (axis_proc-1)).squeeze()
    else:
        return abs(pfb_out).squeeze()
            
def get_van_vleck_inverse(fname_est = 'CorrGainEst.npz'):
    ''' Extracts the Van Vleck inverse function from estimates stored to file
    
    Function that gets the inverse gain curve that will aim to remove the non-linearity 
    associated with a 4-bit correlator. The input file contains two arrays, x_dB and y_dB 
    that has been estimated from a series of measurements.
    
    Parameters
    ----------
    fname_est : string
        The file 'fname_est' is a '*.npz' file that contains the estimated non-linear
        amplitude transfer function, represented by two numpy.ndarray 2-D arrays, 
        x_dB and y_dB
        
    Returns
    -------
    x_inv_dB : 1-D numpy.ndarray
        Finely sampled array of expected non-linear correlator output values, 
        normalized by the accumulation length of the correlator.
    y_inv_dB : 1-D numpy.ndarray
        Associated array representing the linearized corresponding correlator 
        output values
        
    Notes
    -----
    Typical usage would be:
    X_inv_dB,y_inv_dB = roach_hsys.get_van_vleck_inverse()
    XEng_raw_data,XEng_accum_per_int = roach_hsys.read_dbe_correlator_h5(fname,'0x0x')
    XEng_raw_data = roach_hsys.apply_van_fleck_inverse(XEng_raw_data,x_inv_dB,y_inv_dB)
        
    See also
    --------
    roach_hsys.apply_van_fleck_inverse
    
    '''
    est = load(fname_est)
    x_dB = est['x_dB']
    y_dB = est['y_dB']
    idx_0 = np.argmin(np.abs(np.diff(y_dB)/np.diff(x_dB) - 1)) # Find the index where the data is closest to being linear
    x_inv_dB = y_dB
    y_inv_dB = x_dB - x_dB[idx_0] + y_dB[idx_0] # Normalize to ensure that the gain is unity where the data is closest to being linear
    return x_inv_dB,y_inv_dB

def read_dbe_correlator_h5(fname,corr_str = '0x0x'):
    f_h5 = h5py.File(fname)
    # Correlator mapping of DBE input string to correlation product index (Miriad-style numbering)
    input_map = f_h5['Correlator']['input_map'].value
    dbestr_to_corr_id = dict(zip(input_map['dbe_inputs'], input_map['correlator_product_id']))
    data = f_h5['Scans']['CompoundScan0']['Scan0']['data'][str(dbestr_to_corr_id[corr_str])]
    accum_per_int = int(f_h5['Correlator'].attrs['accum_per_int'])
    f_h5.close()
    return data,accum_per_int
    
def apply_van_fleck_inverse(data,x_inv_dB,y_inv_dB):
    ''' Apply the Van Vleck inverse correction to the input data, based on the estimated non-linearity
    
    Function that applies the Van Vleck linearization based on an imperical inverse
    transfer function, x_inv_dB,y_inv_dB
    
    This algorithm simply flattens the input data, performs a 1-D linear interpolation,
    and finaly reshapes the data into the same shape as the input
    
    Parameters
    ----------
    data : numpy.ndarray
        Non-linear correlator data of any shape. The first step is to take the absolute 
        value of the input data and process on the absolute value, since it assumed that 
        the input is purely real
    x_inv_dB : 1-D numpy.ndarray
        Finely sampled array of expected non-linear correlator output values, 
        normalized by the accumulation length of the correlator.
    y_inv_dB : 1-D numpy.ndarray
        Associated array representing the linearized corresponding correlator 
        output values
    
    Returns
    -------
    data_lin : numpy.ndarray
        Linearized correlator data with the same shape as the input
        
    Notes
    -----
    Typical usage would be:
    X_inv_dB,y_inv_dB = roach_hsys.get_van_vleck_inverse()
    XEng_raw_data,XEng_accum_per_int = roach_hsys.read_dbe_correlator_h5(fname,'0x0x')
    XEng_raw_data = roach_hsys.apply_van_fleck_inverse(XEng_raw_data,x_inv_dB,y_inv_dB)
        
    See also
    --------
    roach_hsys.get_van_fleck_inverse
        
    '''
    data_shape = data.shape
    data_lin = 10**(0.1*np.interp(10*log10(abs(data)+1e-20).ravel(),x_inv_dB,y_inv_dB).reshape(data_shape))
    return data_lin
    

def make_van_vleck_inverse_function():
    """
        Returns a function with the following parameters & return values:
        @param h5fn: filename, which ends with the Unix epoch timestamp+".h5".
        @param x: the uncorrected detector output data.
        @param chan: "x" or "y", referring to DBE channel suffix (default None).
        @return: the corrected detector output data.
    """
    basedir = os.path.dirname(__file__)
    _xy = {None: get_van_vleck_inverse(basedir+"/CorrGainEst_SA_Tot.npz"),
           "0x": get_van_vleck_inverse(basedir+"/CorrGainEst_SA_Ch0x.npz"),
           "0y": get_van_vleck_inverse(basedir+"/CorrGainEst_SA_Ch0y.npz"),
           "1x": get_van_vleck_inverse(basedir+"/CorrGainEst_SA_Ch1x.npz"),
           "1y": get_van_vleck_inverse(basedir+"/CorrGainEst_SA_Ch1y.npz")}
    _xy_512 = {None: get_van_vleck_inverse(basedir+"/CorrGainEst_SA_Tot_x512.npz"),
               "0x": get_van_vleck_inverse(basedir+"/CorrGainEst_SA_Ch0x_x512.npz"),
               "0y": get_van_vleck_inverse(basedir+"/CorrGainEst_SA_Ch0y_x512.npz"),
               "1x": get_van_vleck_inverse(basedir+"/CorrGainEst_SA_Ch1x_x512.npz"),
               "1y": get_van_vleck_inverse(basedir+"/CorrGainEst_SA_Ch1y_x512.npz")}
    def inv_van_vleck(h5fn,x,chan=None):
        version = 1
	if (version < 2 and h5fn[-13:-3] < "1300881600"): # '*512' correction was made on 23 March 2011 at 14h00 SASTS, not documented otherwise
            return apply_van_fleck_inverse(x, _xy_512[chan][0], _xy_512[chan][1])
        elif (version < 2):
            return apply_van_fleck_inverse(x, _xy[chan][0], _xy[chan][1])
        else: # DBE 7 not yet characterised
            return x
    return inv_van_vleck
