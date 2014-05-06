#! /usr/bin/python
#
# Baseline calibration for multiple baselines using HDF5 format version 1 and 2 files.
#
#  modified sjcm for averaging

import optparse

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import katfile
import scape
import katpoint
import struct

import reduction_subfunctions as redsub

import scipy as sc

print "\nLoading and processing data...\n"

data = katfile.open('1314375234.h5', channel_range=(200, 800))  # shorter one.
#data = katfile.open('1315492546.h5', channel_range=(200, 800))  # really long
array_ant = katpoint.Antenna('ant0, -30:43:17.3, 21:24:38.5, 1038.0, 0.0')

opts = struct;
opts.pol = 'H';
opts.ants = ('ant1', 'ant2', 'ant3', 'ant4', 'ant5', 'ant6', 'ant7');
opts.time_lim = 8;

pols = ('H', 'V');

new_ants = {                                                                                                                        
  'ant1' : ('ant1, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 25.095 -9.095 0.045, , 1.22',   23220.506e-9, 23228.551e-9, 'ped1'),      
  'ant2' : ('ant2, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 90.288 26.389 -0.238, , 1.22',  23282.549e-9, 23285.573e-9, 'ped2'),      
  'ant3' : ('ant3, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 3.985 26.899 -0.012, , 1.22',   23406.720e-9, 23398.971e-9, 'ped3'),      
  'ant4' : ('ant4, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -21.600 25.500 0.000, , 1.22',  23514.801e-9, 23514.801e-9, 'ped4'),      
  'ant5' : ('ant5, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -38.264 -2.586 0.371, , 1.22',  23674.52e-9, 23667.0e-9, 'ped5'),         
  'ant6' : ('ant6, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -61.580 -79.685 0.690, , 1.22', 23782.614e-9, 23781.954e-9, 'ped6'),      
  'ant7' : ('ant7, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -87.979 75.756 0.125, , 1.22',  24046.038e-9, 24038.167e-9, 'ped7'),      
}                                                                                                                                   

both_diff_delay = [];
both_diff_delay_errs = [];
for opts.pol in pols:
    # Create antenna objects with latest positions for antennas used in experiment, and list of inputs and cable delays 
    ants = dict([(ant.name, katpoint.Antenna(new_ants[ant.name][0])) for ant in data.ants])
    inputs = []
    for ant in sorted(ants):
        if ant + opts.pol in data.inputs:
            inputs.append(ant + opts.pol)
    # Extract available cross-correlation products, as pairs of indices into input list
    crosscorr = [corrprod for corrprod in data.all_corr_products(inputs) if corrprod[0] != corrprod[1]]
    input_pairs = np.array(crosscorr).T;
    # fringe stop the data.  that way you're just solving for corrections on the current values.  
    (obs_vis, obs_timestamp, target_list, noise_diode) = redsub.stop_fringes_all(data, opts, True);
    # do some time averaging.
    (mean_vis, obs_var, mean_time)  = redsub.bin_scans_noweight(obs_vis, obs_timestamp);
    for n in range(len(mean_vis)):
        obs_var[n] = obs_var[n].reshape( (1, obs_var[n].shape[0], obs_var[n].shape[1]));
        mean_vis[n] = mean_vis[n].reshape( (1, mean_vis[n].shape[0], mean_vis[n].shape[1]));
        mean_vis[n] = mean_vis[n].data;
    # should be able to get a mean bandpass from this: 
    (gainsol, bp_to_apply) = redsub.scan_bandpass_weights(mean_vis, obs_var, True, data.ref_ant, opts, inputs, input_pairs);
    num_bases = obs_var[0].shape[2];
    # next we reconstruct the bandpass, and remove its shape from our data                                                                               
    bp_scale = zeros( (bp_to_apply.shape[0], num_bases), np.complex128);
    for n in xrange(bp_to_apply.shape[0]):
        g1, g2 = bp_to_apply[n, input_pairs[0]], bp_to_apply[n, input_pairs[1]];
        bp_scale[n,:] = g1*g2.conj();
    mean_amp = (np.abs(bp_scale).mean(axis=0));
    mean_phase = np.arctan2(bp_scale.imag, bp_scale.real).mean(axis=0);
    final_bp_scale = zeros( (bp_scale.shape) , np.complex128);
    # rescale the bandpass just to make sure the mean values don't change                                                                                
    for n, vis in enumerate(bp_scale):
        mean_gain = mean_amp*exp(1j*mean_phase);
        vis = vis/mean_gain;
        final_bp_scale[n,:] = vis;
    # apply the bandpass
    vis_withbp = [];
    for n in range( len(mean_vis)):
        with_bp = zeros( (mean_vis[n].shape) , np.complex128 );
        for m, vis in enumerate(mean_vis[n]):
            with_bp[m,:,:] = vis/final_bp_scale;
        vis_withbp.append(with_bp);
    # Filter available antennas via script --ants option, if provided
    ants = [ant for ant in data.ants if opts.ants is None or ant.name in opts.ants]
    ref_ant_ind = [ant.name for ant in ants].index(data.ref_ant)
    # Form desired signal paths and obtain all baselines connecting them
    signals = [ant.name + opts.pol for ant in ants]
    baselines = data.all_corr_products(signals)
    # Throw out autocorrelations
    baselines = [(antA, antB) for antA, antB in baselines if antA != antB]
    baseline_names = ['%s - %s' % (ants[antA].name, ants[antB].name) for antA, antB in baselines]
    num_bls = len(baselines)
    if num_bls == 0:
        raise RuntimeError('No baselines based on the requested antennas and polarisation found in data set')
    time_list = mean_time;
    ## so now we have the time-averaged visibilities that are bandpass calibrated (vis_withbp) and
    ## the times (time_list) associated with them.  from the times, I should be able to calculate the az/el, etc
    ## we also have the list of targets (target_list)
    # given that all we care about is a slope across the band, we can
    # remove the mean phase in each baseline in the hopes of reducing the
    # wrapping problems.
    vis_array  = np.vstack(mean_vis);
    vis_ang    = np.arctan2(vis_array.imag, vis_array.real);
    vis_ang    = np.unwrap(vis_ang, axis=1, discont=0.2);
    mean_phase = np.mean(vis_ang, axis=1);
    mean_phase = mean_phase.reshape( (mean_phase.shape[0], 1, mean_phase.shape[1]));
    mean_phase = np.kron( np.ones( (1, vis_array.shape[1], 1) ), mean_phase);
    mean_phase = exp(1j*mean_phase);
    vis_array  = np.vstack(mean_vis)/mean_phase;
    time_vec   = np.vstack(time_list);
    var_array  = np.vstack(obs_var);
    target_vec = np.vstack(target_list);
    time_lim   = opts.time_lim; #minutes
    delw = [];
    diff_delays = [];
    diff_delay_errs = [];
    # first we take the differences between sources within time_lim of each other
    for n, this_time in enumerate(time_vec):
        ind_time = (time_vec - this_time > 0) & (time_vec - this_time < time_lim*60)
        # find the ones where it's true
        for m in ind_time.nonzero()[0]:
            this_diff = vis_array[n,:,:]/vis_array[m,:,:];
            this_var  = np.sqrt(obs_var[n]**2 + obs_var[m]**2)
            # next we fit a line across the slope to get the w factors.
            (delay, delay_errs) = redsub.calc_delay(data.channel_freqs, np.arctan2(this_diff.imag, this_diff.real), this_var.data.squeeze());
            diff_delays.append(delay);
            diff_delay_errs.append(delay_errs);
    diff_delays     = np.vstack(diff_delays);
    diff_delay_errs = np.vstack(diff_delay_errs);
    both_diff_delay.append(diff_delays);
    both_diff_delay_errs.append(diff_delay_errs);


# define the functions we'll fit.

def calc_del_w(params, ants, input_pairs, time_vec, target_list, time_lim, ant_ref):
    # first we calculate the w values for all entries on all baselines (geometric delay)
    wvals = [];
    for n, target in enumerate(target_list):
        this_time = time_vec[n];
        this_w    = np.zeros( (1, len(input_pairs[0]) ) );
        ra, dec = target.apparent_radec(time_vec[n], array_ant);
        for m in range(len(input_pairs[0])):
            antA = input_pairs[0][m];
            antB = input_pairs[1][m];
            offA = params[ (antA)*3:antA*3+3];
            offB = params[ (antB)*3:antB*3+3];
            # add on the offsets for the baseline parameters
            ant0 = ants[antA];
            ant1 = ants[antB];
            haA  = ant0.local_sidereal_time(time_vec[n]) - ra;
            haB  = ant1.local_sidereal_time(time_vec[n]) - ra;
            xyztmsa = katpoint.enu_to_xyz(offA[0], offA[1], offA[2], ant_ref.observer.lat);
            xyztmsb = katpoint.enu_to_xyz(offB[0], offB[1], offB[2], ant_ref.observer.lat);
            w_a  = cos(dec)*cos(haA)*xyztmsa[0] - cos(dec)*sin(haA)*xyztmsa[1] + sin(dec)*xyztmsa[2];
            w_b  = cos(dec)*cos(haB)*xyztmsb[0] - cos(dec)*sin(haB)*xyztmsb[1] + sin(dec)*xyztmsb[2];
            this_w[0,m] = (w_a - w_b)/katpoint.lightspeed*1e9;  
        wvals.append(this_w);
    diff_delay_mod = [];
    wvals = np.vstack(wvals);
    # first we take the differences between sources within time_lim of each other
    for n, this_time in enumerate(time_vec):
        ind_time = (time_vec - this_time > 0) & (time_vec - this_time < time_lim*60)
        # find the ones where it's true
        for m in ind_time.nonzero()[0]:
            this_diff = wvals[n,:] - wvals[m,:];
            diff_delay_mod.append(this_diff);
    diff_delay_mod = np.vstack(diff_delay_mod);
    return diff_delay_mod;


# we need to define the chi-squared that is the difference between the model and the real.
def chi_sq_w(params, ants, input_pairs, time_vec, target_list, time_lim, ant_ref, diff_delay, diff_delay_errs):

    full_params[params_to_fit] = params;
    diff_delay_mod = calc_del_w(full_params, ants, input_pairs, time_vec, target_list, time_lim, ant_ref);
    dd = np.abs(diff_delay_mod - diff_delay)/diff_delay_errs;
    return dd.ravel();

# we need to define the chi-squared that is the difference between the model and the real.
def chi_sq_w_2pol(params, ants, input_pairs, time_vec, target_list, time_lim, ant_ref, diff_delay, diff_delay_errs):

    full_params[params_to_fit] = params;
    diff_delay_mod = calc_del_w(full_params, ants, input_pairs, time_vec, target_list, time_lim, ant_ref);
    pol1 = np.abs(diff_delay_mod - both_diff_delay[0])/both_diff_delay_errs[0];
    pol2 = np.abs(diff_delay_mod - both_diff_delay[1])/both_diff_delay_errs[1];
    cc   = [pol1, pol2];
    cc   = np.vstack(cc);
    return cc.ravel();


 
# so now we have the delay differences
# next, we need to optimize
ref_ant_num = 0;
ant_ref = ants[ref_ant_num];

both_diff_delay_errs[0][:] = 1;
both_diff_delay_errs[1][:] = 1;

# i'll try only solving for ant2 and ant7

full_params   = np.zeros(len(ants)*3);
params_to_fit = range(len(ants)*3);
#params_to_fit = array( ([3, 4, 5, 18, 19, 20]));
 
params_to_fit.pop(3*ref_ant_num);
params_to_fit.pop(3*ref_ant_num);
params_to_fit.pop(3*ref_ant_num);
params        = np.zeros(len(ants*3));
initial_locs  = np.zeros(len(ants*3));

data_bunch = (ants, input_pairs, time_vec, target_list, time_lim, ant_ref, both_diff_delay, both_diff_delay_errs);
#
aa = sc.optimize.leastsq(chi_sq_w_2pol, initial_locs, data_bunch);
#
#
#
#
#
#
