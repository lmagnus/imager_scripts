# import the libraries                                                                                                                                 
import katfile
import katpoint
import numpy as np
from gaincal import NoiseDiodeModel
from hdf5 import remove_duplicates
import pickle
import struct
import image_red_functions as imred
import scape
import reduction_subfunctions as redsub



# Stephen's "script" on checking whether the noise in a band is comparable to the RMS in band-averaging.
# This is check will be done on a track of 3c273
# CAVEAT:  What you are about to see is really bad python programming.
#
# Code Outline
#-1.  Read in the data, fringe-stop the data
# 0.  Calculate the nominal system temperature of the data.  
# 1.  Calculate the average bandpass (like in scanbp.m)
# 2.  Apply a bandpass calibration
# 3.  Band-average the data, calculating the variance over the band
# 4.  Compare the band-averaged variance to that predicted for tsys on a per-second timescale
# 5.  In band-averaged data, bin in time.
# 6.  Check time averaging reduces noise according to theory.

# from inspecting the data, it seems the part of the band which is good is the one from about 1.69GHz to 1.95GHz.  which is index 200 to 850

#------------------------------------------------------------
# step -1:  read in/fringe-stop the data
#------------------------------------------------------------
data = katfile.open('1313842550_3c273_big.h5', channel_range=(200, 849));


opts = struct;
opts.pol = 'H';
opts.image_target = '3C 273';
opts.bandpass_cal = '3C 273';
opts.gain_cal     = '3C 273';
opts.time_slice   = 30;
num_ants          = 7;


new_ants = {
  'ant1' : ('ant1, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 25.095 -9.095 0.045, , 1.22', 23220.506e-9, 23228.551e-9),
  'ant2' : ('ant2, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 90.288 26.389 -0.238, , 1.22', 23282.549e-9, 23285.573e-9),
  'ant3' : ('ant3, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 3.985 26.899 -0.012, , 1.22', 23406.720e-9, 23398.971e-9),
  'ant4' : ('ant4, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -21.600 25.500 0.000, , 1.22', 23514.801e-9, 23514.801e-9),
  'ant5' : ('ant5, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -38.264 -2.586 0.371, , 1.22', 23676.033e-9, 23668.223e-9),
  'ant6' : ('ant6, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -61.580 -79.685 0.690, , 1.22', 23782.854e-9, 23782.150e-9),
  'ant7' : ('ant7, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -87.979 75.756 0.125, , 1.22', 24048.922e-9, 24040.487e-9),
}


# Create antenna objects with latest positions for antennas used in experiment, and list of inputs and cable delays                                    
ants = dict([(ant.name, katpoint.Antenna(new_ants[ant.name][0])) for ant in data.ants])
inputs, delays = [], {}
for ant in sorted(ants):
    if ant + opts.pol in data.inputs:
        inputs.append(ant + opts.pol)
        delays[ant + opts.pol] = new_ants[ant][1 if opts.pol == 'H' else 2]


# Extract available cross-correlation products, as pairs of indices into input list                                                                    
crosscorr = [corrprod for corrprod in data.all_corr_products(inputs) if corrprod[0] != corrprod[1]]
input_pairs = np.array(crosscorr).T;


# Extract frequency information                                                                                                                        
center_freqs = data.channel_freqs
wavelengths = katpoint.lightspeed / center_freqs

# Create catalogue of targets found in data                                                                                                            
targets = katpoint.Catalogue()
for scan_ind, cs_ind, state, target in data.scans():
    if state == 'track' and target.name not in targets:
        targets.add(target)



image_target = targets[opts.image_target]
bandpass_cal = targets[opts.bandpass_cal]
gain_cal = targets[opts.gain_cal]

############################## STOP FRINGES ####################################                                                                       

print "Assembling bandpass calibrator data and checking fringe stopping..."

(bp_cal_vis_samples, bp_cal_timestamps, noise_diode_times) = redsub.stop_fringes(data, bandpass_cal, opts, True);


################### CALCULATE THE SYSTEM TEMPERATURE ######################

print "Calculating system temperatures"

tsys = redsub.calc_tsys(data, noise_diode_times, opts);

# --------------------------------------------------------------------
# --- Plot the system temperature
# --------------------------------------------------------------------
antVals =  ['ant1', 'ant2', 'ant3', 'ant4', 'ant5', 'ant6', 'ant7'];
figure(1);
figure(num=1, figsize=(12,10))
for plotnum, (this_tsys, this_time) in enumerate(tsys):
    subplot(3,3,plotnum+1)
    plot(center_freqs/1000, transpose(this_tsys[1:len(this_tsys)-1]));
    if plotnum%3 == 0:
        ylabel('tsys [K]');
    if plotnum in range(4,7):
        xlabel('Frequency [GHz]');
    title(antVals[plotnum]);


############################## BANDPASS ####################################                                                                       

# there should be no noise source observations during our track.

# --------------------------------------------------------------------
# --- Calculate the average bandpass on the calibrators
# --------------------------------------------------------------------

print "Calculating Bandpasses" 

(gainsol, bp_to_apply) = redsub.scan_bandpass(bp_cal_vis_samples, True, data.ref_ant, opts, inputs, input_pairs);

print "done with bandpass"

# --------------------------------------------------------------------
# --- Plot the average bandpass
# --------------------------------------------------------------------
gain_abs = np.abs(bp_to_apply);
gain_abs[gain_abs==0] = np.nan;
gain_phase = np.arctan2(bp_to_apply.imag, bp_to_apply.real);
gain_phase[np.isnan(gain_abs)] = np.nan;
miny = np.nanmin(gain_abs);
maxy = np.nanmax(gain_abs);
axvals = [0.99*center_freqs.min()/1000, 1.01*center_freqs.max()/1000, 0.99*miny, 1.01*maxy];

fig = figure(num=2, figsize=(8,6))
plot(center_freqs/1000, gain_abs);
xlabel('Frequency [GHz]');
ylabel('Normalized Bandpass Shape');
legend( ('ant1', 'ant2', 'ant3', 'ant4', 'ant5', 'ant6', 'ant7') );
title('KAT7 Antenna Bandpass Amplitude');

fig = figure(num=3, figsize=(8,6))
plot(center_freqs/1000, gain_phase);
xlabel('Frequency [GHz]');
ylabel('Normalized Bandpass Shape');
legend( ('ant1', 'ant2', 'ant3', 'ant4', 'ant5', 'ant6', 'ant7') );
title('KAT7 Antenna Bandpass Phase');

# next we reconstruct the bandpass, and remove its shape from our data
bp_scale = zeros( (bp_to_apply.shape[0], 21), np.complex128);
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

# --------------------------------------------------------------------
# --- Apply the average bandpass
# --------------------------------------------------------------------

vis_bp = [];
vis_bp_avg = [];
vis_bp_var = [];
# so we apply this to bp_cal_vis_samples
for n in range( len(bp_cal_vis_samples) ):
    with_bp = zeros( (bp_cal_vis_samples[n].shape) , np.complex128 );
    with_bp_avg = zeros( (bp_cal_vis_samples[n].shape[0], bp_cal_vis_samples[n].shape[2]) , np.complex128 );
    for m, vis in enumerate(bp_cal_vis_samples[n]):
        with_bp[m,:,:] = vis/final_bp_scale;
    with_bp_avg = imred.nanmean(with_bp, axis=1);
    with_bp_var = imred.nanvarc(with_bp, axis=1);
    vis_bp.append(with_bp);
    vis_bp_avg.append(with_bp_avg);
    vis_bp_var.append(with_bp_var);
	
############################## PHASE/AMP CAL ##############################


# now we can do some band averaging and check the resulting time-varying gain
time_gain = np.vstack(vis_bp_avg);
time_var  = np.vstack(vis_bp_var);
time_vals = np.hstack(bp_cal_timestamps);
time_vals = (time_vals - time_vals[0])/60;

time_var[time_gain==0]  = 1e50;

# now we should calculate these on a per-antenna basis.  
time_gainsol = zeros( (time_gain.shape[0], 7), dtype=np.complex64 );
input_pairs = np.array(crosscorr).T;
for n in xrange(time_gain.shape[0]):
    vis = time_gain[n,:];
    var = time_var[n,:];
    fitter = scape.fitting.NonLinearLeastSquaresFit(apply_gains, initial_gains)
    fitter.fit(input_pairs, np.vstack((vis.real, vis.imag)), np.vstack( (1/sqrt(var.real), 1/sqrt(var.real)) ));
    #fitter.fit(input_pairs, np.vstack((vis.real, vis.imag)));
    full_params[params_to_fit] = fitter.params * np.sign(fitter.params[2 * ref_input_index])
    time_gainsol[n,:] = full_params.view(np.complex128)


# --------------------------------------------------------------------
# --- Plot the gain as a function of time.
# --------------------------------------------------------------------
time_amp = np.abs(time_gainsol);
time_amp[time_amp==0] = np.nan;
time_phase = np.arctan2(time_gainsol.imag, time_gainsol.real);
time_phase[np.isnan(time_amp)] = nan;

tcal = np.hstack(bp_cal_timestamps);
tcal = (tcal - tcal[0])/60;
fig = figure(num=4, figsize=(8,6))
plot(tcal, time_amp);
xlabel('Time from track start [minutes]');
ylabel('Amp');
legend( ('ant1', 'ant2', 'ant3', 'ant4', 'ant5', 'ant6', 'ant7') );
title('KAT7 Antenna Amplitude of Gain');

fig = figure(num=5, figsize=(8,6))
plot(tcal, time_phase*180/np.pi);
xlabel('Time from track start [minutes]');
ylabel('Phase [degrees]');
legend( ('ant1', 'ant2', 'ant3', 'ant4', 'ant5', 'ant6', 'ant7') );
title('KAT7 Antenna Phase of Gain');




















############################## SCRATCH AREA ##############################

# fit line to phases to determine fixed delay.
#inters = slopes = zeros(7);
#for n in range(7):
#    y = gain_phase[:,n];
#    ind    = np.isnan(y) == False;
#    x = data.channel_freqs[ind];
#    y = y[ind];
#    xx     = pow(x,2);
#    sum_x  = nansum(x);
#    sum_xx = nansum(xx);
#    sum_y  = nansum(y);
#    xy     = x*y;
#    sum_xy = nansum(xy);
#    num    = len(y);
#    inters[n]=(-sum_x*sum_xy+sum_xx*sum_y)/(num*sum_xx-sum_x*sum_x)
#    slopes[n]=(-sum_x*sum_y+num*sum_xy)/(num*sum_xx-sum_x*sum_x)
#
## slope is in radian/Hz, convert to nanosecond
#delays = slopes*1e9/(2*np.pi);
#
## to check if it worked, calculate the scan again
#


# for our purposes, flag out the RFI channels.
#mean_var = imred.nanmean(mean_weight).mean();
#var_lim  = 3*mean_var;
#(x_bad, y_bad)  = (mean_weight > var_lim).nonzero();  # first entry is x-axis, second is y-axis
#chan_bad = unique(x_bad);
#for n in chan_bad:
#    mean_val[n, :] = np.nan;
#    mean_weight[n,:] = inf;

# Next I want to calculate the actual bandpass on the source
# for this I want to use the existing solver.

