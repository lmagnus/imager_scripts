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
from scipy import ndimage



# Stephen's "script" for reducing data.
# CAVEAT:  What you are about to see is really bad python programming.
#
# Code Outline
# 0.  Read in the data
# 1.  Bandpass calibrate the data
#     a.  select only bandpass cal observations, calculate mean bandpass
# 2.  Calculate the system temperatures for the track
# 3.  Phase calibrate the data
#      a. select only phase cal, 
#      b. apply the bandpasss
#      c. bin?
#      d. calculate the time-varying gain
# 4.  Apply calibrations to source data
#      a. select source data
#      b. Apply bandpass
#      c. apply time-varying gain cal
#      d. bin?
# 5.  Compare the theoretical rms to band-avg rms
# 5.  Map.
# 6.  Remove source, calculate dynamic range.
#

#------------------------------------------------------------
# step -1:  read in/fringe-stop the data
#------------------------------------------------------------
#data = katfile.open('1313842550_3c273_big.h5', channel_range=(200, 849));
#new_ants = {
#  'ant1' : ('ant1, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 25.095 -9.095 0.045, , 1.22', 23220.506e-9, 23228.551e-9),
#  'ant2' : ('ant2, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 90.288 26.389 -0.238, , 1.22', 23282.549e-9, 23285.573e-9),
#  'ant3' : ('ant3, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 3.985 26.899 -0.012, , 1.22', 23406.720e-9, 23398.971e-9),
#  'ant4' : ('ant4, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -21.600 25.500 0.000, , 1.22', 23514.801e-9, 23514.801e-9),
#  'ant5' : ('ant5, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -38.264 -2.586 0.371, , 1.22', 23676.033e-9, 23668.223e-9),
#  'ant6' : ('ant6, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -61.580 -79.685 0.690, , 1.22', 23782.854e-9, 23782.150e-9),
#  'ant7' : ('ant7, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -87.979 75.756 0.125, , 1.22', 24048.922e-9, 24040.487e-9),
#}

#data = katfile.open('1315924459.h5', channel_range=(200, 800));
data = katfile.open('/home/ffuser/sjcm/carina.h5', channel_range=(200, 800));
#data = katfile.open('/home/schwardt/CenA/1313240388.h5', channel_range=(200, 800));

#new_ants = {
#  'ant1' : ('ant1, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 25.095 -9.095 0.045, , 1.22',   23220.506e-9, 23228.551e-9, 'ped1'),
#  'ant2' : ('ant2, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 90.288 26.389 -0.238, , 1.22',  23282.549e-9, 23285.573e-9, 'ped2'),
#  'ant3' : ('ant3, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 3.985 26.899 -0.012, , 1.22',   23406.720e-9, 23398.971e-9, 'ped3'),
#  'ant4' : ('ant4, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -21.600 25.500 0.000, , 1.22',  23514.801e-9, 23514.801e-9, 'ped4'),
#  'ant5' : ('ant5, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -38.264 -2.586 0.371, , 1.22',  23674.52e-9, 23667.0e-9, 'ped5'),
#  'ant6' : ('ant6, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -61.580 -79.685 0.690, , 1.22', 23782.614e-9, 23781.954e-9, 'ped6'),
#  'ant7' : ('ant7, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -87.979 75.756 0.125, , 1.22',  24046.038e-9, 24038.167e-9, 'ped7'),
#}

# from baseline solution on data from 8.26.2011
new_ants = {
  'ant1' : ('ant1, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, +25.095 -09.095 +0.045, , 1.22', 23243.947e-9, 23228.551e-9,'ped1'),
  'ant2' : ('ant2, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, +90.217 +26.202 -0.567, , 1.22', 23393.860e-9, 23285.573e-9,'ped2'),
  'ant3' : ('ant3, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, +04.110 +26.746 -0.298, , 1.22', 23428.945e-9, 23398.971e-9,'ped3'),
  'ant4' : ('ant4, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -21.363 +25.295 -0.432, , 1.22', 23537.214e-9, 23514.801e-9,'ped4'),
  'ant5' : ('ant5, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -37.958 -02.769 -0.130, , 1.22', 23697.833e-9, 23667.000e-9,'ped5'),
  'ant6' : ('ant6, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -61.187 -79.738 +0.193, , 1.22', 23802.640e-9, 23781.954e-9,'ped6'),
  'ant7' : ('ant7, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -87.545 +75.347 -0.596, , 1.22', 24066.958e-9, 24038.167e-9,'ped7'),
}


opts = struct;
opts.pol = 'H';
# my observation
opts.image_target = 'PKS 1934-63';
opts.bandpass_cal = '3C 273';
opts.gain_cal     = 'PKS 1421-490';

# simon's carina obs
opts.image_target = 'PKS 1215-457';
opts.bandpass_cal = 'PKS 1934-63';
opts.gain_cal     = 'PKS 1036-697';
opts.image_target = 'Car II';

#opts.image_target = 'Cen A';
#opts.bandpass_cal = 'PKS 1421-490';
#opts.gain_cal     = 'PKS 1421-490';
opts.time_slice   = 30;
num_ants          = len(data.ants);
num_bases         = num_ants*(num_ants-1)/2;
t_start           = data.timestamps().min();


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

#tsys = redsub.calc_tsys(data, noise_diode_times, opts);
#
## --------------------------------------------------------------------
## --- Plot the system temperature
## --------------------------------------------------------------------
#figure(31);
#figure(num=31, figsize=(12,10))
#for plotnum, (this_tsys, this_time) in enumerate(tsys):
#    subplot(3,3,plotnum+1)
#    if(len(this_tsys)>0):
#        plot(center_freqs/1e9, transpose(this_tsys[1:len(this_tsys)-1]));
#    if plotnum%3 == 0:
#        ylabel('tsys [K]');
#    if plotnum in range(4,7):
#        xlabel('Frequency [GHz]');
#    title(inputs[plotnum]);
#
#
#tsys_mean = [];
#tsys_time = [];
#for plotnum, (this_tsys, this_time) in enumerate(tsys):
#    mtsys = this_tsys.mean(axis=1);
#    tsys_mean.append(mtsys);
#    tsys_time.append(this_time);
#
#
#figure(31);
#figure(num=32, figsize=(12,10))
#for plotnum, (this_tsys, this_time) in enumerate(tsys):
#    subplot(3,3,plotnum+1)
#    if(len(this_tsys)>0):
#        plot(center_freqs/1e9, transpose(this_tsys[1:len(this_tsys)-1]));
#    if plotnum%3 == 0:
#        ylabel('tsys [K]');
#    if plotnum in range(4,7):
#        xlabel('Frequency [GHz]');
#    title(inputs[plotnum]);
#
#
#figure(33).clear();
#for plotnum, this_tsys in enumerate(tsys_mean):
#    subplot(3,3,plotnum+1)
#    this_time = (tsys_time[plotnum] - t_start)/60;
#    if(len(this_tsys)>0):
#        plot(this_time, this_tsys);
#    if plotnum%3 == 0:
#        ylabel('tsys [K]');
#    if plotnum in range(4,7):
#        xlabel('Time [min] from start');
#    title(inputs[plotnum]);





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
axvals = [0.99*center_freqs.min()/1e9, 1.01*center_freqs.max()/1e9, 0.99*miny, 1.01*maxy];

fig = figure(num=2, figsize=(8,6))
plot(center_freqs/1e9, gain_abs);
xlabel('Frequency [GHz]');
ylabel('Normalized Bandpass Shape');
legend( inputs );
title('KAT7 Antenna Bandpass Amplitude');

fig = figure(num=3, figsize=(8,6))
plot(center_freqs/1e9, gain_phase);
xlabel('Frequency [GHz]');
ylabel('Normalized Bandpass Shape');
legend( inputs );
title('KAT7 Antenna Bandpass Phase');


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

# ok.  now for the phase calibrator.  
# first we upload the gain calibrator data and fringe-track it.

(gain_cal_vis_samples, gain_cal_timestamps, noise_diode_times) = redsub.stop_fringes(data, gain_cal, opts, True);

# --------------------------------------------------------------------
# --- Apply the average bandpass
# --------------------------------------------------------------------

vis_gain = [];
vis_gain_avg = [];
vis_gain_var = [];
# so we apply this to gain_cal_vis_samples
for n in range( len(gain_cal_vis_samples) ):
    with_bp = zeros( (gain_cal_vis_samples[n].shape) , np.complex128 );
    with_bp_avg = zeros( (gain_cal_vis_samples[n].shape[0], gain_cal_vis_samples[n].shape[2]) , np.complex128 );
    for m, vis in enumerate(gain_cal_vis_samples[n]):
        with_bp[m,:,:] = vis/final_bp_scale;
    with_bp_avg = imred.nanmean(with_bp, axis=1);
    with_bp_var = imred.nanvarc(with_bp, axis=1);
    vis_gain.append(with_bp);
    vis_gain_avg.append(with_bp_avg);
    vis_gain_var.append(with_bp_var);

# next we check what the average bandpass on the gain calibrator looks like.
gainsol_pre = redsub.scan_bandpass(gain_cal_vis_samples, False, data.ref_ant, opts, inputs, input_pairs);


gainsol_post = redsub.scan_bandpass(vis_gain, False, data.ref_ant, opts, inputs, input_pairs);

# --------------------------------------------------------------------
# --- Plot the average bandpass on the gain calibrator
# --------------------------------------------------------------------

# -- BEFORE CORRECTION
gain_abs = np.abs(gainsol_pre);
gain_abs[gain_abs==0] = np.nan;
gain_phase = np.arctan2(gainsol_pre.imag, gainsol_pre.real);
gain_phase[np.isnan(gain_abs)] = np.nan;
miny = np.nanmin(gain_abs);
maxy = np.nanmax(gain_abs);
axvals = [0.99*center_freqs.min()/1e9, 1.01*center_freqs.max()/1e9, 0.99*miny, 1.01*maxy];

fig = figure(num=4, figsize=(8,6))
plot(center_freqs/1e9, gain_abs);
xlabel('Frequency [GHz]');
ylabel('Normalized Bandpass Shape');
legend( inputs );
title('KAT7 Gain Calibrator Bandpass Amplitude');

fig = figure(num=5, figsize=(8,6))
plot(center_freqs/1e9, gain_phase);
xlabel('Frequency [GHz]');
ylabel('Normalized Bandpass Shape');
legend( inputs );
title('KAT7 Gain Calibrator Bandpass Phase');


# -- AFTER CORRECTION

gain_abs = np.abs(gainsol_post);
gain_abs[gain_abs==0] = np.nan;
gain_phase = np.arctan2(gainsol_post.imag, gainsol_post.real);
gain_phase[np.isnan(gain_abs)] = np.nan;
miny = np.nanmin(gain_abs);
maxy = np.nanmax(gain_abs);
axvals = [0.99*center_freqs.min()/1e9, 1.01*center_freqs.max()/1e9, 0.99*miny, 1.01*maxy];

fig = figure(num=6, figsize=(8,6))
plot(center_freqs/1e9, gain_abs);
xlabel('Frequency [GHz]');
ylabel('Normalized Bandpass Shape');
legend( inputs );
title('KAT7 Corrected Gain Calibrator Bandpass Amplitude');

fig = figure(num=7, figsize=(8,6))
plot(center_freqs/1e9, gain_phase);
xlabel('Frequency [GHz]');
ylabel('Normalized Bandpass Shape');
legend( inputs );
title('KAT7 Corrected Gain Calibrator Bandpass Phase');


# now we can do some band averaging and check the resulting time-varying gain
time_gain = np.vstack(vis_gain_avg);
time_var  = np.vstack(vis_gain_var);
time_vals = np.hstack(gain_cal_timestamps);

time_var[time_gain==0] = 1e50;
# now we should calculate these on a per-antenna basis.  
time_gainsol = redsub.calculate_gains(time_gain, time_var, inputs, crosscorr, data.ref_ant, opts);

# --------------------------------------------------------------------
# --- Plot the gain as a function of time with no time average.
# --------------------------------------------------------------------
time_amp = np.abs(time_gainsol);
time_amp[time_amp==0] = np.nan;
time_phase = np.arctan2(time_gainsol.imag, time_gainsol.real);
time_phase = np.unwrap(time_phase);
time_phase[np.isnan(time_amp)] = nan;

tcal = np.hstack(gain_cal_timestamps);
tcal = (tcal - tcal[0])/60;
fig = figure(num=8, figsize=(8,6))
plot(tcal, time_amp);
xlabel('Time from track start [minutes]');
ylabel('Amp');
legend( inputs );
title('KAT7 Antenna Amplitude of Gain');

fig = figure(num=9, figsize=(8,6))
plot(tcal, time_phase*180/np.pi);
xlabel('Time from track start [minutes]');
ylabel('Phase [degrees]');
legend( inputs );
title('KAT7 Antenna Phase of Gain');



# bin our calibrator scans, then calculate the gains

(mean_vis, mean_var, mean_time) = redsub.bin_scans(vis_gain_avg, vis_gain_var, gain_cal_timestamps);

# now we can do some band averaging and check the resulting time-varying gain
time_gain = np.vstack(mean_vis);
time_var  = np.vstack(mean_var);
time_vals = np.hstack(mean_time);

# now we should calculate these on a per-antenna basis.  
time_gainsol = redsub.calculate_gains(time_gain, time_var, inputs, crosscorr, data.ref_ant, opts);

mean_amp_single_val = np.abs(time_gainsol);
mean_amp_single_val = mean_amp_single_val.mean(axis=0);
mean_amp_single_val_norm = mean_amp_single_val/mean_amp_single_val.mean();



# --------------------------------------------------------------------
# --- Plot the gain as a function of time with no time average.
# --------------------------------------------------------------------
time_amp = np.abs(time_gainsol);
time_amp[time_amp==0] = np.nan;
time_phase = np.arctan2(time_gainsol.imag, time_gainsol.real);
time_phase[np.isnan(time_amp)] = nan;
time_phase = np.unwrap(time_phase, axis=0);


tplot = (time_vals - t_start)/60;
fig = figure(num=10, figsize=(8,6))
plot(tplot, time_amp, '+-');
xlabel('Time from track start [minutes]');
ylabel('Amp');
legend( inputs );
title('KAT7 Antenna Amplitude of time-avg Gain');

fig = figure(num=11, figsize=(8,6))
plot(tplot, time_phase*180/np.pi, '+-');
xlabel('Time from track start [minutes]');
ylabel('Phase [degrees]');
legend( inputs );
title('KAT7 Antenna Phase of time-avg Gain');


# next we calculate the phase and amplitude correction factors.

num_ants = time_phase.shape[1];
phase_corr = np.zeros( ( len(data.timestamps()), num_ants) );
amp_corr = np.zeros( ( len(data.timestamps()), num_ants) );
for n in range(num_ants):
    phase_corr[:,n] = np.interp(x=data.timestamps(), xp=time_vals, fp=time_phase[:,n]);
    amp_corr[:,n]   = np.interp(x=data.timestamps(), xp=time_vals, fp=time_amp[:,n]);



# the point of the phase correction is to remove instrumental phase 
#  between the telescopes.  Point of amplitude correction is to remove the 
#  instrumental gain between the telescopes.
#  As such, the mean correction for the amplitude should be equal to 1
#  This assumes the source is point-like and the true flux of the source is the 
#   average of what's seen on all telescopes
mean_amp = amp_corr.mean(axis=1);
for m, amp in enumerate(transpose(amp_corr)):
    amp_corr[:,m] = amp/mean_amp;






#------------------------------------------------------------
#  Lastly, we read in our source data, fringe-track, bandpass calibrate, and gain calibrate.
#------------------------------------------------------------

(image_vis_samples, image_timestamps, noise_diode_times) = redsub.stop_fringes(data, image_target, opts, True);

# --------------------------------------------------------------------
# --- Apply the average bandpass
# --------------------------------------------------------------------

vis_image = [];
vis_image_avg = [];
vis_image_var = [];
# so we apply this to image_vis_samples
for n in range( len(image_vis_samples) ):
    with_bp = zeros( (image_vis_samples[n].shape) , np.complex128 );
    with_bp_avg = zeros( (image_vis_samples[n].shape[0], image_vis_samples[n].shape[2]) , np.complex128 );
    for m, vis in enumerate(image_vis_samples[n]):
        with_bp[m,:,:] = vis/final_bp_scale;
    with_bp_avg = imred.nanmean(with_bp, axis=1);
    with_bp_var = imred.nanvarc(with_bp, axis=1);
    vis_image.append(with_bp);
    vis_image_avg.append(with_bp_avg);
    vis_image_var.append(with_bp_var);


image_vis_calibrated = [];
image_vis_calibrated_norm = [];
# time for some average gain-ness. 
for m in range( len(vis_image_avg) ) :
    this_time      = image_timestamps[m];
    this_cal_vis   = np.zeros( vis_image_avg[m].shape, np.complex128) ;
    this_cal_vis_norm = np.zeros( vis_image_avg[m].shape, np.complex128) ;
    for n, (indexA, indexB) in enumerate(crosscorr):
        amp_corr_A  = np.interp(this_time, xp=data.timestamps(), fp=amp_corr[:,indexA]);
        amp_corr_B  = np.interp(this_time, xp=data.timestamps(), fp=amp_corr[:,indexB]);
        ph_corr_A   = np.interp(this_time, xp=data.timestamps(), fp=phase_corr[:,indexA]);
        ph_corr_B   = np.interp(this_time, xp=data.timestamps(), fp=phase_corr[:,indexB]);
        amp_norm_A  = mean_amp_single_val_norm[indexA];
        amp_norm_B  = mean_amp_single_val_norm[indexB];
        # we want to divide by the amp_corr and remove the ph_corr
        bl_ph_corr  = ph_corr_A - ph_corr_B; 
        bl_amp_corr = (amp_corr_A*amp_corr_B);
        bl_corr     = bl_amp_corr*exp(1j*bl_ph_corr);
        bl_corr_norm= amp_norm_A*amp_norm_B*exp(1j*bl_ph_corr);
        aa = vis_image_avg[m][:,n];
        this_cal_vis[:,n] = aa.data/(bl_corr);
        this_cal_vis_norm[:,n] = aa.data/bl_corr_norm;
    image_vis_calibrated.append(this_cal_vis);
    image_vis_calibrated_norm.append(this_cal_vis_norm);    

# shall we see on an antenna basis?
time_gain = np.vstack(image_vis_calibrated);
time_var  = np.vstack(vis_image_var);
time_vals = (np.hstack(image_timestamps) - t_start)/60;
time_var[time_gain==0] = 1e50;
# now we should calculate these on a per-antenna basis.  
time_gainsol = redsub.calculate_gains(time_gain, time_var, inputs, crosscorr, data.ref_ant, opts);

num_timestamps = time_gain.shape[0];
timestamps     = np.hstack(image_timestamps);


# --- plot -- 

time_amp = np.abs(time_gainsol);
time_amp[time_amp==0] = np.nan;
time_phase = np.arctan2(time_gainsol.imag, time_gainsol.real);
time_phase[np.isnan(time_amp)] = nan;

tplot = (time_vals - tcal[0])/60;
fig = figure(num=12, figsize=(8,6))
plot(time_vals, time_amp, '+-');
xlabel('Time from track start [minutes]');
ylabel('Amp');
legend( inputs );
title('Amplitude of time-avg Gain on source - AMP INTERP');

fig = figure(num=13, figsize=(8,6))
plot(time_vals, time_phase*180/np.pi, '+-');
xlabel('Time from track start [minutes]');
ylabel('Phase [degrees]');
legend( inputs );
title('Phase of time-avg Gain on source');


# let's compare when we do the simple scaling of the bandpass 
# with a single number.

# shall we see on an antenna basis?
time_gain = np.vstack(image_vis_calibrated_norm);
time_var  = np.vstack(vis_image_var);
time_vals = (np.hstack(image_timestamps) - t_start)/60;
time_var[time_gain==0] = 1e50;
# now we should calculate these on a per-antenna basis.  
time_gainsol = redsub.calculate_gains(time_gain, time_var, inputs, crosscorr, data.ref_ant, opts);


src_flux =  np.abs(time_gain);
src_flux =  np.mean(src_flux);
src_removed = time_gain - src_flux;

# --- plot -- 

time_amp = np.abs(time_gainsol);
time_amp[time_amp==0] = np.nan;
time_phase = np.arctan2(time_gainsol.imag, time_gainsol.real);
time_phase[np.isnan(time_amp)] = nan;

tplot = (time_vals - tcal[0])/60;
fig = figure(num=14, figsize=(8,6))
plot(time_vals, time_amp, '+-');
xlabel('Time from track start [minutes]');
ylabel('Amp');
legend( inputs );
title('Amplitude of time-avg Gain on source - AMP SINGLE');



# calculate uvw for source
uvw = np.zeros((3, num_timestamps, len(wavelengths), len(crosscorr)))

for n, (indexA, indexB) in enumerate(crosscorr):
    inputA, inputB = inputs[indexA], inputs[indexB]
    antA, antB = inputA[:-1], inputB[:-1]
    # Get uvw coordinates of A->B baseline as multiples of the channel wavelength
    uvw[:, :, :, n] = np.array(target.uvw(ants[antB], timestamps, ants[antA]))[:, :, np.newaxis] / wavelengths

mean_uvw = uvw.mean(axis=2);
u_samples, v_samples, w_samples = mean_uvw[0].ravel(), mean_uvw[1].ravel(), mean_uvw[2].ravel();
uvdist = np.sqrt(u_samples * u_samples + v_samples * v_samples)
vis_samples = np.vstack(image_vis_calibrated_norm).ravel();

vis_no_source = np.vstack(image_vis_calibrated_norm).ravel();
vis_no_source = vis_no_source - src_flux;








# now let's produce a dirty image.
print "Producing dirty image of '%s'..." % (image_target.name,)

# Set up image grid coordinates (in radians)                                                                           #                                                          
# First get some basic data parameters together (center freq in Hz, primary beamwidth in rads)                         #                                                          
band_center = center_freqs[len(center_freqs) // 2]
ref_ant = ants[data.ref_ant]
primary_beam_width = ref_ant.beamwidth * katpoint.lightspeed / band_center / ref_ant.diameter
# The pixel size is a fixed fraction of the synthesised beam width                                                     #                                                          
image_grid_step = 0.1 / uvdist.max()
# The number of pixels is determined by the primary beam width                                                         #                                                          
# (and kept a power of two for compatibility with other packages that use the FFT instead of DFT)                      #                                                          
image_size = 2 ** int(np.log2(primary_beam_width / image_grid_step))
num_pixels = image_size * image_size
# Create image pixel (l,m) coordinates similar to CASA (in radians)                                                    #                                                          
m_range = (np.arange(image_size) - image_size // 2) * image_grid_step
l_range = np.flipud(-m_range)
l_image, m_image = np.meshgrid(l_range, m_range)
n_image = np.sqrt(1 - l_image*l_image - m_image*m_image)
lm_positions = np.array([l_image.ravel(), m_image.ravel()]).transpose()

# Direct Fourier imaging (DFT) of dirty beam and image                                                                 #                                                          
dirty_beam = np.zeros((image_size, image_size), dtype='double')
dirty_image = np.zeros((image_size, image_size), dtype='double')
for u, v, vis in zip(u_samples, v_samples, vis_samples):
    arg = 2*np.pi*(u*l_image + v*m_image)
    dirty_beam += np.cos(arg)
    dirty_image += np.abs(vis) * np.cos(arg - np.angle(vis))

dirty_beam *= n_image / len(vis_samples)
dirty_image *= n_image / len(vis_samples)

dirty_beam = np.zeros((image_size, image_size), dtype='double')
dirty_image_no_source = np.zeros((image_size, image_size), dtype='double')
for u, v, vis in zip(u_samples, v_samples, vis_no_source):
    arg = 2*np.pi*(u*l_image + v*m_image)
    dirty_beam += np.cos(arg)
    dirty_image_no_source += np.abs(vis) * np.cos(arg - np.angle(vis))

dirty_beam *= n_image / len(vis_samples)
dirty_image_no_source *= n_image / len(vis_samples)

# Plot ranges for casapy                                                                                               #                                                          
arcmins = 60 * 180 / np.pi
l_plot = l_range * arcmins
m_plot = m_range * arcmins

figure(17).clear();
plot(u_samples, v_samples, '.', markersize=2)
plot(-u_samples, -v_samples, '.r', markersize=2)
xlabel('u (lambda)')
ylabel('v (lambda)')
title("UV coverage for '%s' target" % (image_target.name,))
uvmax = max(np.abs(axis()))
axis('image')
axis((-uvmax, uvmax, -uvmax, uvmax))

figure(18).clear();
imshow(dirty_beam, origin='lower', interpolation='bicubic', extent=[l_plot[0], l_plot[-1], m_plot[0], m_plot[-1]])
xlabel('l (arcmins)')
ylabel('m (arcmins)')
title('Dirty beam')
axis('image')

figure(19).clear()
imshow(dirty_image, origin='lower', interpolation='bicubic', extent=[l_plot[0], l_plot[-1], m_plot[0], m_plot[-1]])
xlabel('l (arcmins)')
ylabel('m (arcmins)')
title("Dirty image of '%s' at %.0f MHz" % (image_target.name, band_center / 1e6))
axis('image')


figure(33).clear()
imshow(dirty_image_no_source, origin='lower', interpolation='bicubic', extent=[l_plot[0], l_plot[-1], m_plot[0], m_plot[-1]])
xlabel('l (arcmins)')
ylabel('m (arcmins)')
colorbar()
title("Dirty image of '%s' at %.0f MHz" % (image_target.name, band_center / 1e6))
axis('image')


################################## CLEAN #######################################

print "CLEANing the image..."

# The CLEAN variant(s) that will be used
def omp(A, y, S, At_times=None, A_column=None, N=None, printEveryIter=1, resThresh=0.0):
    """Orthogonal Matching Pursuit.
    
    This approximately solves the linear system A x = y for sparse x, where A is
    an MxN matrix with M << N.
    
    Parameters
    ----------
    A : array, shape (M, N)
        The measurement matrix of compressed sensing (None for implicit functions)
    y : array, shape (M,)
        A vector of measurements
    S : integer
        Maximum number of sparse components to find (sparsity level), between 1 and M.
    At_times : function
        Function that calculates At_times(x) = A' * x implicitly. It takes
        an array of shape (M,) as argument and returns an array of shape (N,).
        Default is None.
    A_column : function
        Function that returns the n-th column of A as A_column(n) = A[:, n]. It takes
        an integer as argument and returns an array of shape (M,). Default is None.
    N : integer
        Number of columns in matrix A (dictionary size in MP-speak). Default is None,
        which means it is automatically determined from A.
    printEveryIter : integer
        A progress line is printed every 'printEveryIter' iterations (0 for no
        progress report). The default is a progress report after every iteration.
    resThresh : real
        Stop iterating if the residual l2-norm (relative to the l2-norm of the 
        measurements y) falls below this threshold. Default is 0.0 (no threshold).

    Returns
    -------
    x : array of same type as y, shape (N,)
        The approximate sparse solution to A x = y.
    
    """
    # Convert explicit A matrix to functional form (or use provided functions)
    if At_times is None:
        At_times = lambda x: np.dot(A.conjugate().transpose(), x) 
        A_column = lambda n: A[:, n]
        N = A.shape[1]
    M = len(y)
    # (1) -- Initialization
    residual = y
    resSize = 1.0
    atoms = np.zeros((M, S), dtype=y.dtype)
    atomIndex = np.zeros(S, dtype='int32')
    # (6) -- Loop until the desired number of atoms are found (may stop before that)
    for s in xrange(S):
        # (2) -- Easy optimization problem (peakpicking the dirty residual image |A' r|)
        atomIndex[s] = np.abs(At_times(residual)).argmax()
        # Stop if this atom has already been included in active set (if OMP revisits an
        # atom, it can only be due to round-off error), or if residual threshold is crossed
        if (atomIndex[s] in atomIndex[:s]) or (resSize < resThresh):
            atomIndex = atomIndex[:s]
            break
        # (3) -- Update matrix of atoms
        atoms[:, s] = A_column(atomIndex[s])
        activeAtoms = atoms[:, :s+1]
        # (4) -- Solve least-squares problem to obtain new signal estimate
        atomWeights = np.linalg.lstsq(activeAtoms, y)[0].real
        # (5) -- Calculate new residual
        residual = y - np.dot(activeAtoms, atomWeights)
        resSize = np.linalg.norm(residual) / np.linalg.norm(y)
        if printEveryIter and ((s+1) % printEveryIter == 0):
            print "%d : atom = %d, dual = %.3e, residual l2 = %.3e" % \
                  (s, atomIndex[s], atomWeights[s], resSize)
    # (7) -- Create signal estimate
    x = np.zeros(N, dtype=y.dtype)
    x[atomIndex] = atomWeights[:len(atomIndex)]
    if printEveryIter:
        print 'omp: atoms = %d, residual = %.3e' % (sum(x != 0.0), resSize)
    return x

def omp_plus(A, y, S, At_times=None, A_column=None, N=None, printEveryIter=1, resThresh=0.0):
    """Positive Orthogonal Matching Pursuit.

    This approximately solves the linear system A x = y for sparse positive real x,
    where A is an MxN matrix with M << N. This is very similar to the NNLS algorithm of
    Lawson & Hanson (Solving Least Squares Problems, 1974, Chapter 23).

    Parameters
    ----------
    A : array, shape (M, N)
        The measurement matrix of compressed sensing (None for implicit functions)
    y : array, shape (M,)
        A vector of measurements
    S : integer
        Maximum number of sparse components to find (sparsity level), between 1 and M.
    At_times : function
        Function that calculates At_times(x) = A' * x implicitly. It takes
        an array of shape (M,) as argument and returns an array of shape (N,).
        Default is None.
    A_column : function
        Function that returns the n-th column of A as A_column(n) = A[:, n]. It takes
        an integer as argument and returns an array of shape (M,). Default is None.
    N : integer
        Number of columns in matrix A (dictionary size in MP-speak). Default is None,
        which means it is automatically determined from A.
    printEveryIter : integer
        A progress line is printed every 'printEveryIter' iterations (0 for no
        progress report). The default is a progress report after every iteration.
    resThresh : real
        Stop iterating if the residual l2-norm (relative to the l2-norm of the
        measurements y) falls below this threshold. Default is 0.0 (no threshold).

    Returns
    -------
    x : array of same type as y, shape (N,)
        The approximate sparse positive solution to A x = y.

    """
    # Convert explicit A matrix to functional form (or use provided functions)
    if At_times is None:
        At_times = lambda x: np.dot(A.conjugate().transpose(), x)
        A_column = lambda n: A[:, n]
        N = A.shape[1]
    M = len(y)
    # Initialization
    residual = y
    resSize = 1.0
    atoms = np.zeros((M, S), dtype=y.dtype)
    atomIndex = -np.ones(S, dtype='int32')
    atomWeights = np.zeros(S, dtype='float64')
    atomHistory = set()
    numAtoms = 0
    iterCount = 0
    # Maximum iteration count suggested by Lawson & Hanson
    iterMax = 3 * N

    try:
        # MAIN LOOP (a la NNLS) to find all atoms
        # Loop until the desired number of components / atoms are found, or the
        # residual size drops below the threshold (an earlier exit is also possible)
        while (numAtoms < S) and (resSize >= resThresh):
            # Form the real part of the dirty image residual A' r
            # This happens to be the negative gradient of 0.5 || y - A x ||_2^2,
            # the associated l2-norm (least-squares) objective, and also the dual vector
            dual = At_times(residual).real
            # Ensure that no existing atoms will be selected again
            dual[atomIndex[:numAtoms]] = 0.0
            # Loop until a new atom with positive weight is found, or die trying
            while True:
                newAtom = dual.argmax()
                # Stop if atom is already in active set, or gradient is non-positive
                if dual[newAtom] <= 0.0:
                    break
                # Tentatively add new atom to active set
                atomIndex[numAtoms] = newAtom
                atoms[:, numAtoms] = A_column(newAtom)
                activeAtoms = atoms[:, :numAtoms+1]
                # Solve unconstrained least-squares problem (Gram-Schmidt orthogonalisation step)
                newWeights = np.linalg.lstsq(activeAtoms, y)[0].real
                # If weight of new atom is non-positive, discard it and go for next best atom
                if newWeights[-1] <= 0.0:
                    dual[newAtom] = 0.0
                else:
                    break
            if dual[newAtom] <= 0.0:
                break
            # If search has been in this state before, it will get stuck in endless loop
            # until iterMax is reached, which is pointless
            # TODO: check the effort involved in this check (maybe we don't need it if the
            # endless loop is due to a bug somewhere else?)
            atomState = tuple(sorted(atomIndex[:numAtoms+1]))
            if atomState in atomHistory:
                print "endless loop detected, terminating"
                break
            else:
                atomHistory.add(atomState)
            numAtoms += 1
            # SECONDARY LOOP (a la NNLS) to get all atom weights to be positive simultaneously
            # Forced to terminate if it takes too long
            while iterCount <= iterMax:
                iterCount += 1
                # Check for non-positive weights
                nonPos = [n for n in xrange(len(newWeights)) if newWeights[n] <= 0.0]
                if len(nonPos) == 0:
                    break
                # Interpolate between old and new weights so that at least one atom
                # with negative weight now has zero weight, and can therefore be discarded
                oldWeights = atomWeights[:numAtoms]
                alpha = oldWeights[nonPos] / (oldWeights[nonPos] - newWeights[nonPos])
                worst = alpha.argmin()
                oldWeights += alpha[worst] * (newWeights - oldWeights)
                # Make sure the selected atom really has 0 weight (round-off could change it)
                oldWeights[nonPos[worst]] = 0.0
                # Only keep the atoms with positive weights (could be more efficient...)
                goodAtoms = [n for n in xrange(len(oldWeights)) if oldWeights[n] > 0.0]
                numAtoms = len(goodAtoms)
                print "iter %d : best atom = %d, found negative weights, worst at %d, reduced atoms to %d" % \
                      (iterCount, newAtom, atomIndex[nonPos[worst]], numAtoms)
                atomIndex[:numAtoms] = atomIndex[goodAtoms].copy()
                atomIndex[numAtoms:] = -1
                activeAtoms = atoms[:, goodAtoms].copy()
                atoms[:, :numAtoms] = activeAtoms
                atoms[:, numAtoms:] = 0.0
                atomWeights[:numAtoms] = atomWeights[goodAtoms].copy()
                atomWeights[numAtoms:] = 0.0
                # Solve least-squares problem again to get new proposed atom weights
                newWeights = np.linalg.lstsq(activeAtoms, y)[0].real
            if iterCount > iterMax:
                break
            # Accept new weights, update residual and continue with main loop
            atomWeights[:numAtoms] = newWeights
            residual = y - np.dot(activeAtoms, newWeights)
            resSize = np.linalg.norm(residual) / np.linalg.norm(y)
            if printEveryIter and (iterCount % printEveryIter == 0):
                print "iter %d : best atom = %d, dual = %.3e, atoms = %d, residual l2 = %.3e" % \
                      (iterCount, newAtom, dual[newAtom], numAtoms, resSize)

    # Return last results on Ctrl-C, for the impatient ones
    except KeyboardInterrupt:
        # Create sparse solution vector
        x = np.zeros(N, dtype='float64')
        x[atomIndex[:numAtoms]] = atomWeights[:numAtoms]
        if printEveryIter:
            print 'omp: atoms = %d, residual = %.3e (interrupted)' % (sum(x != 0.0), resSize)

    else:
        # Create sparse solution vector
        x = np.zeros(N, dtype='float64')
        x[atomIndex[:numAtoms]] = atomWeights[:numAtoms]
        if printEveryIter:
            print 'omp: atoms = %d, residual = %.3e' % (sum(x != 0.0), resSize)

    return x

# Set up CLEAN boxes around main peaks in dirty image
# Original simplistic attempt at auto-boxing
# mask = (dirty_image > 0.3 * dirty_image.max()).ravel()
## First try to pick a decent threshold based on a knee shape in sorted amplitudes
sorted_dirty = np.sort(dirty_image.ravel())
norm_sd_x = np.linspace(0, 1, len(sorted_dirty))
norm_sd_y = sorted_dirty / sorted_dirty[-1]
# Break graph into coarse steps, in order to get less noisy slope estimates
norm_sd_coarse_steps = norm_sd_y.searchsorted(np.arange(0., 1., 0.05))
norm_sd_coarse_x = norm_sd_coarse_steps / float(len(sorted_dirty))
norm_sd_coarse_y = norm_sd_y[norm_sd_coarse_steps]
norm_sd_coarse_slope = np.diff(norm_sd_coarse_y) / np.diff(norm_sd_coarse_x)
# Look for rightmost point in graph with a tangent slope of around 1
knee = norm_sd_coarse_steps[norm_sd_coarse_slope.searchsorted(2., side='right') + 1]
# Look for closest point in graph to lower right corner of plot
# knee = np.sqrt((norm_sd_x - 1) ** 2 + norm_sd_y ** 2).argmin()
mask = (dirty_image > sorted_dirty[knee]).ravel()
mask_image = mask.reshape(image_size, image_size)
# Create measurement matrix (potentially *very* big - use smaller mask to reduce it)
masked_phi = np.exp(2j * np.pi * np.dot(np.c_[u_samples, v_samples], lm_positions.T[:, mask]))
# Desired number of pixels (the sparsity level m of the signal)
num_components = 20
vis_snr_dB = 20
# Pick a more sensible threshold in the case of noiseless data
effective_snr_dB = min(vis_snr_dB, 40.0)
res_thresh = 1.0 / np.sqrt(1.0 + 10 ** (effective_snr_dB / 10.0))

# Clean the image
masked_comps = omp_plus(A=masked_phi, y=vis_samples, S=num_components, resThresh=res_thresh)
model_vis_samples = np.dot(masked_phi, masked_comps)
clean_components = np.zeros(image_size * image_size)
clean_components[mask] = masked_comps
clean_components = clean_components.reshape(image_size, image_size)

# Create residual image
residual_vis = vis_samples - model_vis_samples
residual_image = np.zeros((image_size, image_size), dtype='double')
for u, v, vis in zip(u_samples, v_samples, residual_vis):
    arg = 2*np.pi*(u*l_image + v*m_image)
    residual_image += np.abs(vis) * np.cos(arg - np.angle(vis))
residual_image *= n_image / len(residual_vis)

# Create restoring beam from inner part of dirty beam

# Threshold the dirty beam image and identify blobs
blob_image, blob_count = ndimage.label(dirty_beam > 0.2)
# Pick the centre blob and enlarge it slightly to make up for the aggressive thresholding in the previous step
centre_blob = ndimage.binary_dilation(blob_image == blob_image[image_size // 2, image_size // 2])
# Fit Gaussian beam to central part of dirty beam
beam_weights = centre_blob * dirty_beam
lm = np.vstack((l_image.ravel(), m_image.ravel()))
beam_cov = np.dot(lm * beam_weights.ravel(), lm.T) / beam_weights.sum()
restoring_beam = np.exp(-0.5 * np.sum(lm * np.dot(np.linalg.inv(beam_cov), lm), axis=0)).reshape(image_size, image_size)
# Create clean image by restoring with clean beam
clean_image = np.zeros((image_size, image_size), dtype='double')
comps_row, comps_col = clean_components.nonzero()
origin = (image_size // 2, image_size // 2 - 1)
for comp_row, comp_col in zip(comps_row, comps_col):
    flux = clean_components[comp_row, comp_col]
    clean_image += ndimage.shift(flux * restoring_beam, (comp_row - origin[0], comp_col - origin[1]))
# Get final image and corresponding DR estimate
final_image = clean_image + residual_image

print "Estimated dynamic range = ", final_image.max() / residual_image.std()

fig = plt.figure(20).clear();
imshow(0.2 * mask_image, interpolation='nearest', origin='lower',
          extent=[l_plot[0], l_plot[-1], m_plot[0], m_plot[-1]], cmap=mpl.cm.gray_r, vmin=0., vmax=1.)
imshow(np.ma.masked_array(clean_components, clean_components == 0), interpolation='nearest', origin='lower',
          extent=[l_plot[0], l_plot[-1], m_plot[0], m_plot[-1]])
xlabel('l (arcmins)')
ylabel('m (arcmins)')
title('Clean components')
axis('image')

figure(21).clear()
imshow(residual_image, origin='lower', interpolation='bicubic',
          extent=[l_plot[0], l_plot[-1], m_plot[0], m_plot[-1]])
colorbar();
imshow(mask_image, interpolation='nearest', origin='lower',
          extent=[l_plot[0], l_plot[-1], m_plot[0], m_plot[-1]], cmap=mpl.cm.gray_r, alpha=0.5)
xlabel('l (arcmins)')
ylabel('m (arcmins)')
title("Residual image")
axis('image')

figure(22).clear()
imshow(final_image, origin='lower', interpolation='bicubic', extent=[l_plot[0], l_plot[-1], m_plot[0], m_plot[-1]])
xlabel('l (arcmins)')
ylabel('m (arcmins)')
title("Clean image of '%s' at %.0f MHz" % (image_target.name, band_center / 1e6))
axis('image')




## checking the system temperature in time.
#aa = data.file['MetaData']['Sensors']['Antennas']['ant6']['pos.actual-scan-elev']['value']
#bb = data.file['MetaData']['Sensors']['Antennas']['ant6']['pos.actual-scan-elev']['timestamp']
#tt = (bb - t_start)/60;
#
#elev_tsys = np.interp(x=(tsys_time-t_start)/60, xp=tt, fp=aa);
#
#ang = np.zeros( (50) );
#for n in range(20,70):
#    ang[n-20] = n*np.pi/180;
#
#
##figure(34).clear();
##for plotnum, this_tsys in enumerate(tsys_mean):
##    subplot(3,3,plotnum+1)
##    this_time = (tsys_time[plotnum] - t_start)/60;
##    min_elev = elev_tsys[plotnum].min();
##    max_tsys = this_tsys.max();
##    tsys_line = max_tsys/sin(ang)*sin(min_elev*np.pi/180);
##    if(len(this_tsys)>0):
#        plot(elev_tsys[plotnum], this_tsys, '+');
##        plot(ang*180/np.pi, tsys_line);
#    if plotnum%3 == 0:
#        ylabel('tsys [K]');
#    if plotnum in range(4,7):
#        xlabel('Elevation');
#    title(inputs[plotnum]);
#
#
#
#figure()
#plot(elev_tsys, 
#
#    if(len(this_tsys)>0):
#        plot(this_time, this_tsys);
#    if plotnum%3 == 0:
#        ylabel('tsys [K]');
#    if plotnum in range(4,7):
#        xlabel('Time [min] from start');
#    title(inputs[plotnum]);
#




# check for my residual
