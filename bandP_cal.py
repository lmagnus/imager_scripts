#
# Characterisation of bandpass calibrations ... stollen from the
# script that produced first KAT-7 image.
#
# Originally: Ludwig Schwardt
# 18 July 2011
# Now: Lindsay Magnus
# 11 Sep 2011
#

import os
import time
import optparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import ndimage

import katfile
import katpoint
import scape

parser = optparse.OptionParser(usage="%prog [options] <data file> [<data file> ...]",
                               description='Produce image from HDF5 data file(s). Please identify '
                                           'the image target and calibrators via the options.')

parser.add_option('-b', '--bandpass-cal', help="Bandpass calibrator name (**required**)")

parser.add_option('-a', '--ants',
                  help="Comma-separated subset of antennas to use (e.g. 'ant1,ant2'), default is all antennas")
parser.add_option("-f", "--freq-chans", default='200,800',
                  help="Range of frequency channels to keep (zero-based 'start,end', default = %default)")
parser.add_option("--chan-avg", type='int', default=60,
                  help="Number of adjacent frequency channels to average together in MFS imaging (default = %default)")
parser.add_option("--time-avg", type='int', default=90,
                  help="Number of consecutive dumps to average together for imaging (default %default)")
parser.add_option('-p', '--pol', type='choice', choices=['H', 'V'], default='H',
                  help="Polarisation term to use ('H' or 'V'), default is %default")
parser.add_option('-r', '--ref', dest='ref_ant', help="Reference antenna, default is first antenna in file")
parser.add_option("-t", "--time-offset", type='float', default=0.0,
                  help="Time offset to add to DBE timestamps, in seconds (default = %default)")
parser.add_option("--time-slice", type='int', default=30,
                  help="Index of sample relative to start of each scan where vis "
                       "is plotted as function of frequency (default = %default)")
parser.add_option("--freq-slice", type='int', default=250,
                  help="Frequency channel index for which vis is plotted as a function of time (default = %default)")
(opts, args) = parser.parse_args()



# Quick way to set options for use with cut-and-pasting of script bits
# opts = optparse.Values()
# opts.image_target = 'Cen A'
# opts.bandpass_cal = '3C 273'
# opts.gain_cal = 'PKS 1421-490'
# opts.ants = None
# opts.freq_chans = '200,800'
# opts.chan_avg = 60
# opts.time_avg = 90
# opts.pol = 'H'
# opts.ref_ant = 'ant2'
# opts.time_offset = 0.0
# opts.time_slice = 30
# opts.freq_slice = 250
# import glob
# args = sorted(glob.glob('*.h5'))
# args = ['1313238698.h5', '1313240388.h5']

# Frequency channel range to keep, and number of channels to average together into band
freq_chans = [int(chan_str) for chan_str in opts.freq_chans.split(',')]
first_chan, one_past_last_chan = freq_chans[0], freq_chans[1]
channels_per_band, dumps_per_vis = opts.chan_avg, opts.time_avg
# Slices for plotting
time_slice = opts.time_slice
freq_slice = opts.freq_slice - first_chan

# Latest KAT-7 antenna positions and H / V cable delays via recent baseline cal (1313748602 dataset, not joint yet)
new_ants = {
  'ant1' : ('ant1, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 25.095 -9.095 0.045, , 1.22', 23220.506e-9, 23228.551e-9),
  'ant2' : ('ant2, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 90.288 26.389 -0.238, , 1.22', 23283.799e-9, 23286.823e-9),
  'ant3' : ('ant3, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 3.985 26.899 -0.012, , 1.22', 23407.970e-9, 23400.221e-9),
  'ant4' : ('ant4, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -21.600 25.500 0.000, , 1.22', 23514.801e-9, 23514.801e-9),
  'ant5' : ('ant5, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -38.264 -2.586 0.371, , 1.22', 23676.033e-9, 23668.223e-9),
  'ant6' : ('ant6, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -61.580 -79.685 0.690, , 1.22', 23782.854e-9, 23782.150e-9),
  'ant7' : ('ant7, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -87.979 75.756 0.125, , 1.22', 24047.672e-9, 24039.237e-9),
}

katpoint.logger.setLevel(30)

################################ LOAD DATA #####################################

print "Opening data file(s)..."

# Open data files
data = katfile.open(args, ref_ant=opts.ref_ant, channel_range=(first_chan, one_past_last_chan - 1),
                    time_offset=opts.time_offset)

# Create antenna objects with latest positions for antennas used in experiment, and list of inputs and cable delays
ants = dict([(ant.name, katpoint.Antenna(new_ants[ant.name][0])) for ant in data.ants])
inputs, delays = [], {}
for ant in sorted(ants):
    if ant + opts.pol in data.inputs:
        inputs.append(ant + opts.pol)
        delays[ant + opts.pol] = new_ants[ant][1 if opts.pol == 'H' else 2]
# Extract available cross-correlation products, as pairs of indices into input list
crosscorr = [corrprod for corrprod in data.all_corr_products(inputs) if corrprod[0] != corrprod[1]]

# Extract frequency information
center_freqs = data.channel_freqs
wavelengths = katpoint.lightspeed / center_freqs

# Create catalogue of targets found in data
targets = katpoint.Catalogue()
for scan_ind, cs_ind, state, target in data.scans():
    if state == 'track' and target.name not in targets:
        targets.add(target)


for i,curr_tar in enumerate(targets):

    ############################## STOP FRINGES ON BANDPASS CAL####################################

    print "Assembling bandpass calibrator data and checking fringe stopping..."

    # Assemble fringe-stopped visibility data for main (bandpass) calibrator
    orig_cal_vis_samples, cal_vis_samples, cal_timestamps = [], [], []
    for scan_ind, cs_ind, state, target in data.scans():
        if state != 'track' or target != curr_tar:
            continue
        timestamps = data.timestamps()
        if len(timestamps) < 2:
            continue
        vis_pre = np.zeros((len(timestamps), len(wavelengths), len(crosscorr)), dtype=np.complex64)
        vis_post = np.zeros((len(timestamps), len(wavelengths), len(crosscorr)), dtype=np.complex64)
        # Iterate through baselines and assemble visibilities
        for n, (indexA, indexB) in enumerate(crosscorr):
            inputA, inputB = inputs[indexA], inputs[indexB]
            antA, antB = inputA[:-1], inputB[:-1]
            vis = data.vis((inputA, inputB))
            vis_pre[:, :, n] = vis
            # Get uvw coordinates of A->B baseline
            u, v, w = target.uvw(ants[antB], timestamps, ants[antA])
            # Number of turns of phase that signal B is behind signal A due to geometric delay
            geom_delay_turns = - w[:, np.newaxis] / wavelengths
            # Number of turns of phase that signal B is behind signal A due to cable / receiver delay
            cable_delay_turns = (delays[inputB] - delays[inputA]) * center_freqs
            # Visibility <A, B*> has phase (A - B), therefore add (B - A) phase to stop fringes (i.e. do delay tracking)
            vis *= np.exp(2j * np.pi * (geom_delay_turns + cable_delay_turns))
            vis_post[:, :, n] = vis
        orig_cal_vis_samples.append(vis_pre)
        cal_vis_samples.append(vis_post)
        cal_timestamps.append(timestamps)


    ############################## BANDPASS CAL ####################################

    print "Performing bandpass calibration on '%s'..." % (curr_tar.name,)

    # Vector that contains real and imaginary gain components for all signal paths
    full_params = np.zeros(2 * len(inputs))
    # Indices of gain parameters that will be optimised
    params_to_fit = range(len(full_params))
    ref_input_index = inputs.index(data.ref_ant + opts.pol)
    # Don't fit the imaginary component of the gain on the reference signal path (this is assumed to be zero)
    params_to_fit.pop(2 * ref_input_index + 1)
    initial_gains = np.tile([1., 0.], len(inputs))[params_to_fit]

    def apply_gains(params, input_pairs, model_vis=1.0):
        """Apply relevant antenna gains to model visibility to estimate measurements.

        This corrupts the ideal model visibilities by applying a set of complex
        antenna gains to them.

        Parameters
        ----------
        params : array of float, shape (2 * N - 1,)
            Array of gain parameters with 2 parameters per signal path (real and
            imaginary components of complex gain), except for phase reference input
            which has a single real component
        input_pairs : array of int, shape (2, M)
            Input pair(s) for which visibilities will be calculated. The inputs are
            specified by integer indices into the main input list.
        model_vis : complex or array of complex, shape (M,), optional
            The modelled (ideal) source visibilities on each specified baseline.
            The default model is that of a point source with unit flux density.

        Returns
        -------
        estm_vis : array of float, shape (2,) or (2, M)
            Estimated visibilities specified by their real and imaginary components

        """
        full_params[params_to_fit] = params
        antA, antB = input_pairs[0], input_pairs[1]
        reA, imA, reB, imB = full_params[2 * antA], full_params[2 * antA + 1], full_params[2 * antB], full_params[2 * antB + 1]
        # Calculate gain product (g_A g_B*)
        reAB, imAB = reA * reB + imA * imB, imA * reB - reA * imB
        re_model, im_model = np.real(model_vis), np.imag(model_vis)
        return np.vstack((reAB * re_model - imAB * im_model, reAB * im_model + imAB * re_model)).squeeze()

    # Vector that contains gain phase components for all signal paths
    phase_params = np.zeros(len(inputs))
    # Indices of phase parameters that will be optimised
    phase_params_to_fit = range(len(phase_params))
    # Don't fit the phase on the reference signal path (this is assumed to be zero)
    phase_params_to_fit.pop(ref_input_index)
    initial_phases = np.zeros(len(inputs))[phase_params_to_fit]


    # Solve for antenna bandpass gains
    bandpass_gainsols = []
    bp_source_vis = np.ones(len(center_freqs)) #This needs to be updated once we had good SEDs for our calibrators LM
    # Iterate over solution intervals
    for solint_vis in cal_vis_samples:
        gainsol = np.zeros((len(inputs), solint_vis.shape[1]), dtype=np.complex64)
        input_pairs = np.tile(np.array(crosscorr).T, solint_vis.shape[0])
        # Iterate over frequency channels
        for n in xrange(solint_vis.shape[1]):
            vis, model_vis = solint_vis[:, n, :].ravel(), bp_source_vis[n]
            fitter = scape.fitting.NonLinearLeastSquaresFit(lambda p, x: apply_gains(p, x, model_vis), initial_gains)
            fitter.fit(input_pairs, np.vstack((vis.real, vis.imag)))
            full_params[params_to_fit] = fitter.params * np.sign(fitter.params[2 * ref_input_index])
            gainsol[:, n] = full_params.view(np.complex128)
        bandpass_gainsols.append(gainsol)


    fig = plt.figure(i)
    for n in range(len(inputs)):
        ax = fig.add_subplot(len(inputs),1,n+1)
        for m in range(len(bandpass_gainsols)):
            plt.plot(np.abs(bandpass_gainsols[m][n,:]))


