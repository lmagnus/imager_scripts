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




# GLOBAL VARIABLES 
# Latest KAT-7 antenna positions and H / V cable delays via recent baseline cal (1313748602 dataset, not joint yet)
new_ants = {
  'ant1' : ('ant1, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 25.095 -9.095 0.045, , 1.22', 23220.506e-9, 23228.551e-9, 'ped1'),
  'ant2' : ('ant2, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 90.288 26.389 -0.238, , 1.22', 23283.799e-9, 23286.823e-9, 'ped2'),
  'ant3' : ('ant3, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 3.985 26.899 -0.012, , 1.22', 23407.970e-9, 23400.221e-9, 'ped3'),
  'ant4' : ('ant4, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -21.600 25.500 0.000, , 1.22', 23514.801e-9, 23514.801e-9, 'ped4'),
  'ant5' : ('ant5, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -38.264 -2.586 0.371, , 1.22', 23676.033e-9, 23668.223e-9, 'ped5'),
  'ant6' : ('ant6, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -61.580 -79.685 0.690, , 1.22', 23782.854e-9, 23782.150e-9, 'ped6'),  
  'ant7' : ('ant7, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -87.979 75.756 0.125, , 1.22', 24047.672e-9, 24039.237e-9, 'ped7'),
}

# from baseline differencing of sources
#old = [[25.095, -9.095, 0.045] ,
#[90.288, 26.389, -0.238],
#[3.985, 26.899, -0.012],
#[-21.600, 25.500, 0.000],
#[-38.264, -2.586, 0.371],
#[-61.580, -79.685, 0.690],
#[-87.979, 75.756, 0.125]]
#
#adding the difference  GOT WORSE!!!
#subtracting the difference -- MUCH MUCH BETTER!!!!  
# time interval 8 min -- amazing dynamic range.  313
#new_ants = {
#  'ant1' : ('ant1, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 25.0950000 -9.09500000 0.0450000000,  , 1.22', 23220.506e-9, 23228.551e-9, 'ped1'),
#  'ant2' : ('ant2, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 90.2877166 26.38624 -0.245031,  , 1.22', 23283.799e-9, 23286.823e-9, 'ped2'),
#  'ant3' : ('ant3, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 3.9852065 26.900599 -0.036116,  , 1.22', 23407.970e-9, 23400.221e-9, 'ped3'),
#  'ant4' : ('ant4, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -21.604464 25.50187987 -0.01166634,  , 1.22', 23514.801e-9, 23514.801e-9, 'ped4'),
#  'ant5' : ('ant5, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -38.26475 -2.5877790 0.3715746,  , 1.22', 23676.033e-9, 23668.223e-9, 'ped5'),
#  'ant6' : ('ant6, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -61.5849815 -79.689752 0.688983825,  , 1.22', 23782.854e-9, 23783.150e-9, 'ped6'),  
#  'ant7' : ('ant7, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -87.98377566 75.7600725 0.126947246,  , 1.22', 24047.672e-9, 24039.237e-9, 'ped7'),
#}

##long_10hr_track --- PRETTY DAMN GOOD. greatest offset from zero is 5 degrees.  (ant2 and ant7 not so great)
new_ants = {
  'ant1' : ('ant1, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 25.0950000 -9.09500000 0.045000000,  , 1.22', 23220.506e-9, 23228.551e-9, 'ped1'),
  'ant2' : ('ant2, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 90.2992336 26.3971374 -0.226073325, , 1.22', 23283.799e-9, 23286.823e-9, 'ped2'),
  'ant3' : ('ant3, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 3.98474582 26.8928983 0.000404602,  , 1.22', 23407.970e-9, 23400.221e-9, 'ped3'),
  'ant4' : ('ant4, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -21.6053051 25.4935739 0.018615350, , 1.22', 23514.801e-9, 23514.801e-9, 'ped4'),
  'ant5' : ('ant5, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -38.2719878 -2.59173391 0.391362235, , 1.22', 23676.033e-9, 23668.223e-9, 'ped5'),
  'ant6' : ('ant6, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -61.5945075 -79.6988745 0.701597845, , 1.22', 23782.854e-9, 23783.150e-9, 'ped6'),
  'ant7' : ('ant7, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -87.9943477 75.7434337 0.149769360, , 1.22', 24047.672e-9, 24039.237e-9, 'ped7'),
}

##long_10hr_track --- PRETTY DAMN GOOD. Carina delays
new_ants = {
  'ant1' : ('ant1, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 25.0950000 -9.09500000 0.045000000,  , 1.22', 23220.506e-9, 23228.551e-9, 'ped1'),
  'ant2' : ('ant2, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 90.2992336 26.3971374 -0.226073325, , 1.22', 23282.549e-9, 23286.823e-9, 'ped2'),
  'ant3' : ('ant3, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 3.98474582 26.8928983 0.000404602,  , 1.22', 23407.970e-9, 23400.221e-9, 'ped3'),
  'ant4' : ('ant4, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -21.6053051 25.4935739 0.018615350, , 1.22', 23516.051e-9, 23514.801e-9, 'ped4'),
  'ant5' : ('ant5, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -38.2719878 -2.59173391 0.391362235, , 1.22', 23677.283e-9, 23668.223e-9, 'ped5'),
  'ant6' : ('ant6, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -61.5945075 -79.6988745 0.701597845, , 1.22', 23781.604e-9, 23783.150e-9, 'ped6'),
  'ant7' : ('ant7, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -87.9943477 75.7434337 0.149769360, , 1.22', 24047.672e-9, 24039.237e-9, 'ped7'),
}

# modified ant 2 and 7 -- REALLY GOOD!  BETTER THAN BEFORE.
new_ants = {
  'ant1' : ('ant1, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 25.0950 -9.0950 0.0450,  , 1.22', 23220.506e-9, 23228.551e-9, 'ped1'),
  'ant2' : ('ant2, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 90.2844 26.3804 -0.22636, , 1.22', 23282.549e-9, 23286.823e-9, 'ped2'),
  'ant3' : ('ant3, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 3.98474 26.8929 0.0004046,  , 1.22', 23407.970e-9, 23400.221e-9, 'ped3'),
  'ant4' : ('ant4, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -21.6053 25.4936 0.018615, , 1.22', 23516.051e-9, 23514.801e-9, 'ped4'),
  'ant5' : ('ant5, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -38.2720 -2.5917 0.391362, , 1.22', 23677.283e-9, 23668.223e-9, 'ped5'),
  'ant6' : ('ant6, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -61.5945 -79.6989 0.701598, , 1.22', 23781.604e-9, 23783.150e-9, 'ped6'),
  'ant7' : ('ant7, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -87.9881 75.7543 0.138305,, 1.22', 24047.672e-9, 24039.237e-9, 'ped7'),
}



def stop_fringes(data, source_name, opts, remove_noise=True):
    """ Function that should do the fringe tracking
        And return a list of observations on source_name """

    # Create antenna objects with latest positions for antennas used in experiment, and list of inputs and cable delays
    ants = dict([(ant.name, katpoint.Antenna(new_ants[ant.name][0])) for ant in data.ants])
    inputs, delays, peds = [], {}, []
    for ant in sorted(ants):
        if ant + opts.pol in data.inputs:
            inputs.append(ant + opts.pol)
            delays[ant + opts.pol] = new_ants[ant][1 if opts.pol == 'H' else 2]
            peds.append(new_ants[ant][3]);

    # Extract available cross-correlation products, as pairs of indices into input list 
    crosscorr = [corrprod for corrprod in data.all_corr_products(inputs) if corrprod[0] != corrprod[1]]
    input_pairs = np.array(crosscorr).T;

    # Extract frequency information
    center_freqs = data.channel_freqs
    wavelengths = katpoint.lightspeed / center_freqs

    crosscorr = [corrprod for corrprod in data.all_corr_products(inputs) if corrprod[0] != corrprod[1]]
    orig_source_vis_samples, source_vis_samples, source_timestamps, noise_vis, noise_timestamps = [], [], [], [], []
    for scan_ind, cs_ind, state, target in data.scans():
        if state != 'track' or target != source_name:
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
        orig_source_vis_samples.append(vis_pre)
        source_vis_samples.append(vis_post)
        source_timestamps.append(timestamps)

    if remove_noise==True:
        # ---- remove the noise source diode firings from the data set.
        datatime = data.timestamps();
        rows = len(datatime);
        columns = len(ants);
        dimensions = (rows,columns);
        # first we sort out a boolean array for when the source was on
        noise_diode_times = [];
        sensors_group = data.file['MetaData/Sensors'];

        j=1;
        numOn = 0;
        pedNums = peds;
        for i in range(len(pedNums)):
            noise_value     = sensors_group['Pedestals'][pedNums[i]]['rfe3.rfe15.noise.coupler.on']['value'];
            noise_timestamp = sensors_group['Pedestals'][pedNums[i]]['rfe3.rfe15.noise.coupler.on']['timestamp'];
            time_on  = noise_timestamp[noise_value=='1'];
            time_off = noise_timestamp[noise_value=='0'];
            this_ant_times = np.zeros( (len(time_on), 2) );
            # all i really care about is knowing when the noise source is on, 
            # so i want to know when it goes off after the first instance of going on 
            for index, n in enumerate(time_on):
                t_on = n;
                aa   = (time_off - n) > 0;
                if(len(aa.ravel().nonzero()[0]) != 0):
                    index_off = aa.ravel().nonzero()[0][0];
                    t_off     = time_off[index_off];
                else:
                    # noise is on at the end of the day.
                    t_off     = timestamps.max();
                this_ant_times[index,0] = t_on;
                this_ant_times[index,1] = t_off;
            noise_diode_times.append(this_ant_times);
        # so now times is a list, each entry corresponds to a given antenna
        # within each list member, there is an Nx2 array, with t_on, t_off 
        # first let's remove those times from the calibrator visibility data sets
        # unfortunately you could have the noise diode going on one set of antennas and not another 
        # or they might not be synchronized.  In which case, the best way to deal with this would be 
        # to set the visibilities to NaN in the cal observations, even though I hate actually doing
        # this because you're changing your data.  But alas, noise diode on data shouldn't be used in 
        # in the calibrator visibilities anyway.
        for n, cal_time in enumerate(source_timestamps):
            for m, times in enumerate(noise_diode_times):
                for t_on, t_off in times:
                    ind_time = (cal_time > t_on) & (cal_time < t_off);
                    ind_time = ind_time.ravel().nonzero();
                    ind_vis  = (input_pairs[0] == m) | (input_pairs[1] == m);
                    ind_vis  = ind_vis.ravel().nonzero();
                    if(len(ind_time[0]) != 0):
                        for k in ind_time[0]:
                            source_vis_samples[n][k, :, ind_vis[0]] = np.nan + 1j*np.nan;
        return (source_vis_samples, source_timestamps, noise_diode_times);
    else:
        return (source_vis_samples, source_timestamps);
    
        


def stop_fringes_all(data, opts, remove_noise=True):
    """ Function that should do the fringe tracking
        And return a list of observations on all sources """

    # Create antenna objects with latest positions for antennas used in experiment, and list of inputs and cable delays
    ants = dict([(ant.name, katpoint.Antenna(new_ants[ant.name][0])) for ant in data.ants])
    inputs, delays, peds = [], {}, []
    for ant in sorted(ants):
        if ant + opts.pol in data.inputs:
            inputs.append(ant + opts.pol)
            delays[ant + opts.pol] = new_ants[ant][1 if opts.pol == 'H' else 2]
            peds.append(new_ants[ant][3]);

    # Extract available cross-correlation products, as pairs of indices into input list 
    crosscorr = [corrprod for corrprod in data.all_corr_products(inputs) if corrprod[0] != corrprod[1]]
    input_pairs = np.array(crosscorr).T;

    # Extract frequency information
    center_freqs = data.channel_freqs
    wavelengths = katpoint.lightspeed / center_freqs

    crosscorr = [corrprod for corrprod in data.all_corr_products(inputs) if corrprod[0] != corrprod[1]]
    orig_source_vis_samples, source_vis_samples, source_timestamps, target_list, noise_vis, noise_timestamps = [], [], [], [], [],[]
    for scan_ind, cs_ind, state, target in data.scans():
        if state != 'track':
            continue
        timestamps = data.timestamps()
        if len(timestamps) < 2:
            continue
        target_list.append(target);
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
        orig_source_vis_samples.append(vis_pre)
        source_vis_samples.append(vis_post)
        source_timestamps.append(timestamps)

    if remove_noise==True:
        # ---- remove the noise source diode firings from the data set.
        datatime = data.timestamps();
        rows = len(datatime);
        columns = len(ants);
        dimensions = (rows,columns);
        # first we sort out a boolean array for when the source was on
        noise_diode_times = [];
        sensors_group = data.file['MetaData/Sensors'];

        j=1;
        numOn = 0;
        pedNums = peds;
        for i in range(len(pedNums)):
            noise_value     = sensors_group['Pedestals'][pedNums[i]]['rfe3.rfe15.noise.coupler.on']['value'];
            noise_timestamp = sensors_group['Pedestals'][pedNums[i]]['rfe3.rfe15.noise.coupler.on']['timestamp'];
            time_on  = noise_timestamp[noise_value=='1'];
            time_off = noise_timestamp[noise_value=='0'];
            this_ant_times = np.zeros( (len(time_on), 2) );
            # all i really care about is knowing when the noise source is on, 
            # so i want to know when it goes off after the first instance of going on 
            for index, n in enumerate(time_on):
                t_on = n;
                aa   = (time_off - n) > 0;
                if(len(aa.ravel().nonzero()[0]) != 0):
                    index_off = aa.ravel().nonzero()[0][0];
                    t_off     = time_off[index_off];
                else:
                    # noise is on at the end of the day.
                    t_off     = timestamps.max();
                this_ant_times[index,0] = t_on;
                this_ant_times[index,1] = t_off;
            noise_diode_times.append(this_ant_times);
        # so now times is a list, each entry corresponds to a given antenna
        # within each list member, there is an Nx2 array, with t_on, t_off 
        # first let's remove those times from the calibrator visibility data sets
        # unfortunately you could have the noise diode going on one set of antennas and not another 
        # or they might not be synchronized.  In which case, the best way to deal with this would be 
        # to set the visibilities to NaN in the cal observations, even though I hate actually doing
        # this because you're changing your data.  But alas, noise diode on data shouldn't be used in 
        # in the calibrator visibilities anyway.
        for n, cal_time in enumerate(source_timestamps):
            for m, times in enumerate(noise_diode_times):
                for t_on, t_off in times:
                    ind_time = (cal_time > t_on) & (cal_time < t_off);
                    ind_time = ind_time.ravel().nonzero();
                    ind_vis  = (input_pairs[0] == m) | (input_pairs[1] == m);
                    ind_vis  = ind_vis.ravel().nonzero();
                    if(len(ind_time[0]) != 0):
                        for k in ind_time[0]:
                            source_vis_samples[n][k, :, ind_vis[0]] = np.nan + 1j*np.nan;
        return (source_vis_samples, source_timestamps, target_list, noise_diode_times);
    else:
        return (source_vis_samples, source_timestamps, target_list);
    
        
def calc_tsys(data, noise_diode_times, opts):
    """ Function that calculates the system temperature of your data"""


    # --------------------------------------------------------------------
    # --- First we get the power measurements from the auto-correlations
    # --------------------------------------------------------------------
    power = [];
    ants = dict([(ant.name, katpoint.Antenna(new_ants[ant.name][0])) for ant in data.ants])
    inputs, peds = [], []
    for ant in sorted(ants):
        if ant + opts.pol in data.inputs:
            inputs.append(ant + opts.pol)
            peds.append(new_ants[ant][3]);
    
    num_ants = len(peds);
    antVals = sorted(ants); 

    for m, times in enumerate(noise_diode_times):
        this_power = np.zeros( (len(times), len(data.channel_freqs), 4) );
        this_time  = np.zeros( (len(times) ) );
        for n, (t_on, t_off) in enumerate(times):
            ind_time_on  = (data.timestamps() > t_on) & (data.timestamps() < t_off);
            ind_off_pre  = (data.timestamps() > t_on - 6) & (data.timestamps() < t_on - 1);
            ind_off_post = (data.timestamps() > t_off +1) & (data.timestamps() < t_off +6);
            ind_time_off = ind_off_pre | ind_off_post;
            ind_time_on  = ind_time_on.ravel().nonzero()[0];
            ind_time_off = ind_time_off.ravel().nonzero()[0];
            autoOff      = data.vis( (inputs[m], inputs[m]))[ind_time_off,:];
            autoOn       = data.vis( (inputs[m], inputs[m]))[ind_time_on,:];
            this_power[n,:,0] = autoOn.mean(axis=0);
            this_power[n,:,1] = autoOff.mean(axis=0);
            this_power[n,:,2] = autoOn.var(axis=0);
            this_power[n,:,3] = autoOff.var(axis=0);
            this_time[n]    = (t_on + t_off)/2;
        list_entry = (this_power, this_time);
        power.append(list_entry);

    # --------------------------------------------------------------------
    # --- Next we get the actual noise profiles
    # --------------------------------------------------------------------
    data_freqs = data.channel_freqs/1e6;
    ND =  np.zeros( (num_ants, len(data_freqs) ) );
    # The noise diode values are in a completely random-ass order
    for i, ant in enumerate(antVals):
        if opts.pol == 'H':
            thisData = data.file['MetaData']['Configuration']['Antennas'][ant]['h_coupler_noise_diode_model'];
        else:
            thisData = data.file['MetaData']['Configuration']['Antennas'][ant]['v_coupler_noise_diode_model'];
        nd_model = NoiseDiodeModel(thisData[:,0]/1e6, thisData[:,1], **dict(thisData.attrs));
        aa = np.interp(xp=nd_model.freq, x=data_freqs, fp=nd_model.temp);
        ND[i,:] = aa;

    # --------------------------------------------------------------------
    # --- Calculate tsys under assumption of no bright source
    # --------------------------------------------------------------------
    y = [];
    tsys = [];
    for m, pow in enumerate(power):
        (this_power, this_time) = pow;
        this_y     = this_power[:,:,0]/this_power[:,:,1];
        this_var_y = this_power[:,:,2]/(this_power[:,:,1]**2) + (this_power[:,:,0]**2)/(this_power[:,:,1]**4)*this_power[:,:,3];
        this_entry = (this_y, this_var_y, this_time);
        y.append(this_entry);
        this_tsys_val = np.zeros( (this_y.shape) );
        for k, this_samp in enumerate(this_y):
            this_tsys_val[k,:] = ND[m,:]/(this_samp-1);
        this_entry = (this_tsys_val, this_time);
        tsys.append(this_entry);

    # tsys is a list of tsys values, where the list goes in antenna order, and     within each
    #  there are Nx650 tsys values, for times the noise diode went on, and band    . 
    
    # --------------------------------------------------------------------
    # --- Calculate tsys without assumption of no bright source
    # --- requires, opacity, etc, etc. 
    # --- also requires knowing the source flux, so we'll skip this for now
    # --------------------------------------------------------------------
    #
    ## an estimate for tau.
    #nd_times = np.vstack(noise_diode_times);
    #nd_times = np.hstack(nd_times);
    #nd_times = unique(nd_times);
    #
    #temp       = sensors_group['Enviro']['asc.air.temperature']['value'];
    #temp_time  = sensors_group['Enviro']['asc.air.temperature']['timestamp'];
    #temp       = np.interp(xp=temp_time, fp=temp, x=nd_times)+273.15;
    #humid      = sensors_group['Enviro']['asc.air.relative-humidity']['value']    ;
    #humid_time = sensors_group['Enviro']['asc.air.relative-humidity']['timesta    mp'];
    #humid      = np.interp(xp=humid_time, fp=humid, x=nd_times)+273.15;
    #
    ## i have a C function that does this, and I just calculated it for the
    ## mean values since this effect should be small
    #tauZenith = 0.0134;
    #
    ## elevation -- just using an approximation that it's at the center time.
    #
    #el = [];
    #for i, ant in enumerate(antVals):
    #   ant_sensors = sensors_group['Antennas'][ant]['pos.actual-scan-elev'];
    #   original_coord = remove_duplicates(ant_sensors);
    #   timeVal = original_coord['timestamp'];
    #   elval   = original_coord['value'];
    #   this_el = np.interp(xp=timeVal, fp=elval, x=tsys[i][1])*np.pi/180;
    #   el.append(this_el);
    #
    #tauZenith = 0.0134;
    #Tsrc = 0.034*30;
    #tsys_noapprox = [];
    #for m, (yval, vary, ytime) in enumerate(y):
    #    this_tsys = np.zeros( (yval.shape) );
    #    for n, this_y in enumerate(yval):
    #        this_tsys[n,:] = ( (1-this_y)*Tsrc*exp(-tauZenith/sin(el[m][n])) +     ND[m,:] )/(this_y-1);
    #    this_entry = (this_tsys, ytime);
    #    tsys_noapprox.append(this_entry);
    #

    return tsys;


def calculate_gains(time_gain, time_var, inputs, crosscorr, ref_ant, opts):

    full_params = np.zeros(2 * len(inputs))
    # Indices of gain parameters that will be optimised                                                                             
    params_to_fit = range(len(full_params))
    ref_input_index = inputs.index(ref_ant + opts.pol)
    # Don't fit the imaginary component of the gain on the reference signal pat    h (this is assumed to be zero)                   
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
    

    time_gainsol = np.zeros( (time_gain.shape[0], len(inputs)), dtype=np.complex64 );
    input_pairs = np.array(crosscorr).T;
    for n in xrange(time_gain.shape[0]):
        vis = time_gain[n,:];
        var = time_var[n,:];
        fitter = scape.fitting.NonLinearLeastSquaresFit(apply_gains, initial_gains)
        fitter.fit(input_pairs, np.vstack((vis.real, vis.imag)), np.vstack( (1/np.sqrt(var.real), 1/np.sqrt(var.real)) ));
        full_params[params_to_fit] = fitter.params * np.sign(fitter.params[2 * ref_input_index])
        time_gainsol[n,:] = full_params.view(np.complex128)
    
    return time_gainsol;



def scan_bandpass(bp_cal_vis_samples, normalize, ref_ant, opts, inputs, input_pairs):
    """
       Calculate the average bandpass a list of calibrator observations """


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
    
    
    num_chans = bp_cal_vis_samples[0].shape[1];
    num_bases = bp_cal_vis_samples[0].shape[2]; 
    num_ants  = len(inputs);

    
    bp_list = [];
    for n in range(len(bp_cal_vis_samples)):
        chan_mean = bp_cal_vis_samples[n];
        if(chan_mean.ndim==2):
            chan_mean = chan_mean.reshape( (1, chan_mean.shape[0], chan_mean.shape[1]) );
        chan_mean = imred.nanmean(chan_mean, axis=1);
        # preserve phase, set amplitude to unity
        chan_mean = chan_mean/np.abs(chan_mean);
        # repeat the aa matrix, and apply it to the data.
        chan_mean = chan_mean.reshape( (chan_mean.shape[0], 1, chan_mean.shape[1]) );
        chan_mean = np.kron( np.ones( (1, num_chans, 1) ), chan_mean );
        vis_phase_removed = bp_cal_vis_samples[n]*np.conj(chan_mean);
        # append it to the list of tracking events
        bp_list.append(vis_phase_removed);
    
    mean_bp_list = [];
    var_bp_list  = [];
    for n in range(len(bp_list)):
        mean_bp_list.append( imred.nanmean(bp_list[n], axis=0).reshape( (1, num_chans, num_bases) ) );
        var_bp_list.append( imred.nanvar(bp_list[n], axis=0).reshape( (1, num_chans, num_bases) ) );
     
    # take the variance weighted mean.
    weight_array = mean_array = np.zeros( (len(mean_bp_list), mean_bp_list[0].shape[0], mean_bp_list[0].shape[1]) );
    mean_array = np.vstack(mean_bp_list);
    weight_array = np.vstack(var_bp_list);
    weight_array[weight_array==0] = np.inf;
    
    mean_bp = np.average(mean_array, weights=weight_array, axis=0, returned=True);
    mean_val    = mean_bp[0];
    mean_weight = mean_bp[1];
    mean_weight = mean_weight.real;
    mean_val[np.isnan(mean_weight)] = 0;
    mean_weight[np.isnan(mean_weight)] = 1e20;
    
    # Solve for antenna bandpass gains                                             
    # Vector that contains real and imaginary gain components for all signal paths
    
    full_params = np.zeros(2 * len(inputs))
    # Indices of gain parameters that will be optimised                                                                             
    params_to_fit = range(len(full_params))
    ref_input_index = inputs.index(ref_ant + opts.pol)
    # Don't fit the imaginary component of the gain on the reference signal pat    h (this is assumed to be zero)                   
    params_to_fit.pop(2 * ref_input_index + 1)
    initial_gains = np.tile([1., 0.], len(inputs))[params_to_fit]
    
    
    gainsol = np.zeros( (mean_val.shape[0], num_ants), dtype=np.complex64 );
    for n in xrange(mean_val.shape[0]):
        vis = mean_val[n,:];
        var = mean_weight[n,:];
        fitter = scape.fitting.NonLinearLeastSquaresFit(apply_gains, initial_gains);
        fitter.fit(input_pairs, np.vstack((vis.real, vis.imag)), np.vstack( (1/np.sqrt(var.real), 1/np.sqrt(var.real)) ) );
        full_params[params_to_fit] = fitter.params * np.sign(fitter.params[2*ref_input_index])
        gainsol[n,:] = full_params.view(np.complex128)
    
    
       
    # when correcting for the bandpass, all we want to do is correct the variations ACROSS the band
    # so we need to divide our correction by the mean amplitude across the band to ensure we're
    # not changing the value of the data at all.
    if normalize==True:
        bp_mean     = np.abs(gainsol);
        bp_mean[bp_mean==0] = np.nan;
        bp_mean = imred.nanmean(bp_mean, axis=0);
        bp_mean = bp_mean.reshape( (1, num_ants) );
        bp_mean = np.kron( np.ones( (gainsol.shape[0], 1)), bp_mean);
        bp_to_apply = gainsol/bp_mean;
        return (gainsol, bp_to_apply);
    else:
        return gainsol;



def scan_bandpass_weights(bp_cal_vis_samples, bp_var_samples, normalize, ref_ant, opts, inputs, input_pairs):
    """
       Calculate the average bandpass a list of calibrator observations """


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
    
    
    num_chans = bp_cal_vis_samples[0].shape[1];
    num_bases = bp_cal_vis_samples[0].shape[2]; 
    num_ants  = len(inputs);
    
    
    # bp and var are already mean/var, so no need to take it again.
    var_bp_list, mean_bp_list = [], [];
    for n in range(len(bp_cal_vis_samples)):
        chan_mean = bp_cal_vis_samples[n];
        chan_vars = bp_var_samples[n];
        if(chan_mean.ndim==2):
            chan_mean = chan_mean.reshape( (1, chan_mean.shape[0], chan_mean.shape[1]) );
            chan_vars = chan_mean.reshape( (1, chan_vars.shape[0], chan_vars.shape[1]) );
        chan_mean = imred.nanmean(chan_mean, axis=1);
        # preserve phase, set amplitude to unity
        chan_vars = chan_vars/np.abs(chan_mean);
        chan_mean = chan_mean/np.abs(chan_mean);
        # repeat the mean matrix, and apply it to the data.
        chan_mean = chan_mean.reshape( (chan_mean.shape[0], 1, chan_mean.shape[1]) );
        chan_mean = np.kron( np.ones( (1, num_chans, 1) ), chan_mean );
        vis_phase_removed = bp_cal_vis_samples[n]*np.conj(chan_mean);
        # append it to the list of tracking events
        mean_bp_list.append(vis_phase_removed);
        var_bp_list.append(chan_vars);
    
    
    # take the variance weighted mean.
    weight_array = mean_array = np.zeros( (len(mean_bp_list), mean_bp_list[0].shape[0], mean_bp_list[0].shape[1]) );
    mean_array = np.vstack(mean_bp_list);
    weight_array = np.vstack(var_bp_list);
    weight_array[weight_array==0] = np.inf;
    
    mean_bp = np.average(mean_array, weights=weight_array, axis=0, returned=True);
    mean_val    = mean_bp[0];
    mean_weight = mean_bp[1];
    mean_weight = mean_weight.real;
    mean_val[np.isnan(mean_weight)] = 0;
    mean_weight[np.isnan(mean_weight)] = 1e20;
    
    # Solve for antenna bandpass gains                                             
    # Vector that contains real and imaginary gain components for all signal paths
    
    full_params = np.zeros(2 * len(inputs))
    err_params = np.zeros(2 * len(inputs))
    # Indices of gain parameters that will be optimised                                                                             
    params_to_fit = range(len(full_params))
    ref_input_index = inputs.index(ref_ant + opts.pol)
    # Don't fit the imaginary component of the gain on the reference signal pat    h (this is assumed to be zero)                   
    params_to_fit.pop(2 * ref_input_index + 1)
    initial_gains = np.tile([1., 0.], len(inputs))[params_to_fit]
    
    
    gainsol = np.zeros( (mean_val.shape[0], num_ants), dtype=np.complex64 );
    gainerr = np.zeros( (mean_val.shape[0], num_ants), dtype=np.complex64 );
    # not sure the errors are right, I just need *some* estimate
    for n in xrange(mean_val.shape[0]):
        vis = mean_val[n,:];
        var = mean_weight[n,:];
        fitter = scape.fitting.NonLinearLeastSquaresFit(apply_gains, initial_gains);
        fitter.fit(input_pairs, np.vstack((vis.real, vis.imag)), np.vstack( (1/np.sqrt(var.real), 1/np.sqrt(var.real)) ) );
        full_params[params_to_fit] = fitter.params * np.sign(fitter.params[2*ref_input_index])
        err_params[params_to_fit]  = fitter.cov_params * np.sign(fitter.cov_params[2*ref_input_index])
        gainsol[n,:] = full_params.view(np.complex128)
        gainerr[n,:] = err_params.view(np.complex128);

   
    # when correcting for the bandpass, all we want to do is correct the variations ACROSS the band
    # so we need to divide our correction by the mean amplitude across the band to ensure we're
    # not changing the value of the data at all.
    if normalize==True:
        bp_mean     = np.abs(gainsol);
        bp_mean[bp_mean==0] = np.nan;
        bp_mean = imred.nanmean(bp_mean, axis=0);
        bp_mean = bp_mean.reshape( (1, num_ants) );
        bp_mean = np.kron( np.ones( (gainsol.shape[0], 1)), bp_mean);
        bp_to_apply = gainsol/bp_mean;
        return (gainsol, bp_to_apply);
    else:
        return gainsol;


def fixed_delay_calc(data, gain_phase):
    """ Does a quick line fit to calculate the fixed delay offset """

    num_ants = gain_phase.shape[1];
    # fit line to phases to determine fixed delay.
    slopes = np.zeros(num_ants);
    inters = np.zeros(num_ants);
    for n in range(num_ants):
        y = gain_phase[:,n];
        ind    = np.isnan(y) == False;
        x = data.channel_freqs[ind];
        y = y[ind];
        xx     = pow(x,2);
        sum_x  = np.nansum(x);
        sum_xx = np.nansum(xx);
        sum_y  = np.nansum(y);
        xy     = x*y;
        sum_xy = np.nansum(xy);
        num    = len(y);
        inters[n]=(-sum_x*sum_xy+sum_xx*sum_y)/(num*sum_xx-sum_x*sum_x)
        slopes[n]=(-sum_x*sum_y+num*sum_xy)/(num*sum_xx-sum_x*sum_x)
    
    # slope is in radian/Hz, convert to nanosecond
    delays = slopes*1e9/(2*np.pi);
    
    return delays;


def fixed_delay_calc2(freqs, gain_phase):
    """ Does a quick line fit to calculate the fixed delay offset """

    num_ants = gain_phase.shape[1];
    # fit line to phases to determine fixed delay.
    slopes = np.zeros(num_ants);
    inters = np.zeros(num_ants);
    for n in range(num_ants):
        y = gain_phase[:,n];
        ind    = np.isnan(y) == False;
        x = freqs[ind];
        y = y[ind];
        xx     = pow(x,2);
        sum_x  = np.nansum(x);
        sum_xx = np.nansum(xx);
        sum_y  = np.nansum(y);
        xy     = x*y;
        sum_xy = np.nansum(xy);
        num    = len(y);
        inters[n]=(-sum_x*sum_xy+sum_xx*sum_y)/(num*sum_xx-sum_x*sum_x)
        slopes[n]=(-sum_x*sum_y+num*sum_xy)/(num*sum_xx-sum_x*sum_x)
    
    # slope is in radian/Hz, convert to nanosecond
    delays = slopes*1e9/(2*np.pi);
    
    return delays;


def calc_delay(freqs, gain_phase, gain_err):
    """ Does a quick line fit to calculate the fixed delay offset """

    # the errors are not correct on this.
    num_calcs = gain_phase.shape[1];
    # fit line to phases to determine fixed delay.
    inters     = np.zeros(num_calcs);
    slopes     = np.zeros(num_calcs);
    slope_errs = np.zeros(num_calcs);
    for n in range(num_calcs):
        y      = gain_phase[:,n];
        vary   = np.abs(gain_err[:,n]); 
        ind    = np.isnan(y) == False;
        x = freqs[ind];
        y = y[ind];
        vary   = vary[ind];
        xx     = pow(x,2);
        sum_x  = np.nansum(x);
        sum_xx = np.nansum(xx);
        sum_y  = np.nansum(y);
        xy     = x*y;
        sum_xy = np.nansum(xy);
        num    = len(y);
        ess    = np.nansum(1/vary);
        denom  = (num*sum_xx-sum_x*sum_x);
        slope_errs[n] = ess/denom;
        inters[n]=(-sum_x*sum_xy+sum_xx*sum_y)/denom;
        slopes[n]=(-sum_x*sum_y+num*sum_xy)/denom;
    
    # slope is in radian/Hz, convert to nanosecond
    delays = slopes*1e9/(2*np.pi);
    slope_errs = slope_errs*1e9/(2*np.pi);
    
    return (delays, slope_errs);


# for our purposes, flag out the RFI channels.
#mean_var = imred.nanmean(mean_weight).mean();
#var_lim  = 3*mean_var;
#(x_bad, y_bad)  = (mean_weight > var_lim).nonzero();  # first entry is x-axis, second is y-axis
#chan_bad = unique(x_bad);
#for n in chan_bad:
#    mean_val[n, :] = np.nan;
#    mean_weight[n,:] = np.inf;

# Next I want to calculate the actual bandpass on the source
# for this I want to use the existing solver.


def bin_scans(vis_list, var_list, time_list):

    """ Function to bin the scans (in time), within a given list. 
        It will basically just take the average of all the times in a 
        for each list entry 
    """

    mean_vis, mean_var, mean_time = [],[],[];
    for n in range(len(vis_list)):
        (this_mean, this_var) = np.average(vis_list[n], axis=0, weights=var_list[n], returned=True);
        this_time = np.mean(time_list[n]);
        mean_vis.append(this_mean);
        mean_time.append(this_time);
        mean_var.append(this_var);

    return (mean_vis, mean_var, mean_time);



def bin_scans_noweight(vis_list, time_list):
    """ Function to bin the scans (in time), within a given list. 
        It will basically just take the average of all the times in a 
        for each list entry 
    """
    mean_vis, mean_var, mean_time = [],[],[];
    for n in range(len(vis_list)):
        this_mean = imred.nanmean(vis_list[n], axis=0);
        this_var  = imred.nanvar(vis_list[n], axis=0);
        this_time = np.mean(time_list[n]);
        mean_var.append(this_var);
        mean_vis.append(this_mean);
        mean_time.append(this_time);
    return (mean_vis, mean_var, mean_time);

