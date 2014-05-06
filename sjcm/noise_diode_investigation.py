# import the libraries
import katarchive
import katfile
import katpoint
import numpy as np
from gaincal import NoiseDiodeModel
from hdf5 import remove_duplicates
import matplotlib.pyplot as plt

num_ants = 7;
freq_chans = 800;
start_freq = 100;

#read in the data
f = katarchive.get_archived_product('1313842550.h5')
data = katfile.open(f[0]);
datatime = data.timestamps[:];

rows = len(datatime);
columns = num_ants;
dimensions = (rows,columns)
onInd = np.zeros(dimensions);
offInd = np.zeros(dimensions);


# first we sort out a boolean array for when the source was on
sensors_group = data.file['MetaData/Sensors'];

pedNums =  [ 'ped1', 'ped2', 'ped3', 'ped4', 'ped5', 'ped6', 'ped7'];

j=1;
 
numOn = 0;
for i in range(7):
  noise_value     = sensors_group['Pedestals'][pedNums[i]]['rfe3.rfe15.noise.coupler.on']['value'];
  noise_timestamp = sensors_group['Pedestals'][pedNums[i]]['rfe3.rfe15.noise.coupler.on']['timestamp'];
  for j in range(len(noise_timestamp)):
     corresponding_time = noise_timestamp[j];
     if noise_value[j] == '0':
        onInd[datatime>corresponding_time-2,i] = 0;
     elif noise_value[j] == '1':
        onInd[datatime>corresponding_time+2,i] = 1;
        numOn = numOn+1;
        offInd[datatime > corresponding_time+15,i] = 1;
        offInd[datatime > corresponding_time+25,i] = 0;

# let's determine the time when they were all on
onIndVec = onInd.mean(axis=1) == 1;
offIndVec = offInd.mean(axis=1) == 1;

# next we find out the actual index when things go on and off
# first the on.
realOnStart = np.zeros( (numOn) );
realOnEnd = np.zeros( (numOn) );
fon = onIndVec.ravel().nonzero();
fon = fon[0];
index = 0;
for i in range(len(fon)):
   if i == 0:
      realOnStart[index] = fon[i];
      index = index + 1;
   else:
      if fon[i] - fon[i-1] != 1:
         realOnStart[index] = fon[i];
         realOnEnd[index-1] = fon[i-1]; 
         index = index + 1;


realOnEnd[index-1] = fon[i];
realOnStart = realOnStart[1:index];
realOnEnd   = realOnEnd[1:index];

realOffStart = np.zeros( (numOn) );
realOffEnd = np.zeros( (numOn) );
foff = offIndVec.ravel().nonzero();
foff = foff[0];
index = 0;
for i in range(len(foff)):
   if i == 0:
      realOffStart[index] = foff[i];
      index = index + 1;
   else:
      if foff[i] - foff[i-1] != 1:
         realOffStart[index] = foff[i];
         realOffEnd[index-1] = foff[i-1]; 
         index = index + 1;


realOffEnd[index-1] = foff[i];
realOffStart = realOffStart[1:index];
realOffEnd   = realOffEnd[1:index];

num_samples = len(realOffEnd);

# and finally, now we get our power measurements!!!!
antVals = ['ant1H', 'ant2H', 'ant3H', 'ant4H', 'ant5H', 'ant6H', 'ant7H'];

Pon      = np.zeros((num_ants, num_samples,freq_chans));
Poff     = np.zeros((num_ants, num_samples,freq_chans));
Varon    = np.zeros((num_ants, num_samples,freq_chans));
Varoff   = np.zeros((num_ants, num_samples,freq_chans));
timeOn   = np.zeros((num_samples)); 
timeOff  = np.zeros((num_samples)); 


for i in range(7):
  for j in range(num_samples):
     data.select(ants=antVals[i][:-1],corrprods='auto',pol='H')
     autoOff = data.vis[int(realOffStart[j]):int(realOffEnd[j]),start_freq:start_freq+freq_chans];
     autoOn  = data.vis[int(realOnStart[j]):int(realOnEnd[j]),start_freq:start_freq+freq_chans];
     Pon[i,j,:] = autoOn[:,:,0].mean(axis=0);
     Poff[i,j,:] = autoOff[:,:,0].mean(axis=0);
     Varon[i,j,:] = autoOn[:,:,0].var(axis=0);
     Varoff[i,j,:] = autoOff[:,:,0].var(axis=0);
     aa = data.timestamps[int(realOffStart[j]):int(realOffEnd[j])];
     timeOff[j] = aa.mean();
     aa = data.timestamps[int(realOnStart[j]):int(realOnEnd[j])];
     timeOn[j] = aa.mean();

dataFreqs = data.channel_freqs[start_freq:start_freq+freq_chans]/1e6;

# now we need the actual noise profiles
antVals = ['ant1', 'ant2', 'ant3', 'ant4', 'ant5', 'ant6', 'ant7'];
ND =  np.zeros( (num_ants, freq_chans) );

# The noise diode values are in a completely random-ass order
for i in range(7):
   thisData = data.file['MetaData']['Configuration']['Antennas'][antVals[i]]['h_coupler_noise_diode_model'];
   nd_h_model = NoiseDiodeModel(thisData[:,0]/1e6, thisData[:,1], **dict(thisData.attrs));
   aa = np.interp(xp=nd_h_model.freq, x=dataFreqs, fp=nd_h_model.temp);
   ND[i,:] = aa;


# next up, calculating the y-factors
y = Pon/Poff;
Vary = Varon/(Poff**2) + (Pon**2)/(Poff**4)*Varoff;


# assuming we have no bright source, we have
noise_rep   = ND.reshape( (num_ants,1,freq_chans) );
noise_rep   = np.kron(np.ones( (1,num_samples,1) ), noise_rep);
tsys = noise_rep/(y-1);

# let's calculate the gains across the band as well.
gain = (Pon - Poff)/(1.38e-23*aa*data.channel_width);
bandgain = gain.mean(axis=2);
gaindB = 10*np.log10(gain);
bandgaindB = 10*np.log10(bandgain);

# an estimate for tau.
temp     = sensors_group['Enviro']['asc.air.temperature']['value'];
temptime = sensors_group['Enviro']['asc.air.temperature']['timestamp'];
tempOn   = np.interp(xp=temptime, fp=temp, x=timeOn)+273.15;
tempOff  = np.interp(xp=temptime, fp=temp, x=timeOff)+273.15;

humid     = sensors_group['Enviro']['asc.air.relative-humidity']['value'];
humidtime = sensors_group['Enviro']['asc.air.relative-humidity']['timestamp'];
humidOn   = np.interp(xp=humidtime, fp=humid, x=timeOn)+273.15;
humidOff  = np.interp(xp=humidtime, fp=humid, x=timeOff)+273.15;

# i have a C function that does this, and I just calculated it for the
# mean values since this effect should be small
tauZenith = 0.0134;

elOn      = np.zeros( (num_ants, num_samples) );
elOff     = np.zeros( (num_ants, num_samples) );


# next we want to know the elevation
for i in range(num_ants):
   ant_sensors = sensors_group['Antennas'][antVals[i]]['pos.actual-scan-elev'];
   original_coord = remove_duplicates(ant_sensors);
   timeVal = original_coord['timestamp'];
   elval   = original_coord['value'];
   elOn[i,:] = np.interp(xp=timeVal, fp=elval, x=timeOn)*np.pi/180;
   elOff[i,:] = np.interp(xp=timeVal, fp=elval, x=timeOn)*np.pi/180;      

opacityOn = tauZenith/np.sin(elOn);
opacityOff = tauZenith/np.sin(elOff);

# RECALCULATE TSYS WITHOUT ASSUMING TSRC IS NEGLIGIBLE
tauZenith = 0.0134;
Tsrc = 0.034*30;
elMean = (elOn + elOff)/2;
elMean = elMean.reshape( (num_ants, num_samples, 1) );
elMean = np.kron(np.ones( (1,1,freq_chans) ), elMean);
tsys_noapprox = ( (1-y)*Tsrc*np.exp(-tauZenith/np.sin(elMean)) + noise_rep )/(y-1);

# and now, let's make some plots
plt.figure(1)
plt.figsize=(12,10)
for i in range(num_ants):
   plt.subplot(3,3,i+1)
   plt.plot(dataFreqs/1000, np.transpose(tsys[i,:,:]));
   if i%3 == 0:
      plt.ylabel('tsys [K]');
   if i in range(4,7):
      plt.xlabel('Frequency [GHz]');
   plt.title(antVals[i]);

plt.figure(2)
plt.figsize=(12,10)
for i in range(num_ants):
   plt.subplot(3,3,i+1)
   plt.plot(dataFreqs/1000, np.transpose(tsys_noapprox[i,:,:]));
   if i%3 == 0:
      plt.ylabel('tsys [K]');
   if i in range(4,7):
      plt.xlabel('Frequency [GHz]');
   plt.title(antVals[i]);

plt.figure(3)
plt.figsize=(12,10)
for i in range(num_ants):
   plt.subplot(3,3,i+1)
   plt.plot(dataFreqs/1000, np.transpose(y[i,:,:]));
   if i%3 == 0:
      plt.ylabel('y-factor');
   if i in range(4,7):
      plt.xlabel('Frequency [GHz]');
   plt.title(antVals[i]);

plt.figure(4)
for i in range(num_ants):
   plt.subplot(3,3,i+1)
   plt.plot(dataFreqs/1000, ND[i,:]);
   if i%3 == 0:
      plt.ylabel('Temp [K]');
   if i in range(4,7):
      plt.xlabel('Frequency [GHz]');
   plt.title(antVals[i]);


# plot as a function of time.
tsysTime = tsys_noapprox.mean(axis=2);


# numpy.average can handle 'weights'

def wtvar(X, W, method = "R"):
  sumW = sum(W)
  if method == "nist":
    xbarwt = sum([w * x for w,x in zip(W, X)])/sumW    
    Np = sum([ 1 if (w != 0) else 0 for w in W])
    D = sumW * (Np-1.0)/Np
    return sum([w * (x - xbarwt)**2 for w,x in zip(W,X)])/D
  else: # default is R 
    sumW2 = sum([w **2 for w in W])
    xbarwt = sum([(w * x)  for (w,x) in zip(W, X)])/sumW
    return sum([(w * (x - xbarwt)**2) for (w,x) in zip(W, X)])* sumW/(sumW**2 - sumW2)
