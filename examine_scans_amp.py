#!/usr/bin/python -u
"""
Usage examine_scans_amp.py <path/to/filename>

This script will load up an H5 file and then plot the visibility magnitudes for all the single antenna
polarisations found in the file.

The user can then click on a plotted line element in the plot to display the spectra for that point.
"""

import numpy as np
import matplotlib.pyplot as plt 
import optparse
import pylab

import scape
import katpoint
import h5py
import time

parser = optparse.OptionParser(usage="%prog [options] <directories or files>",description=" ")
(opts, args) = parser.parse_args()

if len(args) ==0:
    print 'Please specify an h5 file to load.'
    sys.exit(1)

"""
Load the file and process it using the H5PY module to extract the anennas and 
correlator dump rates that have been stored
"""
filename = args[0]
f = h5py.File(filename,'r')
ants = f['Antennas'].keys()
dump_rate = f['Correlator'].attrs['dump_rate_hz']
samp_period = 1.0 / dump_rate 
f.close()

print 'Correlator sample period [s]: '+str(samp_period)



"""
Load the satellite TLE elements into a katpoint archive
"""
sats = katpoint.Catalogue(add_specials=False)
sats.add_tle(file('/Users/lindsay/svn/katuilib/katuilib/test/conf/catalogues/gps-ops.txt'))
sats.add_tle(file('/Users/lindsay/svn/katuilib/katuilib/test/conf/catalogues/geo.txt'))
#sats.add_tle(file('/Users/lindsay/svn/katuilib/katuilib/test/conf/catalogues/glo-ops.txt'))


"""
Set up the plotting variables etc
"""
global fig
fig = plt.figure(1)
cl = ['b','g','r','c','m','y','k','b--']
cnt = 0


"""
Set up the accumilating variables that will hold the magnitudes and the associated spectra
""" 
lines = []
ant_spec = []
all_az = []
all_el = []
pol = []

"""
Cycle through all the antennas in the H5 file
"""
for curr_ant in ants:
	"""
	Create the baseline option for DataSet
	"""
	ant_num = curr_ant[7]
	b_line = 'A'+ant_num+'A'+ant_num
	d = scape.DataSet(filename, baseline=b_line)
	"""
	Extract the timestamps and az el values for each antenna
	"""
	t_ime = np.hstack([s.timestamps for s in d.scans]) - d.scans[0].timestamps[0]
	az = np.hstack([s.pointing['az'] for s in d.scans])
	all_az.append(az)
	el = np.hstack([s.pointing['el'] for s in d.scans])
	all_el.append(el)
	"""
	Verical polarization data
	"""
	vv_spec = np.vstack([s.pol('VV') for s in d.scans])
	vv_amp = np.median(vv_spec,axis=1) 	
	if sum(vv_amp) > 0.0:
		ll,=plt.plot(t_ime,vv_amp,'.-',label = curr_ant+' VV',picker = 5)
		lines.append(ll.get_label())
		ant_spec.append(vv_spec)
		pol.append('VV')
		cnt += 1
	"""
	Horizontal polariszation data
	"""
	hh_spec = np.vstack([s.pol('HH') for s in d.scans])
	hh_amp = np.median(hh_spec,axis=1) 	
	if sum(hh_amp) > 0.0:
		ll,=plt.plot(t_ime,hh_amp,'.-',label = curr_ant+' HH',picker = 5)
		lines.append(ll.get_label())
		ant_spec.append(hh_spec)
		pol.append('HH')
		cnt += 1
"""
Load the frequency values for the spectra as well as the text time stamps
"""
freqs = d.freqs
time_names = [katpoint.Timestamp(ts+d.scans[0].timestamps[0]) for ts in t_ime]

"""
Create scan name and numbers for plotting
"""
name = []
scan_number = []
for num,s in enumerate(d.scans):
	nm = s.compscan.target.name+" while in: "+s.label 
	for i in s.data:
		name.append(nm)
		scan_number.append(num)
"""
Make the visibility plot
"""
plt.legend()
plt.xlabel('Time in seconds from: '+time_names[0].to_string())
plt.ylabel('Amplitude')
plt.title(filename)
global mark
global mark_t
global mark_b
global ind
global bottom_ind
global top_ind

"""
setup the index limits and lines
"""
ind = int(len(t_ime)/2)
bottom_ind = int(len(t_ime)/4)
top_ind = ind + bottom_ind
mark = plt.axvline(x=ind)
mark_b = plt.axvline(x=bottom_ind,color='g')
mark_t = plt.axvline(x=top_ind,color='r')

def onpick(event):
	"""
	Function to action the pick event on the the visibility plot
	"""
	global ind
	global fig
	global mark
	"""
	Determine which line was selected
	"""
	label = event.artist.get_label()
	ant_ind = lines.index(label) 

	N = len(event.ind)
	if not N: return True
	"""
	Determine the index value of the point selected
	"""
	ind = event.mouseevent.xdata
	ind = (t_ime>ind).nonzero()[0][0]-1
		
	spec_plot(ind)
	return True

def key_handler(event):
	"""
	choose a range from the current visibility plot to display / save
	"""
	global bottom_ind
	global top_ind
	global ind
	global mark_b
	global mark_t
	global fig
	if event.key in ['B', 'b']:
		bottom_ind = ind
		mark_b.set_xdata(([t_ime[bottom_ind],t_ime[bottom_ind]]))
		fig.show()
	if event.key in ['T', 't']:
		top_ind = ind
		mark_t.set_xdata(([t_ime[top_ind],t_ime[top_ind]]))
		fig.show()
	if event.key in ['S', 's']:
		fig4 = plt.figure(4)
		fig4.clear()
		data = [spec[bottom_ind:top_ind,128:-128] for spec in ant_spec]
		for i,sp in enumerate(data):
			ax = fig4.add_subplot(4,1,i+1)
			pl_spec = np.log10(np.abs(sp))
			#pl_spec = np.abs(sp)
			#pl_spec -= np.average(pl_spec,axis=0)
			ax.pcolor(freqs[128:-128],t_ime[bottom_ind:top_ind],pl_spec)
			#scape.plot_xyz(d.scans[scan_number[ind]],'time','freq','amp',pol=pol[)
			#scape.plots_basic.plot_segments(t_ime[bottom_ind:top_ind],freqs[128:-128],pl_spec)
			fig4.show()
	if event.key in ['P','p']:
		fig5 = plt.figure(5)
		fig5.clear()
		i = 0
		for az,el in zip(all_az,all_el):
			ax = fig5.add_subplot(4,1,i+1)
			ax.plot(t_ime[bottom_ind:top_ind],az[bottom_ind:top_ind])
			ax.plot(t_ime[bottom_ind:top_ind],el[bottom_ind:top_ind])
			fig5.show()
			i+=1
		
	if event.key in ['U', 'u']:
		ind += 1
		spec_plot(ind)
	if event.key in ['D', 'd']:
		ind -= 1
		spec_plot(ind)
	if event.key in ['A','a']:
		log = open(filename+'.csv','a')
		for i,ant in enumerate(ants):
			for j in range(top_ind - bottom_ind):
				log.write('%s, %s, %s, %f, %f\n' % (ant,name[bottom_ind + j],time_names[bottom_ind + j].to_string(),all_az[i][bottom_ind + j],all_el[i][bottom_ind + j]))
		log.close()
	return True
	
def spec_plot(f_ind):
	"""
	Function to plot the spectra of the selected point from the visibility plot
	"""
	global fig
	global mark
	"""
	Show which point is being plotted
	"""
	mark.set_xdata(([t_ime[f_ind],t_ime[f_ind]]))
	fig.show()
	"""
	Plot the spectra
	"""		
	fig2 = plt.figure(2)
	fig2.clear()
	for i,spec in enumerate(ant_spec):
		ax = fig2.add_subplot(4,1,i+1)
		"""
		only plot from [1:-1] to exclude the ADC dc value
		"""
		ax.plot(freqs[1:-1],spec[f_ind,1:-1])
		ax.set_title(lines[i]+" at: "+str(t_ime[f_ind])+"s - "+time_names[f_ind].to_string())
		if i == 3: ax.set_xlabel(str(np.rad2deg(all_az[0][f_ind]))+ ':'+str(np.rad2deg(all_el[0][f_ind])))
	fig2.show()
	"""
	Determine which satellite in the katpoint catelogue is the closest to where the antennas were pointing an print it to the screen
	"""
	sep = sats.closest_to(katpoint.construct_azel_target(all_az[0][f_ind],all_el[0][f_ind]),timestamp=time_names[f_ind],antenna = d.antenna)
	print "%s. Closest to: %s. Azel: %s. Seperated by: %s degrees" % (name[f_ind], sep[0].name,str(np.rad2deg(all_az[0][f_ind]))+ ':'+str(np.rad2deg(all_el[0][f_ind])),sep[1])
	return True
	
	

spec_plot(ind)
fig.canvas.mpl_connect('pick_event', onpick)
fig.canvas.mpl_connect('key_press_event', key_handler)
plt.suptitle('b:bottom set, t:top set, S: spectrum, p:pointing, u:up, d:down, a:append text')
#plt.show()
