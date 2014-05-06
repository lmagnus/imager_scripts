#!/usr/bin/python
'''
This script reads the archive and returns the meta data for the files from the month of interest
It converts the start time string to a MJD for simplicity of arithmetic when determining if the
last file runs into the next day.

April 2013
L Magnus
'''

import numpy as np
import matplotlib.pyplot as plt
import katarchive
import calendar
import katpoint
from matplotlib import mpl
import optparse


parser = optparse.OptionParser(usage="%observation_stats.py -m MONTH -y YEAR",
	description="Creats a observation summary report. If ot options are given then the current month is reported")
parser.add_option("-y", "--year", dest="year", type="int", help="Year as an integer")
parser.add_option("-m", "--month", dest="month", type="int", help="Month of the year as an integer")
parser.set_defaults(year = calendar.datetime.date.today().year, month = calendar.datetime.date.today().month)
(opts, args) = parser.parse_args()

month = opts.month
year = opts.year
mdays = calendar.monthrange(year,month)[1]

if month == 1:
	l_year = year - 1
	last_m = 12
else:
	l_year = year
	last_m = month - 1

last_mdays = calendar.monthrange(l_year,last_m)[1]

c = ['r','r','r','r','r','k','b','g']
percent = []
days = range(1,mdays + 1)

# read the last file in the previous month and see if it ran over to the first day
# if it did run over then calculate by how much
f = katarchive.search_archive(startdate=str(last_mdays)+'/'+str(last_m)+'/'+str(l_year),enddate=str(last_mdays)+'/'+str(last_m)+'/'+str(l_year))
if len(f)>0:
	ts = katpoint.Timestamp(int(f[0].metadata.Filename[:-3])).to_mjd()
	du = f[0].metadata.Duration
	temp = np.floor(ts) + 1 - 7200.0/86400
	start = ((ts + du/86400) - temp) * 86400
	l = len(f[0].metadata.Antennas) # how many antennas were used in the observation
else:
	start = 0
	l = 0

# cycle through all the days in the month
for d in days:
	col = [c[l]]
	# return the file statistics for the day of interest from the archive
	f = katarchive.search_archive(startdate=str(d)+'/'+str(month)+'/'+str(year),enddate=str(d)+'/'+str(month)+'/'+str(year))
	if len(f) == 0: #if there are no files recorded on that day
		percent.append(0)
		plt.figure(2)
		plt.subplot(mdays,1,d)
		ax = plt.gca()
		ax.set_xticks([])
		ax.set_yticks([])
		plt.ylabel(str(d)+': 0%',rotation='horizontal')
		continue
	t = start if start > 0 else 0 # the overlap from the previous day
	yranges=[(0,start/86400)] # start time and duration in fractions of a day
	# process all but the last file ... the archive returns the files in reverse order !
	for m in f[1:]:
		ts = katpoint.Timestamp(int(m.metadata.Filename[:-3])).to_mjd() + 7200.0/86400
		du = m.metadata.Duration/86400.
		l = len(m.metadata.Antennas)
		col.append(c[l])
		yranges.append((ts%1,du)) # mod the MJD with 1 to get the fractional part
		t = t + du * 86400
	# process the last file to see if it runs over into the next day
	ts = katpoint.Timestamp(int(f[0].metadata.Filename[:-3])).to_mjd()
	du = f[0].metadata.Duration
	temp = np.floor(ts) + 1 - 7200.0/86400
	start = ((ts + du/86400) - temp) * 86400
	p = t + du if start < 0 else t + du - start 
	s = du if start < 0 else du - start
	yranges.append(((ts+7200.0/86400)%1,s/86400.)) # a list of start times and durations in fractions of a day
	l = len(f[0].metadata.Antennas)
	col.append(c[l])
	percent.append(100*p/86400)
	# plot the data
	plt.figure(2)
	plt.subplot(mdays,1,d)
	plt.subplots_adjust(wspace=0., hspace=0.)
	ax = plt.gca()
	ax.broken_barh(yranges,(0,1), facecolors=col) # plot the data as bars 
	ax.set_xticks([])
	ax.set_yticks([])
	plt.xlim(0,1)
	plt.ylim(0,1)
	plt.ylabel(str(d)+': '+str(np.rint(100*p/86400))+'%',rotation='horizontal')
plt.subplot(mdays,1,mdays)
plt.xlabel('Time/[SAST]')
plt.xticks(np.arange(0,1,0.25),['0','6','12','18','24'])
plt.subplot(mdays,1,1)
plt.title('System usage for '+calendar.month_name[month]+' in '+str(year)+' ... Average usage: '+str(np.floor(np.sum(percent)/len(percent)))+'%')
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes([0.85, 0.1, 0.05, 0.8])
cmap = mpl.colors.ListedColormap(['r', 'k', 'b', 'g'])
cmap.set_over('0.25')
cmap.set_under('0.75')
bounds = [1, 4, 5, 6, 7]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
cb2 = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
					norm=norm,
					# to use 'extend', you must
					# specify two extra boundaries:
					boundaries=bounds,
					#extend='both',
					ticks=bounds, # optional
					spacing='proportional',
					orientation='vertical')
cb2.set_label('Number of ants used')
plt.suptitle('Day:%',x=0.1,y=0.92)
plt.show()
