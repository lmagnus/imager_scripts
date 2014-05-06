import katfile
import katarchive
import csv
import numpy as np
import katpoint 

f = katarchive.get_archived_products('1355547913.h5')

h5 = katfile.open(f[0])

d = []
h5.select(targets='Vir A',scans='scan')
for cs in h5.compscans():
	d.append(['1355547913',' Vir A',' '+katpoint.Timestamp(np.median(h5.timestamps)).to_string()])

#d = np.vstack(d)

ants = h5.ants

for ant in ants:
	p = []
	ofile = open(ant.name+'_keep.csv','wb')
	w = csv.writer(ofile)
	ofile.write('# antenna = '+ant.description+'\n')
	p.append(['dataset', ' target', ' timestamp_ut'])
	w.writerows(p)
	w.writerows(d)
	ofile.close()


