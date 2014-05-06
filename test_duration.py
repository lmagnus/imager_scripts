import numpy as np
import matplotlib.pyplot as plt
import katarchive
import calendar
import matplotlib.pyplot as plt
import katfile

year = 2012 
p = []
month = ['0'+str(s) for s in range(1,10)]
month = month + [str(s) for s in range(10,13)]
end = [31,28,31,30,31,30,31,31,30,31,30,31]
end = [str(e) for e in end]
for i in range(12): 
	f = katarchive.search_archive(startdate='01/'+month[i]+'/2012',enddate=end[i]+'/'+month[i]+'/2012')
	d = np.asarray([m.metadata.Duration for m in f])
	p.append(d.sum()/(int(end[i])*24*3600))
plt.figure()
plt.bar(range(12),p)


