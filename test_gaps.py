import numpy as np
import katarchive
import katpoint
import matplotlib.pyplot as plt
import katfile

year = 2012 
p = []
month = ['0'+str(s) for s in range(1,10)]
month = month + [str(s) for s in range(10,13)]
end = [31,28,31,30,31,30,31,31,30,31,30,31]
end = [str(e) for e in end]
fi = katarchive.search_archive(startdate='01/01/2012',enddate='31/12/2012')
N = len(fi)
for i in range(N-1):
	start = katpoint.Timestamp(fi[i+1].metadata.StartTime[:19])
	end = start + fi[i+1].metadata.Duration
	start_2 = katpoint.Timestamp(fi[i].metadata.StartTime[:19])
	diff = start_2 - end 
	p.append(diff)
plt.figure()
plt.plot(p)


