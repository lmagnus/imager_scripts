 # script to scan through all point source scans and create a target list.
import katfile
import katarchive

#locate the files 
#myfiles = katarchive.get_katfiles(interactive = False,description = 'point source scan')
myfiles = katarchive.list_katfiles(start='30/7/2011',end='1/8/2011',description='point source scan')
list_len = len(myfiles)
log_file = open('point_source_scan_log_test.txt','w')

for i,file in enumerate(myfiles):
	percent = int(float(i)/list_len * 100.0)
	if percent % 10 == 0: 
		print "Percent = " + str(percent)
	h5 = katfile.open(file)
	try:
		for scan_index, compscan_index, state, target in h5.scans():
			if state != 'scan':
				continue
			timestamps = h5.timestamps()
			time = timestamps[len(timestamps) // 2] 
			log_file.write("%s, %s, %f\n" % (h5.version,target.description,time))
	except KeyError as error:
		print 'Failed to read scans from File: ',file,' with Key Error:',error
	except ValueError as error:
		print 'Failed to read scans from File: ',file,' with Value Error:',error
	except:
		print 'Some other error'
		pass


