#!/usr/bin/python

import katarchive
import katfile
import os

files =  katarchive.search_archive(startdate='1/1/2012',description="Health")

for file in files:
    args = ' --filename ' + file.metadata.Filename 
    os.system('python /home/kat/comm/scripts/test_bandpass.py'+args)


