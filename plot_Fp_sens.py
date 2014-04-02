#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
import glob
import sys

def getint(name):
    basename = name.partition('.')
    basename = basename[0].split('/')
    return int(basename[1])

direc = sys.argv[1]

#obtain all frequencies
freqs = np.genfromtxt('{0:s}/freqlist'.format(direc))
#get filelist for output
result_files = glob.glob("{0:s}/*.out".format(direc))
result_files = sorted(result_files,key=getint)
print result_files
nresults = len(result_files)
results_freqs = []
results_amps = []
#loop through files and see if we have a result already
for i in range(nresults):
    infile = open(result_files[i],"r")
    print getint(result_files[i])
    for line in infile:
        fields = line.split()
        #print fields
        if fields[2] == "FINISHED":
            results_freqs.append(freqs[i])
            results_amps.append(float(fields[0]))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(results_freqs,results_amps)
plt.show()
