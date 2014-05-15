#!/usr/bin/python
import scipy.stats as stat
import numpy as np
import sys

Sample2 = np.genfromtxt(sys.argv[2])
fname = "{0:6.4e}.fp".format(float(sys.argv[1]))
Sample1 = np.genfromtxt(fname)

Sample1 = sorted(Sample1[:,1])
Sample2 = sorted(Sample2[:,1])

threshold = Sample1[int(0.95*len(Sample1))]
count = 0
for s in Sample2:
    if s > threshold:
        count += 1

print float(count)/len(Sample2)

