#!/usr/bin/python
import scipy.stats as stat
import numpy as np
import sys

Sample1 = np.genfromtxt(sys.argv[1])
Sample2 = np.genfromtxt(sys.argv[2])
Sample1 = Sample1[:,1]
Sample2 = Sample2[:,0]

d, p = stat.mstats.ks_twosamp(Sample1,Sample2,alternative='greater')

print p
