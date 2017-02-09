#!/usr/bin/python3
# Filename: plot.py

import mmap
import os, sys

import string
import math

#from matplotlib import *
#from numpy import *
#from matplotlib.pylab import *
#import matplotlib.pyplot
import os.path
import numpy as np

inputfile = "input"
outputfile = "output"

display = 0
mode = 'a'

argc = len(sys.argv)
argv = sys.argv
print "--ARGUMENTS : ", argv, " --"
if argc > 1 :
	inputfile = argv[1]
	print "INPUT file is : ", inputfile 

if argc > 2 :
	outputfile = argv[2]
	print "OUTPUT file is : ", outputfile	

if argc > 3 :
	display = int(argv[3])
	
if display == 0 :
	print "NO DISPLAY"

if argc > 4 :
	mode = str(argv[4])	
	
	
	
#/--------------------------------------------------------/
#/--------------------------------------------------------/
#/--------------------------------------------------------/
#/--------------------------------------------------------/



def writeInFile(filepath, a, strname) :
	f = open(filepath, mode)
	f.write("\t---"+strname+"---"+"\n\n")
	shape = a.shape
	dim1 = shape[0]
	dim2 = shape[1]
	
	for i in range(dim1) :
		f.write("\t")
		for j in range(dim2) :
			aijstr = "%f" % a[i,j]
			f.write( aijstr)
			f.write( "\t" )	
		f.write("\n")
	f.write("\n")
	
	print "SAVING IN FILE : --", filepath, "-- : done."
	f.close
	

#/--------------------------------------------------------/
#/--------------------------------------------------------/
#/--------------------------------------------------------/
#/--------------------------------------------------------/	

	
with open(inputfile,'r') as datasetlist:
	data = np.loadtxt(datasetlist)
	mean = data[0]
	mean = np.mat( data[0] ).transpose()
	
	for i in range(len(data)) :
		y_sample = np.mat( data[i] )
		#print "CURRENT SAMPLE : "
		#print y_sample
		#mean = mean + y_sample
		mean = [ mean, y_sample.transpose() ]
	#mean -= data[0]
	#mean = mean / (len(data)-1)
	if display == 1 :
		print "END : MEAN : " 
		print mean	
	writeInFile(outputfile, mean.transpose(), " values")
	
