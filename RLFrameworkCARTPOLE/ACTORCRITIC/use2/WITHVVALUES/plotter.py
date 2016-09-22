import matplotlib.pyplot as plt
import time
import os.path
import mmap
import os, sys

import string
import math

import threading
from multiprocessing import Process

inputfile = "data.txt"

def plotDATA() :
	data = loadData()
	plt.plot(data)
	plt.show()

def plot(d,mu,sigma) :
	#d = [ e in d if abs(e-mu) < sigma ]
	d1 = []
	preve = d[2]
	number = 1
	for e in d :
		#if abs(e-mu) > sigma/2 :
		#	e = preve

		d1.insert(1,e)
		#number += 1
		#preve = (number-1)/number*preve+e/number
		preve = e;
		#print preve
			
	plt.plot(d1)
	plt.show()


def loadData() :
	f=open(inputfile,'r')
	d = f.readlines()
	f.close()
	data = [float(e.replace(',','').replace(' ','')) for e in d if d]
	return data
	
def computeStats(d) :
	n = len(d)
	print "LIST SIZE : ", n
	mean = d[1]/n

	
	for el in d :
		mean += el/n
	mean -= d[1]/n
	print "MEAN = ", mean
	
	var = 0
	for el in d :
		var += (el-mean)*(el-mean)/(n-1)
	sigma = math.sqrt(var)
	print "SQD = ", sigma
	
	if False :
		prevmean = mean
		mean = 0
		prevd = d[1]
		for i in range(len(d)) :
			if abs(d[i]-prevmean) > sigma/2 :
				d[i] = prevd
			mean += d[i]/n
			prevd = d[i]
		mean -= d[1]/n
		print "MEAN 2 = ", mean
	
		var = 0
		for el in d :
			var += (el-mean)*(el-mean)/(n-1)
		sigma = math.sqrt(var)
		print "SQD 2 = ", sigma
	
		prevmean = mean
		mean = 0
		prevd = d[1]
		for i in range(len(d)) :
			if abs(d[i]-prevmean) > sigma/2 :
				d[i] = prevd
			mean += d[i]/n
			prevd = d[i]
		mean -= d[1]/n
		print "MEAN 3 = ", mean
	
		var = 0
		for el in d :
			var += (el-mean)*(el-mean)/(n-1)
		print "SQD 3 = ", math.sqrt(var)
	
	
	return mean,math.sqrt(var)
	
	
#while True:
	#t = threading.Thread(target=plot)
	
argc = len(sys.argv)
argv = sys.argv
print "--ARGUMENTS : ", argv, " --"
if argc > 1 :
	inputfile = argv[1]
	print "INPUT file is : ", inputfile 


#data = loadData()
#mean,std = computeStats(data)
#plot(data,mean,std)
plotDATA()

#t = Process(target=plot)
#t.start()
#	time.sleep(10)
#	t.terminate()

#data = loadData()
#p = Process(target=plot)
#p.start()

#while True:
#	data = loadData()
#	time.sleep(5)
	