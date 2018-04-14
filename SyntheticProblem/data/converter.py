import os
import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab




def main():

	data = np.loadtxt("synth_core.txt")
	core_depths = data[:,0]
	print 'core_depths', core_depths
	print 'core_depths.size',core_depths.size
	core_data = data[:,1]
	print 'core_data',core_data
	pred_core = np.zeros((core_depths.size,4))

	for n in range(0,core_depths.size):
			if core_data[n] == 0.571:
				pred_core[n,3] = 1 
			if core_data[n] == 0.429:
				pred_core[n,2] = 1 
			if core_data[n] == 0.286:
				pred_core[n,1] = 1 
			if core_data[n] == 0.143:
				pred_core[n,0] = 1 

	print pred_core

	pred_core_ = str(pred_core)

	with file('synth_core_bi.txt','w') as outfile:
		outfile.write ('')


	with file('synth_core_bi.txt','a') as outfile:
		for x in range(0,core_depths.size):
			for y in range(0,4):
				val = str(int(pred_core[x,y]))
				outfile.write(val)
				outfile.write(' ')
			outfile.write('\n')

if __name__ == "__main__": main()
