

#Main Contributers:   Rohitash Chandra and Ratneel Deo  Email: c.rohitash@gmail.com, deo.ratneel@gmail.com

# Bayeslands II: Parallel tempering for multi-core systems - Badlands


from __future__ import print_function, division
import multiprocessing

import numpy as np
import random
import time
import operator
import math
import cmocean as cmo
from pylab import rcParams


import copy
from copy import deepcopy
import cmocean as cmo
from pylab import rcParams
import collections


from scipy import special


import fnmatch
import shutil
from PIL import Image
from io import StringIO
from cycler import cycler
import os


import matplotlib as mpl
mpl.use('Agg')

import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from scipy.spatial import cKDTree
from scipy import stats 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

import itertools

#nb = 0 
import sys


import plotly
import plotly.plotly as py
from plotly.graph_objs import *
plotly.offline.init_notebook_mode()
from plotly.offline.offline import _plot_html

 
from cycler import cycler
from scipy import stats 


from pyReefCore.model import Model
from pyReefCore import (plotResults, saveParameters)



cmap=plt.cm.Set2
c = cycler('color', cmap(np.linspace(0,1,8)) )
plt.rcParams["axes.prop_cycle"] = c



class ptReplica(multiprocessing.Process):
	def __init__(self,  vis, num_communities, vec_parameters, maxlimits_vec, minlimits_vec , stepratio_vec,   check_likelihood,  swap_interval,  simtime, samples, core_data,  core_depths,  filename, xmlinput,  run_nb, tempr, parameter_queue,event , main_proc,   burn_in):
			 
		#--------------------------------------------------------
		multiprocessing.Process.__init__(self)
		self.processID = tempr      
		self.parameter_queue = parameter_queue
		self.event = event
		self.signal_main = main_proc
		self.temperature = tempr
		self.swap_interval = swap_interval
		#self.lhood = 0

		self.filename = filename
		self.input = xmlinput  


		self.simtime = simtime
		self.samples = samples
		self.run_nb = run_nb 

		self.num_param =  vec_parameters.size

		self.font = 9
		self.width = 1 

		self.vec_parameters =  vec_parameters

		self.maxlimits_vec = maxlimits_vec
		self.minlimits_vec  = minlimits_vec 

		self.core_depths = core_depths 
		self.core_data =  core_data 


		self.stepratio_vec = np.asarray(stepratio_vec)

		#self.realvalues_vec = np.asarray(realvalues_vec) # true values of free parameters for comparision. Note this will not be avialable in real world application		 

		self.check_likehood =  check_likelihood 

		self.communities = num_communities

		self.sedsim = True
		self.flowsim = True 
		self.d_sedprop = float(np.count_nonzero(core_data[:,num_communities]))/core_data.shape[0]
		self.initial_sed = []
		self.initial_flow = []
		self.vis = vis

		 

		self.runninghisto = True  # if you want to have histograms of the chains during runtime in pos_variables folder NB: this has issues in Artimis


		self.burn_in = burn_in

	  


	def run_Model(self, reef, input_vector):
		reef.convert_vector(self.communities, input_vector, self.sedsim, self.flowsim) #model.py
		self.initial_sed, self.initial_flow = reef.load_xml(self.input, self.sedsim, self.flowsim)


		if self.vis[0] == True:
			reef.core.initialSetting(size=(8,2.5), size2=(8,3.5)) # View initial parameters
		reef.run_to_time(self.simtime,showtime=100.)
		if self.vis[1] == True:
			from matplotlib.cm import terrain, plasma
			nbcolors = len(reef.core.coralH)+10
			colors = terrain(np.linspace(0, 1.8, nbcolors))
			nbcolors = len(reef.core.layTime)+3
			colors2 = plasma(np.linspace(0, 1, nbcolors))
			reef.plot.drawCore(lwidth = 3, colsed=colors, coltime = colors2, size=(9,8), font=8, dpi=300)
		output_core = reef.plot.core_timetodepth(self.communities, self.core_depths) #modelPlot.py
		# predicted_core = reef.convert_core(self.communities, output_core, self.core_depths) #model.py
		# return predicted_core
		print ('output_core_shape', output_core.shape )
		return output_core 

		
	def convert_core_format(self, core, communities):
		vec = np.zeros(core.shape[0])
		for n in range(len(vec)):
			idx = np.argmax(core[n,:])# get index,
			vec[n] = idx+1 # +1 so that zero is preserved as 'none'
		return vec

	def diff_score(self, z,intervals):
		same= np.count_nonzero(z)
		same = float(same)/intervals
		diff = 1-same
		print ('diff:', diff)
		return diff*100

	def rmse(self, sim, obs):
		# where there is 1 in the sed column, count
		sed = np.count_nonzero(sim[:,self.communities])
		p_sedprop = (float(sed)/sim.shape[0])
		sedprop = np.absolute((self.d_sedprop - p_sedprop)*0.5)
		rmse =(np.sqrt(((sim - obs) ** 2).mean()))*0.5
		
		return rmse + sedprop


	def likelihood_func(self, reef, core_data, input_v):
		pred_core = self.run_Model(reef, input_v)
		pred_core = pred_core.T
		intervals = pred_core.shape[0]
		z = np.zeros((intervals,self.communities+1))    
		for n in range(intervals):
			idx_data = np.argmax(core_data[n,:])
			idx_model = np.argmax(pred_core[n,:])
			if ((pred_core[n,self.communities] != 1.) and (idx_data == idx_model)): #where sediment !=1 and max proportions are equal:
				z[n,idx_data] = 1
		diff = self.diff_score(z,intervals)
		# rmse = self.rmse(pred_core, core_data)
		
		z = z + 0.1
		z = z/(1+(1+self.communities)*0.1)
		loss = np.log(z)
		# print 'sum of loss:', np.sum(loss)        
		return [np.sum(loss), pred_core, diff]
 
 

	def run(self):

		#note this is a chain that is distributed to many cores. The chain is also known as Replica in Parallel Tempering

		samples = self.samples


		burnsamples = int(samples*self.burn_in)

		count_list = [] 
 

		stepsize_vec = np.zeros(self.vec_parameters.size)
		span = np.abs((self.maxlimits_vec-self.minlimits_vec) )


		for i in range(stepsize_vec.size): # calculate the step size of each of the parameters
			stepsize_vec[i] = self.stepratio_vec[i] * span[i]

		print(stepsize_vec, 'stesize_vec')




		v_proposal = self.vec_parameters # initial values of the parameters to be passed to Blackbox model 
		v_current = v_proposal # to give initial value of the chain
 
 


		reef = Model()

		[likelihood, pred_data, diff_score] = self.likelihood_func(reef, self.core_data, v_proposal)
		   
		pred_core_vec = self.convert_core_format(pred_data, self.communities)
		 

		#---------------------------------------
		#now, create memory to save all the accepted   proposals of rain, erod, etc etc, plus likelihood


		pos_param = np.zeros((samples,v_current.size))   
		pos_param[0,:] = v_proposal # assign first proposal

 
		list_core_pred = np.zeros((samples, pred_core_vec.size)) # list of all accepted (plus repeats ) of pred cores  
		list_core_pred[0,:] = pred_core_vec # assign the first core pred
	 
		print ('\tinitial likelihood:', likelihood, 'and difference score:', diff_score)

		#----------------------------------------------------------------------------
 

		print('\tinitial likelihood:', likelihood)
 

		likeh_list = np.zeros((samples,2)) # one for posterior of likelihood and the other for all proposed likelihood
		likeh_list[0,:] = [-10000, -10000] # to avoid prob in calc of 5th and 95th percentile   later



		count_list.append(0) # just to count number of accepted for each chain (replica)
		accept_list = np.zeros(samples)
		 
	
	 
		start = time.time() 

		num_accepted = 0

		num_div = 0 

		#save

		with file(('%s/description.txt' % (self.filename)),'a') as outfile:
			outfile.write('\n\samples: {0}'.format(self.samples))
			outfile.write('\n\tstepsize_vec: {0}'.format(stepsize_vec))  
			outfile.write('\n\tstep_ratio_vec: {0}'.format(self.stepratio_vec)) 
			outfile.write('\n\tswap interval: {0}'.format(self.swap_interval))   
			outfile.write('\n\tInitial_proposed_vec: {0}'.format(v_proposal))   


		
		for i in range(samples-1):

			print (self.temperature, ' - Sample : ', i)

			# Update by perturbing all the  parameters via "random-walk" sampler and check limits
			v_proposal =  np.random.normal(v_current,stepsize_vec)

			print(stepsize_vec, ' stepsize_vec')

			for j in range(v_current.size): # check for flow and sed
				if v_proposal[j] > self.maxlimits_vec[j]:
					v_proposal[j] = v_current[j]
				elif v_proposal[j] < self.minlimits_vec[j]:
					v_proposal[j] = v_current[j]

			v_proposal[25] = -0.003

			v_proposal[26] = -0.001


			'''for k in range( k = v_current.size-3, v_current.size): # check for GLV parameters (commu super and sub ax ay, malthusian)
				print(k, ' k ')
				if v_proposal[k] > self.maxlimits_vec[k]:
					v_proposal[k] = v_current[k]
				elif v_proposal[k] < self.minlimits_vec[k]:
					v_proposal[k] = v_current[k]'''
 

			print(i, ' is sample', self.maxlimits_vec, ' maxlimits_vec')  


			print(i, ' is sample', self.minlimits_vec, ' minlimits_vec')  
			

			print(i, ' is sample', v_proposal, ' v_proposal')  


			v_proposal = v_current
			# Passing paramters to calculate likelihood and diff score

			[likelihood_proposal, pred_data, diff_score] = self.likelihood_func(reef, self.core_data, v_proposal)

			pred_core_vec = self.convert_core_format(pred_data, self.communities) # convert pred core vec to 1D format

			print (i, '\tinitial likelihood:', likelihood, 'and difference score:', diff_score)

		
		 

			 
			# Difference in likelihood from previous accepted proposal
			diff_likelihood = likelihood_proposal - likelihood

			try:
				mh_prob = min(1, math.exp(diff_likelihood))
			except OverflowError as e:
				mh_prob = 1

			u = random.uniform(0,1)


			accept_list[i+1] = num_accepted

			likeh_list[i+1,0] = likelihood_proposal


			print((i % self.swap_interval), i,  self.swap_interval, ' mod swap')



			if u < mh_prob: # Accept sample
				print (v_proposal,   i,likelihood_proposal, self.temperature, num_accepted, ' is accepted - rain, erod, step rain, step erod, likh')
				count_list.append(i)            # Append sample number to accepted list
				
				likelihood = likelihood_proposal 
				v_current = v_proposal 
				pos_param[i+1,:] = v_current # features rain, erodibility and others  (random walks is only done for this vector) 
				likeh_list[i + 1,1]=likelihood  # contains  all proposal liklihood (accepted and rejected ones) 
				list_core_pred[i+1,:] =  pred_core_vec # make a list of core predictions    
				num_accepted = num_accepted + 1 
 
			else: # Reject sample
				likeh_list[i + 1, 1]=likeh_list[i,1]  
				pos_param[i+1,:] = pos_param[i,:] 
				list_core_pred[i+1,:] =  list_core_pred[i,:] 
 

			if ( i % self.swap_interval == 0 ): 

				if i> burnsamples and self.runninghisto == True:
					hist, bin_edges = np.histogram(pos_param[burnsamples:i,0], density=True)
					plt.hist(pos_param[burnsamples:i,0], bins='auto')  # arguments are passed to np.histogram
					plt.title("Parameter 1 Histogram")

					file_name = self.filename + '/posterior/pos_parameters/hist_current' + str(self.temperature)
					plt.savefig(file_name+'_0.png')
					plt.clf()

					np.savetxt(file_name+'.txt',  pos_param[ :i,:] ,  fmt='%1.9f')

					hist, bin_edges = np.histogram(pos_param[burnsamples:i,1], density=True)
					plt.hist(pos_param[burnsamples:i,1], bins='auto')  # arguments are passed to np.histogram
					plt.title("Parameter 2 Histogram")
 
					plt.savefig(file_name + '_1.png')
					plt.clf()

 
				others = np.asarray([likelihood])
				param = np.concatenate([v_current,others])     
			 
				# paramater placed in queue for swapping between chains
				self.parameter_queue.put(param)
				
				
				#signal main process to start and start waiting for signal for main
				self.signal_main.set()              
				self.event.wait()
				

				# retrieve parametsrs fom ques if it has been swapped
				if not self.parameter_queue.empty() : 
					try:
						result =  self.parameter_queue.get()
 
						
						v_current= result[0:v_current.size]     
						likelihood = result[v_current.size]

					except:
						print ('error')

		 
		accepted_count =  len(count_list) 
		accept_ratio = accepted_count / (samples * 1.0) * 100



		#--------------------------------------------------------------- 

		others = np.asarray([ likelihood])
		param = np.concatenate([v_current,others])   

		self.parameter_queue.put(param)

  
 
		file_name = self.filename+'/posterior/pos_parameters/chain_'+ str(self.temperature)+ '.txt'
		np.savetxt(file_name,pos_param ) 

		file_name = self.filename+'/posterior/predicted_core/chain_'+ str(self.temperature)+ '.txt'
		np.savetxt(file_name, list_core_pred, fmt='%1.2f')
 
		file_name = self.filename+'/posterior/pos_likelihood/chain_'+ str(self.temperature)+ '.txt'
		np.savetxt(file_name,likeh_list, fmt='%1.2f')
 
		file_name = self.filename + '/posterior/accept_list/chain_' + str(self.temperature) + '_accept.txt'
		np.savetxt(file_name, [accept_ratio], fmt='%1.2f')

		file_name = self.filename + '/posterior/accept_list/chain_' + str(self.temperature) + '.txt'
		np.savetxt(file_name, accept_list, fmt='%1.2f')
 
 
 
 
		self.signal_main.set()


class ParallelTempering:

	def __init__(self, vec_parameters, num_chains, maxtemp,NumSample,swap_interval, fname, realvalues_vec, num_param,  core_depths, core_data,  simtime,     run_nb, inputxml):

	 

		self.swap_interval = swap_interval
		self.folder = fname
		self.maxtemp = maxtemp
		self.num_chains = num_chains
		self.chains = []
		self.tempratures = []
		self.NumSamples = int(NumSample/self.num_chains)
		self.sub_sample_size = max(1, int( 0.05* self.NumSamples))

		self.show_fulluncertainity = False # needed in cases when you reall want to see full prediction of 5th and 95th percentile of topo. takes more space 

		self.core_depths = core_depths
		self.core_data = core_data
 

		self.num_param = num_param
 
		self.simtime = simtime
  
		self.run_nb =run_nb 

		self.xmlinput = inputxml

		self.vec_parameters = vec_parameters

		self.realvalues  =  realvalues_vec 

		
		# create queues for transfer of parameters between process chain
		self.chain_parameters = [multiprocessing.Queue() for i in range(0, self.num_chains) ]

		# two ways events are used to synchronize chains
		self.event = [multiprocessing.Event() for i in range (self.num_chains)]
		self.wait_chain = [multiprocessing.Event() for i in range (self.num_chains)]
 

	
	# assigin tempratures dynamically   
	def assign_temptarures(self):
		tmpr_rate = (self.maxtemp /self.num_chains)
		temp = 1
		for i in xrange(0, self.num_chains):            
			self.tempratures.append(temp)
			temp += tmpr_rate
			print(self.tempratures[i])
			
	 
	def initialize_chains (self, vis, num_communities, num_sed, num_flow, sedlim, flowlim, max_a, max_m,   maxlimits_vec, minlimits_vec , stepratio_vec,  choose_likelihood,   burn_in):
		self.burn_in = burn_in
 
		self.vec_parameters =   initial_vec(num_communities, num_sed, num_flow, sedlim, flowlim, max_a, max_m)  


		self.assign_temptarures()
		for i in xrange(0, self.num_chains):
			self.chains.append(ptReplica(vis, num_communities, self.vec_parameters,  maxlimits_vec, minlimits_vec , stepratio_vec,  choose_likelihood ,self.swap_interval,   self.simtime, self.NumSamples,  self.core_data,  self.core_depths, self.folder, self.xmlinput,  self.run_nb,self.tempratures[i], self.chain_parameters[i], self.event[i], self.wait_chain[i],burn_in))
	 
		 
	

	def run_chains (self ):
		
		# only adjacent chains can be swapped therefore, the number of proposals is ONE less num_chains
		swap_proposal = np.ones(self.num_chains-1) 
		
		# create parameter holders for paramaters that will be swapped
		replica_param = np.zeros((self.num_chains, self.num_param))  
		lhood = np.zeros(self.num_chains)

		# Define the starting and ending of MCMC Chains
		start = 0
		end = self.NumSamples-1

		number_exchange = np.zeros(self.num_chains)

		filen = open(self.folder + '/num_exchange.txt', 'a')


		
		
		#-------------------------------------------------------------------------------------
		# run the MCMC chains
		#-------------------------------------------------------------------------------------
		for l in range(0,self.num_chains):
			self.chains[l].start_chain = start
			self.chains[l].end = end
		
		#-------------------------------------------------------------------------------------
		# run the MCMC chains
		#-------------------------------------------------------------------------------------
		for j in range(0,self.num_chains):        
			self.chains[j].start()



		flag_running = True 

		
		while flag_running:          

			#-------------------------------------------------------------------------------------
			# wait for chains to complete one pass through the samples
			#-------------------------------------------------------------------------------------

			for j in range(0,self.num_chains): 
				#print (j, ' - waiting')
				self.wait_chain[j].wait()
			

			
			#-------------------------------------------------------------------------------------
			#get info from chains
			#-------------------------------------------------------------------------------------
			
			for j in range(0,self.num_chains): 
				if self.chain_parameters[j].empty() is False :
					result =  self.chain_parameters[j].get()
					replica_param[j,:] = result[0:self.num_param]   
					lhood[j] = result[self.num_param]
 
 

			# create swapping proposals between adjacent chains
			for k in range(0, self.num_chains-1): 
				swap_proposal[k]=  (lhood[k]/[1 if lhood[k+1] == 0 else lhood[k+1]])*(1/self.tempratures[k] * 1/self.tempratures[k+1])

			#print(' before  swap_proposal  --------------------------------------+++++++++++++++++++++++=-')

			for l in range( self.num_chains-1, 0, -1):
				#u = 1
				u = random.uniform(0, 1)
				swap_prob = swap_proposal[l-1]



				if u < swap_prob : 

					number_exchange[l] = number_exchange[l] +1  

					others = np.asarray([  lhood[l-1] ]  ) 
					para = np.concatenate([replica_param[l-1,:],others])   
 
				   
					self.chain_parameters[l].put(para) 

					others = np.asarray([ lhood[l] ] )
					param = np.concatenate([replica_param[l,:],others])
 
					self.chain_parameters[l-1].put(param)
					
				else:


					others = np.asarray([  lhood[l-1] ])
					para = np.concatenate([replica_param[l-1,:],others]) 
 
				   
					self.chain_parameters[l-1].put(para) 

					others = np.asarray([  lhood[l]  ])
					param = np.concatenate([replica_param[l,:],others])
 
					self.chain_parameters[l].put(param)


			#-------------------------------------------------------------------------------------
			# resume suspended process
			#-------------------------------------------------------------------------------------
			for k in range (self.num_chains):
					self.event[k].set()
								

			#-------------------------------------------------------------------------------------
			#check if all chains have completed runing
			#-------------------------------------------------------------------------------------
			count = 0
			for i in range(self.num_chains):
				if self.chains[i].is_alive() is False:
					count+=1
					while self.chain_parameters[i].empty() is False:
						dummy = self.chain_parameters[i].get()

			if count == self.num_chains :
				flag_running = False
			

		#-------------------------------------------------------------------------------------
		#wait for all processes to jin the main process
		#-------------------------------------------------------------------------------------     
		for j in range(0,self.num_chains): 
			self.chains[j].join()

		print(number_exchange, 'num_exchange, process ended')


		pos_param, likelihood_rep, accept_list,  accept,  list_predcore = self.show_results('chain_')
 

 

		#self.view_crosssection_uncertainity(list_xslice, list_yslice)

		#optimal_para, para_5thperc, para_95thperc = self.get_uncertainity(likelihood_rep, pos_param)
		#np.savetxt(self.folder+'/optimal_percentile_para.txt', [optimal_para, para_5thperc, para_95thperc] )
 
		
		



		for s in range(self.num_param):  
			self.plot_figure(pos_param[s,:], 'pos_distri_'+str(s), self.realvalues[s]  ) 

 
					
			

		return (pos_param,likelihood_rep, accept_list,  list_predcore)

	def view_crosssection_uncertainity(self,  list_xslice, list_yslice):
		print ('list_xslice', list_xslice.shape)
		print ('list_yslice', list_yslice.shape)

		ymid = int(self.real_elev.shape[1]/2 ) #   cut the slice in the middle 
		xmid = int(self.real_elev.shape[0]/2)

		print( 'ymid',ymid)
		print( 'xmid', xmid)

		print(self.real_elev)

		print(self.real_elev.shape, ' shape')

		x_ymid_real = self.real_elev[xmid, :] 

		print( x_ymid_real.shape , ' x_ymid_real shape')
		y_xmid_real = self.real_elev[:, ymid ] 

		 
		x_ymid_mean = list_xslice.mean(axis=1)


		print( x_ymid_mean.shape , ' x_ymid_mean shape')


		x_ymid_5th = np.percentile(list_xslice, 5, axis=1)
		x_ymid_95th= np.percentile(list_xslice, 95, axis=1)
  
		y_xmid_mean = list_yslice.mean(axis=1)
		y_xmid_5th = np.percentile(list_yslice, 5, axis=1)
		y_xmid_95th= np.percentile(list_yslice, 95, axis=1)
 

		x = np.linspace(0, x_ymid_mean.size * self.resolu_factor, num=x_ymid_mean.size) 
		x_ = np.linspace(0, y_xmid_mean.size * self.resolu_factor, num=y_xmid_mean.size)

		#ax.set_xlim(-width,len(ind)+width)
 
		plt.plot(x, x_ymid_real, label='ground truth') 
		plt.plot(x, x_ymid_mean, label='pred. (mean)')
		plt.plot(x, x_ymid_5th, label='pred.(5th percen.)')
		plt.plot(x, x_ymid_95th, label='pred.(95th percen.)')


		plt.fill_between(x, x_ymid_5th , x_ymid_95th, facecolor='g', alpha=0.4)
		plt.legend(loc='upper right') 


		plt.title("Uncertainty in topography prediction (cross section)  ")
		plt.xlabel(' Distance (km)  ')
		plt.ylabel(' Height in meters')
		
		plt.savefig(self.folder+'/x_ymid_opt.png')  
		plt.savefig(self.folder+'/x_ymid_opt.svg', format='svg', dpi=400)
		plt.clf()


		plt.plot(x_, y_xmid_real, label='ground truth') 
		plt.plot(x_, y_xmid_mean, label='pred. (mean)') 
		plt.plot(x_, y_xmid_5th, label='pred.(5th percen.)')
		plt.plot(x_, y_xmid_95th, label='pred.(95th percen.)')
		plt.xlabel(' Distance (km) ')
		plt.ylabel(' Height in meters')
		
		plt.fill_between(x_, y_xmid_5th , y_xmid_95th, facecolor='g', alpha=0.4)
		plt.legend(loc='upper right')

		plt.title("Uncertainty in topography prediction  (cross section)  ")
		plt.savefig(self.folder+'/y_xmid_opt.png')  
		plt.savefig(self.folder+'/y_xmid_opt.svg', format='svg', dpi=400)

		plt.clf()

		 



	# Merge different MCMC chains y stacking them on top of each other
	def show_results(self, filename):

		burnin = int(self.NumSamples * self.burn_in)
		pos_param = np.zeros((self.num_chains, self.NumSamples - burnin, self.num_param))

		pred_core = np.zeros((self.num_chains, self.NumSamples - burnin, self.core_data.shape[0])) 

		print(self.core_data.size, ' self.core_data.size')
		print(self.core_data, ' self.core_data ')

 

 
		likehood_rep = np.zeros((self.num_chains, self.NumSamples - burnin, 2 )) # index 1 for likelihood posterior and index 0 for Likelihood proposals. Note all likilihood proposals plotted only
		accept_percent = np.zeros((self.num_chains, 1))

		accept_list = np.zeros((self.num_chains, self.NumSamples )) 
 
 
 
		for i in range(self.num_chains):
			file_name = self.folder + '/posterior/pos_parameters/'+filename + str(self.tempratures[i]) + '.txt'
			dat = np.loadtxt(file_name) 
			pos_param[i, :, :] = dat[burnin:,:]
   

			file_name = self.folder + '/posterior/predicted_core/chain_'+  str(self.tempratures[i]) + '.txt'
			dat = np.loadtxt(file_name) 
			pred_core[i, :, :] = dat[burnin:,:] 

			file_name = self.folder + '/posterior/pos_likelihood/'+filename + str(self.tempratures[i]) + '.txt'
			dat = np.loadtxt(file_name) 
			likehood_rep[i, :] = dat[burnin:]

			file_name = self.folder + '/posterior/accept_list/' + filename + str(self.tempratures[i]) + '.txt'
			dat = np.loadtxt(file_name) 
			accept_list[i, :] = dat 

			file_name = self.folder + '/posterior/accept_list/' + filename + str(self.tempratures[i]) + '_accept.txt'
			dat = np.loadtxt(file_name) 
			accept_percent[i, :] = dat
 
 
 
 
 

		posterior = pos_param.transpose(2,0,1).reshape(self.num_param,-1)     
		list_predcore = pred_core.transpose(2,0,1).reshape(  self.core_data.shape[0],-1)
 

		likelihood_vec = likehood_rep.transpose(2,0,1).reshape(2,-1) 
  
 

		accept = np.sum(accept_percent)/self.num_chains
  

		np.savetxt(self.folder + '/pos_param.txt', posterior.T) 
		
		np.savetxt(self.folder + '/likelihood.txt', likelihood_vec.T, fmt='%1.5f')

		np.savetxt(self.folder + '/accept_list.txt', accept_list, fmt='%1.2f')
  
		np.savetxt(self.folder + '/acceptpercent.txt', [accept], fmt='%1.2f')

		return posterior, likelihood_vec.T, accept_list,   accept,   list_predcore


	def find_nearest(self, array,value): # just to find nearest value of a percentile (5th or 9th from pos likelihood)
		idx = (np.abs(array-value)).argmin()
		return array[idx], idx

	def get_uncertainity(self, likehood_rep, pos_param ): 

		likelihood_pos = likehood_rep[:,1]
 
		a = np.percentile(likelihood_pos, 5)   
		lhood_5thpercentile, index_5th = self.find_nearest(likelihood_pos,a)  
		b = np.percentile(likelihood_pos, 95) 
		lhood_95thpercentile, index_95th = self.find_nearest(likelihood_pos,b)  
 

		max_index = np.argmax(likelihood_pos) # find max of pos liklihood to get the max or optimal pos value  

		optimal_para = pos_param[:, max_index] 
		para_5thperc = pos_param[:, index_5th]
		para_95thperc = pos_param[:, index_95th] 


		return optimal_para, para_5thperc, para_95thperc

 

	def plot_figure(self, list, title, real_value ): 

		list_points =  list

		fname = self.folder
		width = 9 

		font = 9

		fig = plt.figure(figsize=(10, 12))
		ax = fig.add_subplot(111)
 

		slen = np.arange(0,len(list),1) 
		 
		fig = plt.figure(figsize=(10,12))
		ax = fig.add_subplot(111)
		ax.spines['top'].set_color('none')
		ax.spines['bottom'].set_color('none')
		ax.spines['left'].set_color('none')
		ax.spines['right'].set_color('none')
		ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
		ax.set_title(' Posterior distribution', fontsize=  font+2)#, y=1.02)
	
		ax1 = fig.add_subplot(211) 

		n, rainbins, patches = ax1.hist(list_points,  bins = 20,  alpha=0.5, facecolor='sandybrown', normed=False)	
 
   
		ax1.axvline(x=real_value, color='blue', linestyle='dashed', linewidth=1) # comment when go real value is 

		print(real_value)

		ax1.grid(True)
		ax1.set_ylabel('Frequency',size= font+1)
		ax1.set_xlabel('Parameter values', size= font+1)
	
		ax2 = fig.add_subplot(212)

		list_points = np.asarray(np.split(list_points,  self.num_chains ))
 

 

		ax2.set_facecolor('#f2f2f3') 
		ax2.plot( list_points.T , label=None)
		ax2.set_title(r'Trace plot',size= font+2)
		ax2.set_xlabel('Samples',size= font+1)
		ax2.set_ylabel('Parameter values', size= font+1) 

		fig.tight_layout()
		fig.subplots_adjust(top=0.88)
		 
 
		plt.savefig(fname + '/plot_pos/' + title  + '_pos_.png', bbox_inches='tight', dpi=300, transparent=False)
		plt.clf()
 

 

def mean_sqerror(  pred_erodep, pred_elev,  real_elev,  real_erodep_pts):
		 
		elev = np.sqrt(np.sum(np.square(pred_elev -  real_elev))  / real_elev.size)  
		sed =  np.sqrt(  np.sum(np.square(pred_erodep -  real_erodep_pts)) / real_erodep_pts.size  ) 

		return elev + sed, sed


def find_limits(communities, num_sed, num_flow, sedlim, flowlim,  max_a, max_m):

	sedmax_vec =  np.repeat(sedlim[1], communities * num_sed )  
	sedmin_vec =  np.repeat(sedlim[0], communities * num_sed ) 

	flowmax_vec =  np.repeat(flowlim[1], communities * num_flow )  
	flowmin_vec =  np.repeat(flowlim[0], communities * num_flow) 
 
	glv_max = np.array([  0., 0., max_m]) 
	glv_min = np.array([  max_a, max_a, 0.])

	maxlimits_vec = np.concatenate((sedmax_vec , flowmax_vec, glv_max) )
	minlimits_vec = np.concatenate((sedmin_vec , flowmin_vec, glv_min))


	return   maxlimits_vec, minlimits_vec



def initial_vec(communities, num_sed, num_flow, sedlim, flowlim, lim_axy, lim_malthu):
 
 

	sed1 = np.zeros(communities)
	sed2 = np.zeros(communities)
	sed3 = np.zeros(communities)
	sed4 = np.zeros(communities)
 
	for s in range(communities):

		sed1[s] =   np.random.uniform(0.,sedlim[1])
		sed2[s] =  np.random.uniform(sed1[s],sedlim[1])
		sed3[s] =  np.random.uniform(sed2[s],sedlim[1])
		sed4[s] =  np.random.uniform(sed3[s],sedlim[1])

	flow1 = np.zeros(communities)
	flow2 = np.zeros(communities)
	flow3 = np.zeros(communities)
	flow4 = np.zeros(communities) 
			
	for s in range(communities):

		flow1[s] = np.random.uniform(0., flowlim[1])
		flow2[s] =  np.random.uniform(flow1[s], flowlim[1])
		flow3[s] =  np.random.uniform(flow2[s], flowlim[1])
		flow4[s] =   np.random.uniform(flow3[s], flowlim[1])
		
	cm_ax =   np.random.uniform(lim_axy,0.)
	cm_ay =   np.random.uniform(lim_axy,0.)
	m = np.random.uniform(0., lim_malthu)

  
	init_pro = np.concatenate((sed1,sed2,sed3,sed4,flow1,flow2,flow3,flow4))
	init_pro = np.append(init_pro,( cm_ax,cm_ay, m)) 

	  
	return init_pro 


def make_directory (directory): 
	if not os.path.exists(directory):
		os.makedirs(directory)
 

def main():

	random.seed(time.time()) 

	samples = 40 # total number of samples by all the chains (replicas) in parallel tempering

	run_nb = 0

	choose_likelihood = 1 # 1 for Multinomial, 2 for Gaussian Likilihood

  

	
  
	

		 
 


	#problem = input("Which problem do you want to choose 1. Testing (Synthetic, 2. Henon  3. OneTreeReef ")
	problem = 1
	if problem == 1:

		num_communities = 3  # can be 6 for real probs
		num_flow = 4
		num_sed = 4

		simtime = 8500
		timestep = np.arange(0,simtime+1,50)
 
		sedlim = [0., 0.005]
		flowlim = [0.,0.2]

		max_a = -0.10  # community interaction matrix diagonal and sub-diagnoal limits
		max_m = 0.10 # malthusian parameter limit


		true_parameter_vec = np.loadtxt('SyntheticProblem/data/true_param.txt')

		problemfolder = 'SyntheticProblem/'  # change for other reef-core (This is synthetic core) 

		vec_parameters  = initial_vec(num_communities, num_sed, num_flow, sedlim, flowlim, max_a, max_m)
		maxlimits_vec, minlimits_vec = find_limits(num_communities, num_sed, num_flow, sedlim, flowlim, max_a, max_m)

		maxlimits_vec[24] = true_parameter_vec[24] # this is how you make them fixed - eg we making comm interaction parameters fixed
		maxlimits_vec[25] = true_parameter_vec[25]

		minlimits_vec[24] = true_parameter_vec[24]  # this is how you make them fixed
		minlimits_vec[25] = true_parameter_vec[25]

		vec_parameters = true_parameter_vec

		print(vec_parameters)






		print(vec_parameters)
 

		likelihood_sediment = True
 
		 
		
		stepsize_ratio  = 0.02 #   you can have different ratio values for different parameters depending on the problem. Its safe to use one value for now

		stepratio_vec =  np.repeat(stepsize_ratio, vec_parameters.size) 

		
		print(maxlimits_vec, ' maxlimits_vec')


		print(minlimits_vec, ' minlimits_vec')

		num_param = vec_parameters.size 
   
  
	 
	else:
		print('choose some problem  ')



	xmlinput = problemfolder + 'input_synth.xml'
	datafile = problemfolder + 'data/synth_core_vec.txt'

	core_depths, data_vec = np.genfromtxt(datafile, usecols=(0, 1), unpack = True) 
	core_data = np.loadtxt(problemfolder + 'data/synth_core_prop.txt', usecols=(1,2,3,4))

	vis = [False, False] # first for initialisation, second for cores

	sedsim, flowsim = True, True  # can pass this to pt class later
 
 

	fname = ""
	run_nb = 0
	while os.path.exists(problemfolder +'results_%s' % (run_nb)):
		run_nb += 1
	if not os.path.exists(problemfolder +'results_%s' % (run_nb)):
		os.makedirs(problemfolder +'results_%s' % (run_nb))
		fname = (problemfolder +'results_%s' % (run_nb))
 
  
	make_directory((fname + '/posterior/pos_parameters')) 
	make_directory((fname + '/posterior/predicted_core'))
	make_directory((fname + '/posterior/pos_likelihood'))
	make_directory((fname + '/posterior/accept_list')) 

	make_directory((fname + '/plot_pos'))

	run_nb_str = 'results_' + str(run_nb)

	#-------------------------------------------------------------------------------------
	# Number of chains of MCMC required to be run
	# PT is a multicore implementation must num_chains >= 2
	# Choose a value less than the numbe of core available (avoid context swtiching)
	#-------------------------------------------------------------------------------------
	num_chains = 2  # number of Replica's that will run on separate cores. Note that cores will be shared automatically - if enough cores not available
	swap_ratio = 0.1    #adapt these 
	burn_in =0.1  

 

	#parameters for Parallel Tempering
	maxtemp = int(num_chains * 5)/2
	
	swap_interval =   int(swap_ratio * (samples/num_chains)) #how ofen you swap neighbours
	print(swap_interval, ' swap')

	timer_start = time.time()

 

	#-------------------------------------------------------------------------------------
	#Create A a Patratellel Tempring object instance 
	#-------------------------------------------------------------------------------------
 

	pt = ParallelTempering(  vec_parameters,  num_chains, maxtemp, samples,swap_interval,fname, true_parameter_vec, num_param  ,  core_depths, core_data, simtime,    run_nb_str, xmlinput)
	 
	pt.initialize_chains( vis, num_communities, num_sed, num_flow, sedlim, flowlim, max_a, max_m, maxlimits_vec, minlimits_vec , stepratio_vec, choose_likelihood,   burn_in)
	 


	#-------------------------------------------------------------------------------------
	#run the chains in a sequence in ascending order
	#-------------------------------------------------------------------------------------
	pos_param,likehood_rep, accept_list,    pred_core  = pt.run_chains()

	print(pred_core, ' pred core')




	print('sucessfully sampled') 

	timer_end = time.time() 
	likelihood = likehood_rep[:,0] # just plot proposed likelihood  
	likelihood = np.asarray(np.split(likelihood,  num_chains ))
 
 

	plt.plot(likelihood.T)
	plt.savefig( fname+'/likelihood.png')
	plt.clf()
	plt.plot(accept_list.T)
	plt.savefig( fname+'/accept_list.png')
	plt.clf()
 
 
	
	print ('time taken  in minutes = ', (timer_end-timer_start)/60)
	np.savetxt(fname+'/time_sqerror.txt',[ (timer_end-timer_start)/60,  rmse_sed, rmse], fmt='%1.2f'  )

	mpl_fig = plt.figure()
	ax = mpl_fig.add_subplot(111)

	ax.boxplot(pos_param.T) 
	ax.set_xlabel('Badlands parameters')
	ax.set_ylabel('Posterior') 
	plt.legend(loc='upper right') 
	plt.title("Boxplot of Posterior")
	plt.savefig(fname+'plot_pos/pyreef_pos.png')
	plt.savefig(fname+'plot_pos/pyreef_pos.svg', format='svg', dpi=400)


	 


	#stop()
if __name__ == "__main__": main()
