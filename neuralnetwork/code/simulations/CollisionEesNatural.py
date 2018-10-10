from mpi4py import MPI
from neuron import h
from Simulation import Simulation
from cells import AfferentFiber
import random as rnd
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pickle
from tools import seed_handler as sh
sh.set_seed()

comm = MPI.COMM_WORLD
sizeComm = comm.Get_size()
rank = comm.Get_rank()

class CollisionEesNatural(Simulation):
	""" Simulation to evaluate the effect of EES on the natural firing rate in function of the fiber
	legnth, of its firing rate and of the stimulation frequency.
	"""

	def __init__(self, parallelContext, eesFrequencies, fiberDelays, fiberFiringRates, segmentToRecord = None, tstop = 5000):
		""" Object initialization.

		Keyword arguments:
		parallelContext -- Neuron parallelContext object.
		eesFrequencies -- List of stimulation frequencies to test.
		fiberDelays -- List of fiber delays to test.
		fiberFiringRates -- List of fiber firing rates to test.
		segmentToRecord -- Segment to record in case of a real afferent fiber model (default = None).
		tstop -- Time in ms at wich the simulation will stop (default = 500).
		"""

		Simulation.__init__(self,parallelContext)

		if rank==1:
			print "\nMPI execution: the different processes have different stimulation starting time."
			print "The final result is the mean results between each process\n"

		# Variables initializations
		self._eesFrequencies = eesFrequencies
		self._fiberDelays = fiberDelays
		self._fiberFiringRates = fiberFiringRates
		self._segmentToRecord = segmentToRecord

		self._init_lists()
		self._results = np.zeros([len(self._eesFrequencies),len(self._fiberDelays),len(self._fiberFiringRates)])

		self._create_fibers()
		self._create_ees_objects()
		self._connect_ees_to_fibers()

		self._set_tstop(tstop)
		self._set_integration_step(AfferentFiber.get_update_period())


	"""
	Redefinition of inherited methods
	"""

	def _update(self):
		""" Update simulation parameters. """
		self._update_afferents()

	def _end_integration(self):
		""" Print the total simulation time and extract the results. """
		Simulation._end_integration(self)
		self._extract_results()

	def save_results(self,name=""):
		""" Save the simulation results.

		Keyword arguments:
		name -- string to add at predefined file name (default = "").
		"""
		fileName = time.strftime("%Y_%m_%d_resultsCollisionEesNatural"+name+".p")
		with open(self._resultsFolder+fileName, 'w') as pickle_file:
			pickle.dump(self._results, pickle_file)
			pickle.dump(self._eesFrequencies, pickle_file)
			pickle.dump(self._fiberDelays, pickle_file)
			pickle.dump(self._fiberFiringRates, pickle_file)

	def plot(self,delay,nColorLevels=None,name=""):
		""" Plot the simulation results.

		Plot the percantage of collisions for a given delay in fucntion of the afferent
		firing rate and of the stimulation frequency.
		Keyword arguments:
		delay -- fiber delay for which we want the plot.
		Threshold -- threshold to plot binary simulation results (default = None).
		"""
		if rank == 0:
			fig, ax = plt.subplots(figsize=(16,9))

			dataToPlot = self._results[:,delay,:]
			title = "Percentage of sensory information erased by the stimulation\n (delay "+str(self._fiberDelays[delay])+" ms)"
			if nColorLevels is not None:
				dataToPlot=np.round(dataToPlot/100*nColorLevels)*100/nColorLevels
				title += "\nnColorLevels = "+str(nColorLevels)

			# cmap = plt.cm.gray
			cmap = plt.cm.bone_r
			im = ax.imshow(dataToPlot, cmap=cmap, interpolation='nearest',origin="lower",vmin = 0, vmax = 100)
			ax.set_title(title)

			# Move left and bottom spines outward by 10 points
			ax.spines['left'].set_position(('outward', 10))
			ax.spines['bottom'].set_position(('outward', 10))
			# Hide the right and top spines
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
			# Only show ticks on the left and bottom spines
			ax.yaxis.set_ticks_position('left')
			ax.xaxis.set_ticks_position('bottom')
			fig.colorbar(im, orientation='vertical',label='% Erased APs')

			plt.yticks(range(len(self._eesFrequencies)),self._eesFrequencies)
			plt.xticks(range(len(self._fiberFiringRates)),self._fiberFiringRates)
			plt.ylabel('EES eesFrequency (Hz)')
			plt.xlabel('Natural afferent firing rate (hz)')

			fileName = time.strftime("%Y_%m_%d_CollisionEesNatural_Delay_"+str(self._fiberDelays[delay])+name+".pdf")
			plt.savefig(self._resultsFolder+fileName, format="pdf",transparent=True)
			plt.show(block=False)

	"""
	Specific Methods of this class
	"""
	def _init_lists(self):
		""" Initialize lists containg the fibers, netcon objects and ees objects. """
		self._fiberList = [[[] for i in range(len(self._fiberDelays))] for j in range(len(self._eesFrequencies))]
		self._netconList = [[[] for i in range(len(self._fiberDelays))] for j in range(len(self._eesFrequencies))]
		self._eesList = []

	def _create_fibers(self):
		""" Create the fibers with the defined different delays. """
		for i in range(len(self._eesFrequencies)):
			for j in range(len(self._fiberDelays)):
				for k in range(len(self._fiberFiringRates)):
					self._fiberList[i][j].append(AfferentFiber(self._fiberDelays[j]))
					if self._segmentToRecord is None:
						self._fiberList[i][j][k].set_firing_rate(self._fiberFiringRates[k])
					if self._segmentToRecord is not None:
						self._fiberList[i][j][k].set_firing_rate(self._fiberFiringRates[k],False)
						self._fiberList[i][j][k].set_recording(True,self._segmentToRecord)

	def _create_ees_objects(self):
		""" Create different ees objects with the defined stimulation frequencies. """
		scale = rnd.random()
		for i in range(len(self._eesFrequencies)):
			self._eesList.append(h.NetStim())
			self._eesList[i].interval = 1000.0/self._eesFrequencies[i]
			self._eesList[i].number = 10000
			self._eesList[i].start = 10.0*scale
			self._eesList[i].noise = 0

	def _connect_ees_to_fibers(self):
		""" Connect fibers ojects to ees objects to make the stimulation activate these fibers. """
		for i in range(len(self._eesFrequencies)):
			for j in range(len(self._fiberDelays)):
				for k in range(len(self._fiberFiringRates)):
					self._netconList[i][j].append(h.NetCon(self._eesList[i],self._fiberList[i][j][k].cell))
					self._netconList[i][j][k].delay = 1
					self._netconList[i][j][k].weight[0] = AfferentFiber.get_ees_weight()

	def _update_afferents(self):
		""" Update the afferents fiber state. """
		for i in range(len(self._eesFrequencies)):
			for j in range(len(self._fiberDelays)):
				for k in range(len(self._fiberFiringRates)):
					self._fiberList[i][j][k].update(h.t)

	def _extract_results(self):
		""" Extract the simulation results. """
		for i in range(len(self._eesFrequencies)):
			for j in range(len(self._fiberDelays)):
				for k in range(len(self._fiberFiringRates)):
					sent,arr,coll,perc=self._fiberList[i][j][k].get_stats()
					self._results[i,j,k]=perc
		comm.Barrier()
		if sizeComm>1:
			temp = comm.gather(self._results, root=0)
			if rank==0:
				for i in range(1,sizeComm):
					self._results += temp[i]
				self._results/=sizeComm

	def plot_isoinformation_surface(self,percentage=50):
		""" Plot a surface where the number of AP erased by the stimulation is equal. """
		if rank==0:
			Z = np.zeros([len(self._fiberFiringRates),len(self._fiberDelays)])
			temp = (self._results - percentage)**2
			for x in xrange(len(self._fiberFiringRates)):
				for y in xrange(len(self._fiberDelays)):
					Z[x,y]=self._eesFrequencies[temp[:,y,x].argmin()]

			fig, ax = plt.subplots(figsize=(16,9))
			im = ax.imshow(Z, cmap=plt.cm.bone, interpolation='nearest',origin="lower")
			ax.set_title("Isoinformation surface - "+str(percentage)+"% of APs erased")

			# Move left and bottom spines outward by 10 points
			ax.spines['left'].set_position(('outward', 10))
			ax.spines['bottom'].set_position(('outward', 10))
			# Hide the right and top spines
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
			# Only show ticks on the left and bottom spines
			ax.yaxis.set_ticks_position('left')
			ax.xaxis.set_ticks_position('bottom')
			fig.colorbar(im, orientation='vertical', label="Stimulation frequency (Hz)")

			plt.xticks(range(len(self._fiberDelays)),self._fiberDelays)
			plt.yticks(range(len(self._fiberFiringRates)),self._fiberFiringRates)
			plt.xlabel('Fiber delay (ms)')
			plt.ylabel('Natural afferents firing rate (hz)')

			fileName = time.strftime("%Y_%m_%d_CollisionEesNatural_Isoinfo_"+str(percentage)+"perc.pdf")
			plt.savefig(self._resultsFolder+fileName, format="pdf",transparent=True)
			plt.show(block=False)

	def plot_recorded_segment(self, freqInd = 0, delInd = 0, firInd = 0):
		""" Plot recorded spikes from a fiber.

			Keyword arguments:
			freqInd -- index of the stimulation frequencies.
			delInd -- index of the fiber delay.
			firInd -- index of the fiber natural firing rate.
		"""

		if self._segmentToRecord == None: return
		naturalSignals, eesInducedSignals, trigger, time = self._fiberList[freqInd][delInd][firInd].get_recording()
		nNaturalSent,nNaturalArrived,nCollisions,percErasedAp = self._fiberList[freqInd][delInd][firInd].get_stats()

		fig1, ax1 = plt.subplots(1, 1, figsize=(8,4.5))
		msToRec = 20
		sumTrigNaturalSignal = np.zeros(msToRec)
		sumTrigEesSignal = np.zeros(msToRec)
		nPeripheralStims = nNaturalSent*np.ones(msToRec)

		for i,val in enumerate(trigger):
			if val and i+msToRec<len(time):
				sumTrigNaturalSignal += naturalSignals[i:i+msToRec]
				sumTrigEesSignal += eesInducedSignals[i:i+msToRec]
		ax1.plot(sumTrigNaturalSignal,color='b',label='peripheral')
		ax1.plot(sumTrigEesSignal,color='r',label='spinal')
		ax1.plot(nPeripheralStims,color='g',ls = '--', label='n peripheral stims')
		ax1.set_ylim([-10,nNaturalSent+10])

		collisionPerc = 100*(1-sumTrigNaturalSignal.max()/float(nNaturalSent))
		ax1.set_title("Spikes in segment {0}, collision perc: {1:.1f}%".format(self._segmentToRecord,collisionPerc))
		ax1.legend()

		fig2, ax2 = plt.subplots(1, 1, figsize=(16,4))
		ax2.plot(time,naturalSignals,color='b',label='peripheral')
		ax2.plot(time,eesInducedSignals,color='r',label='spinal')
		ax2.plot(time,trigger,color='g',label='trigger',ls='--')
		ax2.set_ylim([-0.5,1.5])
		ax2.legend()
		plt.show()
