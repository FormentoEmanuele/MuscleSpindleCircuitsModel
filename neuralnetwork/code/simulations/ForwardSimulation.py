from mpi4py import MPI
from neuron import h
from Simulation import Simulation
from cells import AfferentFiber
import random as rnd
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tools import firings_tools as tlsf
import pickle
from tools import seed_handler as sh
sh.set_seed()

comm = MPI.COMM_WORLD
sizeComm = comm.Get_size()
rank = comm.Get_rank()

class ForwardSimulation(Simulation):
	""" Integration of a NeuralNetwork object over time given an input (ees or afferent input or both). """

	def __init__(self, parallelContext, neuralNetwork, afferentInput=None, eesObject=None, eesModulation=None, tStop = 100):
		""" Object initialization.

		Keyword arguments:
		parallelContext -- Neuron parallelContext object.
		neuralNetwork -- NeuralNetwork object.
		afferentInput -- Dictionary of lists for each type of fiber containing the
			fibers firing rate over time and the dt at wich the firing rate is updated.
			If no afferent input is desired use None (default = None).
		eesObject -- EES object connected to the NeuralNetwork, usefull for some plotting
			info and mandatory for eesModulation (Default = None).
		eesModulation -- possible dictionary with the following strucuture: {'modulation':
			dictionary containing a	signal of 0 and 1s used to activate/inactivate
			the stimulation for every muscle that we want to modulate (the dictionary
			keys have to be the muscle names used in the neural network structure), 'dt':
			modulation dt}. If no modulation of the EES is intended use None (default = None).
		tStop -- Time in ms at wich the simulation will stop (default = 100). In case
			the time is set to -1 the neuralNetwork will be integrated for all the duration
			of the afferentInput.
		"""

		Simulation.__init__(self,parallelContext)

		if rank==1:
			print "\nMPI execution: the cells are divided in the different hosts\n"

		self._nn = neuralNetwork
		self._Iaf = self._nn.get_primary_afferents_names()[0] if self._nn.get_primary_afferents_names() else []
		self._IIf = self._nn.get_secondary_afferents_names()[0] if self._nn.get_secondary_afferents_names() else []
		self._Mn = self._nn.get_motoneurons_names() if self._nn.get_motoneurons_names() else []

		self._set_integration_step(AfferentFiber.get_update_period())

		# Initialization of the afferent modulation
		if afferentInput == None:
			self._afferentModulation = False
			self._afferentInput = None
			if tStop>0: self._set_tstop(tStop)
			else : raise(Exception("If no afferents input are provided tStop has to be greater than 0."))
		else:
			self._afferentModulation = True
			self._afferentInput = afferentInput[0]
			self._dtUpdateAfferent = afferentInput[1]
			self._init_afferents_fr()
			key=[]
			key.append(self._afferentInput.keys()[0])
			key.append(self._afferentInput[key[0]].keys()[0])
			self._inputDuration = len(self._afferentInput[key[0]][key[1]])*self._dtUpdateAfferent
			if tStop == -1 or tStop>= self._inputDuration: self._set_tstop(self._inputDuration-self._dtUpdateAfferent)
			else : self._set_tstop(tStop)

		self._ees = eesObject
		# Initialization of the binary stim modulation
		if eesModulation == None or eesObject == None:
			self._eesBinaryModulation = False
			self._eesProportionalModulation = False
			self._eesParam = {'state':None, 'amp':None, 'modulation':None, 'dt':None}
		elif eesModulation['type']=="binary":
			self._eesBinaryModulation = True
			self._eesProportionalModulation = False
			current, percIf, percIIf, percMn = self._ees.get_amplitude()
			self._eesParam = {'state':{}, 'modulation':eesModulation['modulation'], 'dt':eesModulation['dt']}
			self._eesParam['amp'] = [percIf, percIIf, percMn]
			for muscle in eesModulation['modulation']:
				self._eesParam['state'][muscle] = 1
		elif eesModulation['type']=="proportional":
			self._eesBinaryModulation = False
			self._eesProportionalModulation = True
			self._eesParam = {'modulation':eesModulation['modulation'], 'dt':eesModulation['dt']}
			current, percIf, percIIf, percMn = self._ees.get_amplitude()
			self._eesParam['maxAmp'] = np.array([percIf, percIIf, percMn])

		#Initialization of the result dictionaries
		self._meanFr = None
		self._estimatedEMG = None
		self._nSpikes = None
		self._nActiveCells = None

	"""
	Redefinition of inherited methods
	"""
	def _initialize(self):
		Simulation._initialize(self)
		self._init_aff_fibers()
		self._timeUpdateAfferentsFr = 0
		self._timeUpdateEES = 0

	def _update(self):
		""" Update simulation parameters. """
		comm.Barrier()
		self._nn.update_afferents_ap(h.t)
		if self._afferentModulation:
			if h.t-self._timeUpdateAfferentsFr>= (self._dtUpdateAfferent-0.5*self._get_integration_step()):
				self._timeUpdateAfferentsFr = h.t
				self._set_afferents_fr(int(h.t/self._dtUpdateAfferent))
				self._nn.set_afferents_fr(self._afferentFr)

		if self._eesBinaryModulation:
			if h.t-self._timeUpdateEES>= (self._eesParam['dt']-0.5*self._get_integration_step()):
				ind = int(h.t/self._eesParam['dt'])
				for muscle in self._eesParam['modulation']:
					if self._eesParam['state'][muscle] != self._eesParam['modulation'][muscle][ind]:
						if self._eesParam['state'][muscle] == 0: self._ees.set_amplitude(self._eesParam['amp'],[muscle])
						else: self._ees.set_amplitude([0,0,0],[muscle])
						self._eesParam['state'][muscle] = self._eesParam['modulation'][muscle][ind]

		if self._eesProportionalModulation:
			if h.t-self._timeUpdateEES>= (self._eesParam['dt']-0.5*self._get_integration_step()):
				ind = int(h.t/self._eesParam['dt'])
				for muscle in self._eesParam['modulation']:
					amp = list(self._eesParam['maxAmp']*self._eesParam['modulation'][muscle][ind])
					self._ees.set_amplitude(amp,[muscle])

	def _end_integration(self):
		""" Print the total simulation time and extract the results. """
		Simulation._end_integration(self)
		self._extract_results()

	"""
	Specific Methods of this class
	"""
	def _init_aff_fibers(self):
		""" Return the percentage of afferent action potentials erased by the stimulation. """
		for muscleName in self._nn.cells:
			for cellName in self._nn.cells[muscleName]:
				if cellName in self._nn.get_afferents_names():
					for fiber in self._nn.cells[muscleName][cellName]:
						fiber.initialise()


	def _init_afferents_fr(self):
		""" Initialize the dictionary necessary to update the afferent fibers. """
		self._afferentFr = {}
		for muscle in self._afferentInput:
			self._afferentFr[muscle]={}
			for cellType in self._afferentInput[muscle]:
				if cellType in self._nn.get_afferents_names():
					self._afferentFr[muscle][cellType]= 0.
				else: raise(Exception("Wrong afferent input structure!"))

	def _set_afferents_fr(self,i):
		""" Set the desired firing rate in the _afferentFr dictionary. """
		for muscle in self._afferentInput:
			for cellType in self._afferentInput[muscle]:
				self._afferentFr[muscle][cellType] = self._afferentInput[muscle][cellType][i]

	def _extract_results(self,samplingRate = 1000.):
		""" Extract the simulation results. """
		if rank==0: print "Extracting the results... ",
		self._firings = {}
		self._meanFr = {}
		self._estimatedEMG = {}
		self._nSpikes = {}
		self._nActiveCells = {}
		for muscle in self._nn.actionPotentials:
			self._firings[muscle]={}
			self._meanFr[muscle]={}
			self._estimatedEMG[muscle]={}
			self._nSpikes[muscle]={}
			self._nActiveCells[muscle]={}
			for cell in self._nn.actionPotentials[muscle]:
				self._firings[muscle][cell] = tlsf.exctract_firings(self._nn.actionPotentials[muscle][cell],self._get_tstop(),samplingRate)
				if rank==0: self._nActiveCells[muscle][cell] = np.count_nonzero(np.sum(self._firings[muscle][cell],axis=1))
				self._nSpikes[muscle][cell] = np.sum(self._firings[muscle][cell])
				self._meanFr[muscle][cell] = tlsf.compute_mean_firing_rate(self._firings[muscle][cell],samplingRate)
				if cell in self._nn.get_motoneurons_names():
					self._estimatedEMG[muscle][cell] = tlsf.synth_rat_emg(self._firings[muscle][cell],samplingRate)
		if rank==0: print "...completed."

	def get_estimated_emg(self,muscleName):
		emg = [self._estimatedEMG[muscleName][mnName] for mnName in self._Mn]
		emg = np.sum(emg,axis=0)
		return emg

	def get_mn_spikes_profile(self,muscleName):
		spikesProfile = [self._firings[muscleName][mnName] for mnName in self._Mn]
		spikesProfile = np.sum(spikesProfile,axis=0)
		spikesProfile = np.sum(spikesProfile,axis=0)
		return spikesProfile

	def _get_perc_aff_ap_erased(self,muscleName,cellName):
		""" Return the percentage of afferent action potentials erased by the stimulation. """
		if cellName in self._nn.get_afferents_names():
			percErasedAp = []
			meanPercErasedAp = None
			for fiber in self._nn.cells[muscleName][cellName]:
				sent,arrived,collisions,perc = fiber.get_stats()
				percErasedAp.append(perc)
			percErasedAp = comm.gather(percErasedAp,root=0)

			if rank==0:
				percErasedAp = sum(percErasedAp,[])
				meanPercErasedAp = np.array(percErasedAp).mean()

			meanPercErasedAp = comm.bcast(meanPercErasedAp,root=0)
			percErasedAp = comm.bcast(percErasedAp,root=0)
			return meanPercErasedAp,percErasedAp
		else: raise(Exception("The selected cell is not and afferent fiber!"))
