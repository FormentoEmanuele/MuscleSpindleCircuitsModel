from mpi4py import MPI
from neuron import h
import numpy as np
from scipy import interpolate
from cells import Motoneuron
from cells import IntFireMn
from cells import AfferentFiber
import random as rnd
import time
from tools import firings_tools as tlsf
from tools import seed_handler as sh
sh.set_seed()

comm = MPI.COMM_WORLD
sizeComm = comm.Get_size()
rank = comm.Get_rank()

class EES():
	""" Epidural Electrical Stimulation model. """

	def __init__(self,parallelContext,neuralNetwork,amplitude,frequency,pulsesNumber=100000,species="rat"):
		""" Object initialization.

		Keyword arguments:
		parallelContext -- Neuron parallelContext object.
		neuralNetwork -- NeuralNetwork instance to connect to this object.
		amplitude -- Aplitude of stimulation. It could either be an integer
		value between _minCur and _maxCur or a list containing the percentages
		of recruited primary afferents, secondary afferents and motoneurons.
		frequency -- Stimulation frequency in Hz; it has to be lower than the
		maximum stimulation frequency imposed by the AfferentFiber model.
		pulsesNumber -- number of pulses to send (default 100000).
		species -- rat or human (it loads different recruitment curves)
		"""
		self._pc = parallelContext
		self._nn = neuralNetwork
		self._species = species
		# Lets assign an Id to the stim object (high value to be sure is not already take from some cell)
		self._eesId = 1000000
		# Initialize a dictionary to contain all the connections between this object and the stimulated cells
		self._connections = {}

		self._maxFrequency = AfferentFiber.get_max_ees_frequency()
		self._current = None
		self._percIf= None
		self._percIIf= None
		self._percMn = None

		# Create the netStim Object in the first process
		if rank==0:
			# Tell this host it has this cellId
			self._pc.set_gid2node(self._eesId, rank)

			# Create the stim objetc
			self._stim = h.NetStim()
			self._stim.number = pulsesNumber
			self._stim.start = 5 #lets give few ms for init purposes
			self._stim.noise = 0

			self._pulses = h.Vector()
			# Associate the cell with this host and id
			# the nc is also necessary to use this cell as a source for all other hosts
			nc = h.NetCon(self._stim,None)
			self._pc.cell(self._eesId, nc)
			# Record the stimulation pulses
			nc.record(self._pulses)

			# Load the recruitment data
			self._load_rec_data()

		# lets define which type of cells are recruited by the stimulation
		self._recruitedCells = sum([self._nn.get_afferents_names(),self._nn.get_motoneurons_names()],[])

		# Connect the stimulation to all muscles of the neural network
		self._connect_to_network()
		# Set stimulation parameters
		self.set_amplitude(amplitude)
		self.set_frequency(frequency)

	def __del__(self):
		self._pc.gid_clear() # It removes also the gid of the network...

	def _load_rec_data(self):
		""" Load recruitment data from a previosly validated FEM model (Capogrosso et al 2013). """
		if rank==0:
			if self._species == "rat":
				recI_MG=np.loadtxt('../recruitmentData/Rat_GM_full_S1_wire1')
				recII_MG=np.loadtxt('../recruitmentData/Rat_GM_full_ii_S1_wire1')
				recMn_MG=np.loadtxt('../recruitmentData/Rat_MGM_full_S1_wire1')
				recI_TA=np.loadtxt('../recruitmentData/Rat_TA_full_S1_wire1')
				recII_TA=np.loadtxt('../recruitmentData/Rat_TA_full_ii_S1_wire1')
				recMn_TA=np.loadtxt('../recruitmentData/Rat_MTA_full_S1_wire1')
			elif self._species == "human":
				recI_MG=np.loadtxt('../recruitmentData/Human_GM_full_S1_wire1')
				recII_MG=np.loadtxt('../recruitmentData/Human_GM_full_ii_S1_wire1')
				recMn_MG=np.loadtxt('../recruitmentData/Human_MGM_full_S1_wire1')
				recI_TA=np.loadtxt('../recruitmentData/Human_TA_full_S1_wire1')
				recII_TA=np.loadtxt('../recruitmentData/Human_TA_full_ii_S1_wire1')
				recMn_TA=np.loadtxt('../recruitmentData/Human_MTA_full_S1_wire1')

			if max(recI_MG) == 0: allPercIf_GM = recI_MG
			else: allPercIf_GM= recI_MG/max(recI_MG)
			if max(recII_MG) == 0: allPercIIf_GM = recII_MG
			else: allPercIIf_GM = recII_MG/max(recII_MG)
			if max(recMn_MG) == 0: allPercMn_GM  = recMn_MG
			else: allPercMn_GM =  recMn_MG/max(recMn_MG)
			if max(recI_TA) == 0: allPercIf_TA = recI_TA
			else: allPercIf_TA = recI_TA/max(recI_TA)
			if max(recII_TA) == 0: allPercIIf_TA = recII_TA
			else: allPercIIf_TA = recII_TA/max(recII_TA)
			if max(recMn_TA) == 0: allPercMn_TA = recMn_TA
			else: allPercMn_TA = recMn_TA/max(recMn_TA)

			self._minCur = 0 #uA
			self._maxCur = 600 #uA

			nVal = recI_MG.size
			allPercIf= (allPercIf_GM+allPercIf_TA)/2
			allPercIIf= (allPercIIf_GM+allPercIIf_TA)/2
			allPercMn = (allPercMn_GM+allPercMn_TA)/2

			currents = np.linspace(self._minCur,self._maxCur,nVal)
			self._tckIf = interpolate.splrep(currents, allPercIf)
			self._tckIIf = interpolate.splrep(currents, allPercIIf)
			self._tckMn = interpolate.splrep(currents, allPercMn)


	def _connect(self,targetsId,cellType,netconList):
		""" Connect this object to target cells.

		Keyword arguments:
		targetsId -- List with the id of the target cells.
		cellType -- String defining the cell type.
		netconList -- List in which we append the created netCon Neuron objects.
		"""

		delay=1 # delay of the connection
		if cellType in self._nn.get_afferents_names(): weight = AfferentFiber.get_ees_weight()
		elif cellType in self._nn.get_real_motoneurons_names(): weight = Motoneuron.get_ees_weight()
		elif cellType in self._nn.get_intf_motoneurons_names(): weight = IntFireMn.get_ees_weight()
		else: raise Exception("undefined celltype for EES...intfireMn still to be implemented")

		for targetId in targetsId:
			# check whether this id is associated with a cell in this host.
			if not self._pc.gid_exists(targetId): continue

			if cellType in self._nn.get_real_motoneurons_names():
				cell = self._pc.gid2cell(targetId)
				target = cell.create_synapse('ees')
			else: target = self._pc.gid2cell(targetId)

			# create the connections
			nc = self._pc.gid_connect(self._eesId,target)
			nc.weight[0] = weight
			nc.delay = delay
			nc.active(False)
			netconList.append(nc)

	def _connect_to_network(self):
		""" Connect this object to the NeuralNetwork object. """
		# Iterate over all DoFs
		for muscle in self._nn.cellsId:
			self._connections[muscle] = {}
			# Iterate over all type of cells
			for cellType in self._nn.cellsId[muscle]:
				if cellType in self._recruitedCells:
					# Add a list to the dictionary of netcons
					self._connections[muscle][cellType] = []
					# connect the netstim to all these cells
					self._connect(self._nn.cellsId[muscle][cellType],cellType,self._connections[muscle][cellType])
				comm.Barrier()

	def _activate_connections(self,netcons,percentage):
		""" Modify which connections are active. """
		for nc in netcons: nc.active(False)
		nCon = comm.gather(len(netcons),root=0)
		nOn = None
		if rank==0:
			nCon = sum(nCon)
			nOnTot = int(round(percentage*nCon))
			nOn = np.zeros(sizeComm) + nOnTot/sizeComm
			for i in xrange(nOnTot%sizeComm): nOn[i]+=1
		nOn = comm.scatter(nOn, root=0)

		ncIndexes = range(len(netcons))
		rnd.shuffle(ncIndexes)
		for indx in ncIndexes[:int(nOn)]: netcons[indx].active(True)

	def set_amplitude(self,amplitude,muscles=None):
		""" Set the amplitude of stimulation.

		Note that currently all DoFs have the same percentage of afferents recruited.
		Keyword arguments:
		amplitude -- Aplitude of stimulation. It coulde either be an integer
		value between _minCur and _maxCur or a list containing the percentages
		of recruited primary afferents, secondary afferents and motoneurons.
		muscles -- list of muscle names on which the stimulation amplitude is
		modifiel. If no value is specified, none is used and all the amplitude is
		modified on all the network muscles.
		"""

		if rank == 0:
			if isinstance(amplitude,int) or isinstance(amplitude,float):
				if amplitude > self._minCur and amplitude <self._maxCur:
					self._current = amplitude
					self._percIf=  interpolate.splev(amplitude,self._tckIf)
					if self._percIf<0:self._percIf=0
					self._percIIf=  interpolate.splev(amplitude,self._tckIIf)
					if self._percIIf<0:self._percIIf=0
					self._percMn =  interpolate.splev(amplitude,self._tckMn)
					if self._percMn<0:self._percMn=0

				else:
					raise Exception("Current amplitude out of bounds - min = "+str(self._minCur)+"/ max = "+str(self._maxCur))
			elif isinstance(amplitude,list) and len(amplitude)==3:
				self._current = -1
				self._percIf= amplitude[0]
				self._percIIf= amplitude[1]
				self._percMn = amplitude[2]
			else: raise Exception("badly defined amplitude")

		self._current = comm.bcast(self._current,root=0)
		self._percIf = comm.bcast(self._percIf,root=0)
		self._percIIf = comm.bcast(self._percIIf,root=0)
		self._percMn = comm.bcast(self._percMn,root=0)

		if muscles is None: muscles = self._nn.cellsId.keys()
		for muscle in muscles:
			# Iterate over all type of cells
			for cellType in self._nn.cellsId[muscle]:
				if cellType in self._nn.get_primary_afferents_names():
					self._activate_connections(self._connections[muscle][cellType],self._percIf)
				elif cellType in self._nn.get_secondary_afferents_names():
					self._activate_connections(self._connections[muscle][cellType],self._percIIf)
				elif cellType in self._nn.get_motoneurons_names():
					self._activate_connections(self._connections[muscle][cellType],self._percMn)

	def set_frequency(self,frequency):
		""" Set the frequency of stimulation.

		Note that currently all DoFs have the same percentage of afferents recruited.
		Keyword arguments:
		frequency -- Stimulation frequency in Hz; it has to be lower than the
		maximum stimulation frequency imposed by the AfferentFiber model.
		"""
		if rank == 0:
			if frequency>0 and frequency<self._maxFrequency:
				self._frequency = frequency
				self._stim.interval = 1000.0/self._frequency
			elif frequency<=0:
				self._frequency = 0
				self._stim.interval = 10000
			elif frequency>=self._maxFrequency:
				raise(Exception("The stimulation frequency exceeds the maximum frequency imposed by the AfferentFiber model."))

	def get_amplitude(self,printFlag=False):
		""" Return the stimulation amplitude and print it to screen.

		Current bug: if set_amplitude was used with the non default 'muscles' parameter,
		the stimulation amplitude here returned is not valid for the whole network.
		Indeed, this function only returns the most recent amplitude value that was used
		to change the stimulation settings. """

		if rank==0 and printFlag:
			print "The stimulation amplitude is set at: "+str(self._current)+" uA"
			print "\t"+str(int(self._percIf*100))+"% of primary afferents recruited"
			print "\t"+str(int(self._percIIf*100))+"% of secondary afferents recruited"
			print "\t"+str(int(self._percMn*100))+"% of motoneuron recruited"

		return self._current, self._percIf, self._percIIf,	self._percMn

	def get_frequency(self,printFlag=False):
		""" Return the stimulation frequency and print it to screen. """
		frequency = None
		if rank==0:
			if printFlag: print "The stimulation frequency is set at: "+str(self._frequency)+" Hz"
			frequency = int(round(1000./self._stim.interval))
		frequency = comm.bcast(frequency,root=0)
		return frequency

	def get_n_pulses(self):
		""" Return the number of pulses to send at each burst. """
		nPulses = None
		if rank==0: nPulses = self._stim.number
		nPulses = comm.bcast(nPulses,root=0)
		return nPulses

	def get_id(self):
		""" Return the ID of the NetStim object. """
		return self._eesId

	def get_pulses(self,tStop,samplingRate=1000.):
		""" Return the stimulation pulses. """
		# Transform neuorn Vector in pyhton list to return
		if rank==0:
			nPulses = self._pulses.size()
			if not nPulses: return None,None
			pulsesTime = np.array([self._pulses.x[pulseInd] for pulseInd in xrange(int(nPulses))])
			# exctracting the stim pulses array of 0 and 1
			dt = 1000./samplingRate
			pulsesTime = (pulsesTime/dt).astype(int)
			pulses = np.zeros(1+int(tStop/dt))
			pulses[pulsesTime]=1
			return pulsesTime,pulses
		return None,None
