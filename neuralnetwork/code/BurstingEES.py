from mpi4py import MPI
from neuron import h
import numpy as np
from scipy import interpolate
from cells import Motoneuron
from cells import IntFireMn
from cells import AfferentFiber
from EES import EES
import random as rnd
import time
from tools import seed_handler as sh
sh.set_seed()

comm = MPI.COMM_WORLD
sizeComm = comm.Get_size()
rank = comm.Get_rank()

class BurstingEES():
	""" Bursting Epidural Electrical Stimulation model.

		Used to implement High-Frequency Low-Amplitude EES
	"""

	def __init__(self,parallelContext,neuralNetwork,amplitude,frequency,burstsFrequency,pulsesNumber,species="rat"):
		""" Object initialization.

		Keyword arguments:
		parallelContext -- Neuron parallelContext object.
		neuralNetwork -- EES object with the number of pulse and the frequency of the pulses during each burst
		amplitude -- Aplitude of stimulation. It could either be an integer
		value between _minCur and _maxCur or a list containing the percentages
		of recruited primary afferents, secondary afferents and motoneurons.
		frequency -- Stimulation frequency in Hz; It has to be set in a way that new bursts
		are not occuring while old burst are still ongoing.
		burstsFrequency -- Burst stimulation frequency in Hz; it has to be lower than the
		maximum stimulation frequency imposed by the AfferentFiber model.
		pulsesNumber -- number of pulses to send.
		"""

		self._pc = parallelContext
		self._burstStim = EES(parallelContext,neuralNetwork,amplitude,burstsFrequency,pulsesNumber,species)
		self._burstStimId = self._burstStim.get_id()
		self._BurstingEesId = 1000001

		burstPulses = self._burstStim.get_n_pulses()
		burstFreq = self._burstStim.get_frequency()
		totBurstDuration = 1000./burstFreq*burstPulses
		self._maxFrequency = 1000./totBurstDuration

		# Create the netStim Object in the first process
		if rank==0:
			# Tell this host it has this cellId
			self._pc.set_gid2node(self._BurstingEesId, rank)

			# Create the stim objetc
			self._stim = h.NetStim()
			self._stim.number = 100000
			self._stim.start = 5 #lets give few ms for init purposes
			self._stim.noise = 0

			self._pulses = h.Vector()
			# Associate the cell with this host and id
			# the nc is also necessary to use this cell as a source for all other hosts
			nc = h.NetCon(self._stim,None)
			self._pc.cell(self._BurstingEesId, nc)
			# Record the stimulation pulses
			nc.record(self._pulses)

		# Connect the stimulation to all dof of the neural network
		self._connect_to_burstStim()
		# Set stimulation frequency
		self.set_frequency(frequency)

	def __del__(self):
		self._pc.gid_clear() # It removes also the gid of the network...

	def _connect_to_burstStim(self):
		""" Connect this object to the EES object. """
		# check whether this id is associated with a cell in this host.
		if not self._pc.gid_exists(self._burstStimId): return
		delay=1
		weight=1
		target = self._pc.gid2cell(self._burstStimId)
		self._connection = self._pc.gid_connect(self._BurstingEesId,target)
		self._connection.weight[0] = weight
		self._connection.delay = delay

	def set_amplitude(self,amplitude):
		""" Set the amplitude of stimulation.

		Note that currently all DoFs have the same percentage of afferents recruited.
		Keyword arguments:
		amplitude -- Aplitude of stimulation. It coulde either be an integer
		value between _minCur and _maxCur or a list containing the percentages
		of recruited primary afferents, secondary afferents and motoneurons.
		"""
		self._burstStim.set_amplitude(amplitude)

	def set_frequency(self,frequency):
		""" Set the frequency of stimulation.

		Keyword arguments:
		frequency -- Stimulation frequency in Hz; it has to be lower than the
		maximum stimulation frequency imposed by the AfferentFiber model. It also
		has to be set in a way that new bursts are not occuring while old burst
		are still ongoing.
		"""
		if rank == 0:
			if frequency>0 and frequency<self._maxFrequency:
				self._frequency = frequency
				self._stim.interval = 1000.0/self._frequency
			elif frequency<=0:
				self._frequency = 0
				self._stim.interval = 10000
			elif frequency>=self._maxFrequency:
				raise(Exception("The stimulation frequency exceeds the maximum frequency imposed by the burst duration."))

	def set_bursts_frequency(self,frequency):
		""" Set the frequency of stimulation inside the bursts.

		Keyword arguments:
		frequency -- Stimulation frequency in Hz; it has to be lower than the
		maximum stimulation frequency imposed by the AfferentFiber model.
		"""
		self._burstStim.set_frequency(frequency)

	def get_amplitude(self,printFlag=False):
		""" Return the stimulation amplitude and print it to screen. """
		current, percIf, percIIf, percMn = self._burstStim.get_amplitude(printFlag)
		return current, percIf, percIIf, percMn

	def get_frequency(self,printFlag=False):
		""" Return the stimulation frequency and print it to screen. """
		frequency = None
		if rank==0:
			if printFlag: print "The stimulation frequency is set at: "+str(self._frequency)+" Hz"
			frequency = int(round(1000./self._stim.interval))
		frequency = comm.bcast(frequency,root=0)
		return frequency

	def get_bursts_frequency(self,printFlag=False):
		""" Return the stimulation frequency and print it to screen. """
		burstsFrequency = self._burstStim.get_frequency(printFlag)
		return burstsFrequency

	def get_pulses(self):
		""" Return the stimulation pulses. """
		# Transform neuorn Vector in pyhton list to return
		raise Exception("Feature to implement for future functionalities")
