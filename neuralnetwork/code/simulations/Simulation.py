import os
import time
from mpi4py import MPI
from neuron import h

comm = MPI.COMM_WORLD
sizeComm = comm.Get_size()
rank = comm.Get_rank()


class Simulation:
	""" Interface class to design different types of neuronal simulation.

	The simulations are based on the python Neuron module and
	can be executed in parallel using MPI.
	"""

	def __init__(self, parallelContext):
		""" Object initialization.

		Keyword arguments:
		parallelContext -- Neuron parallelContext object.
		"""

		# Define the Neuron ParallelContext for parallel simulations
		self._pc = parallelContext
		# Set the temperature in Neuron
		h.celsius = 37
		# Set the Neuron integration dt in ms
		h.dt = 0.025
		# For parallel simulation in Neuron each host has to communicate to the other hosts the cells
		# synaptic activity. The variable __maxStep define the interval between these communications.
		# This has to be lower then the minimum delay between two cells connections.
		self.__maxStep = 0.5

		# variables to set in child clasess
		self.__tstop = None
		self.__integrationStep = None

		if rank==0:
			self.simulationTime = None
			self.__printPeriod = 250 #ms

			self._resultsFolder = "../../results/"
			if not os.path.exists(self._resultsFolder):
				os.makedirs(self._resultsFolder)

	def __del__(self):
		""" Object destruction. """
		# The master host returns immediately. Worker hosts start an infinite loop of requesting tasks for execution.
		self._pc.runworker()
		# Send a QUIT message to all worker hosts
		self._pc.done()

	def __check_parameters(self):
		""" Check whether some parameters necessary for the simulation have been set or not. """
		if self.__tstop == None or self.__integrationStep == None:
			raise Exception("Undefined integration step and maximum time of simulation")

	def _get_tstop(self):
		""" Return the time at which we want to stop the simulation. """
		return self.__tstop

	def _set_tstop(self,tstop):
		""" Set the time at which we want to stop the simulation.

		Keyword arguments:
		tstop -- time at which we want to stop the simulation in ms.
		"""
		if tstop>0: self.__tstop = tstop
		else: raise Exception("The maximum time of simulation has to be greater than 0")

	def _get_integration_step(self):
		""" Return the integration time step. """
		return self.__integrationStep

	def _set_integration_step(self,dt):
		""" Set the integration time step.

		Keyword arguments:
		dt -- integration time step in ms.
		"""
		if dt>0: self.__integrationStep = dt
		else: raise Exception("The integration step has to be greater than 0")

	def _initialize(self):
		""" Initialize the simulation.

		Set the __maxStep varibale and initialize the membrane potential of real cell to -70mV.
		"""
		self._pc.set_maxstep(self.__maxStep)
		h.finitialize(-69.35)
		if rank==0:
			self._start = time.time()
			self.__tPrintInfo=0

	def _integrate(self):
		""" Integrate the neuronal cells for a defined integration time step ."""
		self._pc.psolve(h.t+self.__integrationStep)

	def _update(self):
		""" Update simulation parameters. """
		raise Exception("pure virtual function")

	def _get_print_period(self):
		""" Return the period of time between printings to screen. """
		return self.__printPeriod

	def _set_print_period(self,t):
		""" Set the period of time between printings to screen.

		Keyword arguments:
		t -- period of time between printings in ms.
		"""
		if t>0: self.__printPeriod = t
		else: raise Exception("The print period has to be greater than 0")

	def _print_sim_status(self):
		""" Print to screen the simulation state. """
		if rank == 0:
			if h.t-self.__tPrintInfo>=(self.__printPeriod-0.5*self.__integrationStep):
				if self.__tPrintInfo == 0:
					print "\nStarting simulation:"
				self.__tPrintInfo=h.t
				print "\t"+str(round(h.t))+"ms of "+str(self.__tstop)+"ms integrated..."

	def _end_integration(self):
		""" Print the total simulation time.

		This function, executed at the end of time integration is ment to be modified
		by daughter calsses according to specific needs.
		"""
		if rank==0:
			self.simulationTime = time.time()-self._start
			print "tot simulation time: "+ str(int(self.simulationTime)) + "s"

	def run(self):
		""" Run the simulation. """
		self.__check_parameters()
		self._initialize()
		while h.t<self.__tstop:
			self._integrate()
			self._update()
			self._print_sim_status()
		self._end_integration()

	def set_results_folder(self,resultsFolderPath):
		""" Set a new folder in which to save the results """
		self._resultsFolder = resultsFolderPath
		if not os.path.exists(self._resultsFolder):
			os.makedirs(self._resultsFolder)

	def save_results(self,name=""):
		""" Save the simulation results.

		Keyword arguments:
		name -- string to add at predefined file name (default = "").
		"""
		raise Exception("pure virtual function")

	def plot(self):
		""" Plot the simulation results. """
		raise Exception("pure virtual function")
