from mpi4py import MPI
from neuron import h
from Simulation import Simulation
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



comm = MPI.COMM_WORLD
sizeComm = comm.Get_size()
rank = comm.Get_rank()

class CellsRecording(Simulation):
	""" Record cells membrane potential over time. """

	def __init__(self, parallelContext, cells, modelType, tStop = 100):
		""" Object initialization.

		Keyword arguments:
		parallelContext -- Neuron parallelContext object.
		cells -- dict containing lists of the objects we want to record (either all artificial cells or segments
			of real cells).
		modelType -- dictionary containing the model types ('real' or 'artificial') for every
			list of cells in cells.
		tStop -- Time in ms at wich the simulation will stop (default = 100).
		"""

		Simulation.__init__(self, parallelContext)

		if rank==1:
			print "\nWarning: mpi execution in this simulation is not supported and therfore useless."
			print "Only the results of the first process are considered...\n"

		self._cells = cells
		self._modelType = modelType
		self._set_tstop(tStop)
		# To plot the results with a high resolution we use an integration step equal to the Neuron dt
		self._set_integration_step(h.dt)

	"""
	Redefinition of inherited methods
	"""
	def _initialize(self):
		Simulation._initialize(self)
		# Initialize rec list
		self._initialize_states()

	def _update(self):
		""" Update simulation parameters. """
		for cellName in self._cells:
			if self._modelType[cellName] == "real":
				for i,cell in enumerate(self._cells[cellName]):
					self._states[cellName][i].append(cell(0.5).v)
			elif self._modelType[cellName] == "artificial":
				for i,cell in enumerate(self._cells[cellName]):
					self._states[cellName][i].append(cell.cell.M(0))

	def save_results(self):
		""" Save the simulation results. """
		print "Not implemented...use the plot method to visualize and save the plots"

	def plot(self,name="",title="",block=True):
		""" Plot the simulation results. """
		if rank == 0:

			fig = plt.figure(figsize=(16,7))
			fig.suptitle(title)
			gs = gridspec.GridSpec(self._nCells,1)
			gs.update(left=0.1, right=0.95, top=0.9, bottom=0.1, wspace=0.2,hspace=0.2)
			ax = []

			cmap = plt.get_cmap('autumn')

			colors = cmap(np.linspace(0.1,0.9,self._nCells))
			for i,cellName in enumerate(self._states):
				for state in self._states[cellName]:
					ax.append(plt.subplot(gs[i]))
					ax[-1].plot(np.linspace(0,self._get_tstop(),len(state)),state,color=colors[i])
					ax[-1].set_ylabel('membrane state ')
					ax[-1].set_title(cellName)
			ax[-1].set_xlabel('Time (ms)')


			fileName = time.strftime("%Y_%m_%d_CellsRecording_"+name+".pdf")
			print self._resultsFolder+fileName
			plt.savefig(self._resultsFolder+fileName, format="pdf",transparent=True)
			plt.show(block=block)


	"""
	Specific Methods of this class
	"""

	def _initialize_states(self):
		self._states = {}
		self._nCells = len(self._cells.keys())
		for cellName in self._cells:
			self._states[cellName] = []
			for cell in self._cells[cellName]:
				 self._states[cellName].append([])
