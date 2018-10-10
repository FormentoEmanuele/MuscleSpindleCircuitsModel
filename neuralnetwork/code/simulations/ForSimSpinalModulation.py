from mpi4py import MPI
from neuron import h
from ForwardSimulation import ForwardSimulation
from CellsRecording import CellsRecording
from cells import AfferentFiber
import random as rnd
import time
import numpy as np
from tools import general_tools  as gt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tools import firings_tools as tlsf
import pickle
from tools import seed_handler as sh
sh.set_seed()


comm = MPI.COMM_WORLD
sizeComm = comm.Get_size()
rank = comm.Get_rank()

class ForSimSpinalModulation(ForwardSimulation,CellsRecording):
	""" Integration of a NeuralNetwork object over time given an input.
		The simulation results are the cells membrane potential over time.
	"""

	def __init__(self, parallelContext, neuralNetwork, cells, modelType, afferentInput=None, eesObject=None, eesModulation=None, tStop = 100):
		""" Object initialization.

		Keyword arguments:
		parallelContext -- Neuron parallelContext object.
		neuralNetwork -- NeuralNetwork object.
		cells -- dict containing cells list (or node lists for real cells) from which we record the membrane potetntials.
		modelType -- dictionary containing the model types ('real' or 'artificial') for every
			list of cells in cells.
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

		if rank==1:
			print "\nWarning: mpi execution in this simulation is not supported and therfore useless."
			print "Only the results of the first process are considered...\n"
		CellsRecording.__init__(self, parallelContext, cells, modelType, tStop)

		ForwardSimulation.__init__(self,parallelContext, neuralNetwork, afferentInput, eesObject, eesModulation, tStop)
		self._set_integration_step(h.dt)

	"""
	Redefinition of inherited methods
	"""

	def _initialize(self):
		ForwardSimulation._initialize(self)
		CellsRecording._initialize(self)

	def _update(self):
		""" Update simulation parameters. """
		CellsRecording._update(self)
		ForwardSimulation._update(self)

	def plot(self,muscle,cells,name=""):
		""" Plot the simulation results. """
		if rank==0:
			fig, ax = plt.subplots(figsize=(16,7))
			ax.plot(self._meanFr[muscle][cells])
			ax.set_title('Cells mean firing rate')
			ax.set_ylabel(" Mean firing rate (Hz)")
			fileName = time.strftime("%Y_%m_%d_CellsRecordingMeanFR_"+name+".pdf")
			plt.savefig(self._resultsFolder+fileName, format="pdf",transparent=True)
			CellsRecording.plot(self,name)

	def raster_plot(self,name="",plot=True):
		if rank==0:
			sizeFactor = 0.5
			colorMap = plt.cm.gray
			colorMap.set_over("#0792fd")
			colorMap.set_under("#2C3E50")

			cellsGroups = gt.naive_string_clustering(self._states.keys())
			for cellNameList in cellsGroups:
				cellNameList.sort()
				cellNameList.reverse()
				cellClusterName = "_".join(cellNameList)
				states = self._states[cellNameList[0]]
				for i in xrange(1,len(cellNameList)): states = np.concatenate((states,self._states[cellNameList[i]]))
				fig, ax = plt.subplots(figsize=(40*sizeFactor,10*sizeFactor))
				im = ax.imshow(states, cmap=colorMap, interpolation='nearest',origin="lower",vmin = -0, vmax = 1,aspect='auto')
				ax.set_title("Raster plot: "+name)

				# Move left and bottom spines outward by 10 points
				ax.spines['left'].set_position(('outward', 10))
				ax.spines['bottom'].set_position(('outward', 10))
				# Hide the right and top spines
				ax.spines['right'].set_visible(False)
				ax.spines['top'].set_visible(False)
				# Only show ticks on the left and bottom spines
				ax.yaxis.set_ticks_position('left')
				ax.xaxis.set_ticks_position('bottom')
				fig.colorbar(im, orientation='vertical',label=cellClusterName+' membrane state')

				dt = self._get_integration_step()
				tStop = self._get_tstop()
				plt.xticks(np.arange(0,(tStop+1)/dt,25/dt),range(0,int(tStop+1),25))
				plt.ylabel(cellClusterName)
				plt.xlabel('Time (ms)')

				fileName = time.strftime("%Y_%m_%d_raster_plot_"+name+"_"+cellClusterName+".pdf")
				plt.savefig(self._resultsFolder+fileName, format="pdf",transparent=True)

			plt.show(block=plot)

	def plot_membrane_potatial(self,name="",title="",block=False):
		CellsRecording.plot(self,name,title,block)

	def save_results(self,name):
		""" Save the simulation results. """
		if rank == 0:
			fileName = time.strftime("%Y_%m_%d_FSSM_nSpikes")+name+".p"
			with open(self._resultsFolder+fileName, 'w') as pickle_file:
				pickle.dump(self._nSpikes, pickle_file)
				pickle.dump(self._nActiveCells, pickle_file)
