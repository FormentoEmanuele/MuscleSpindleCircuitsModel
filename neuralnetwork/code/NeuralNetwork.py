from mpi4py import MPI
from neuron import h
from cells import Motoneuron
from cells import IntFireMn
from cells import IntFire
from cells import AfferentFiber
import random as rnd
import time
import numpy as np
from tools import seed_handler as sh
sh.set_seed()

comm = MPI.COMM_WORLD
sizeComm = comm.Get_size()
rank = comm.Get_rank()

class NeuralNetwork():
	""" Spiking neural network model.

	Model of a spiking neural network that can be built in parallel hosts using MPI.
	"""

	def __init__(self, parallelContext, inputFile):
		""" Object initialization.

		Keyword arguments:
		parallelContext -- Neuron parallelContext object.
		inputFile -- txt file specifying the neural network structure.
		"""

		self._pc = parallelContext
		if rank==0: self._inputFile = inputFile

		# Initialize the flags to decide which neurons to record.
		self.recordMotoneurons = True
		self.recordAfferents = True
		self.recordIntFire = True

		# Initialize the lists containing the names of the different types of cells.
		self._realMotoneuronsNames = []
		self._intMotoneuronsNames = []
		self._motoneuronsNames = []
		self._primaryAfferentsNames = []
		self._secondaryAfferentsNames = []
		self._afferentsNames = []
		self._interNeuronsNames = []

		self._connections = []
		# Build the neural network
		self._read()
		self._init_dictionaries()
		self._create_cells()
		self._create_common_connections()
		self._create_inter_muscles_sensorimotor_connections()
		self._create_special_connections()

	def __del__(self):
		""" Object destruction and clean all gid references. """
		self._pc.gid_clear()

	def _read(self):
		""" Define the neural network structure from the input file. """
		self._infoMuscles = []
		self._infoCommonCellsInMuscles = []
		self._infoSpecialCells = []
		self._infoCommonMuscleConnections = []
		self._infoInterMuscSensorimotorConnections = {}
		self._infoSpecialConnections = []
		if rank==0:
			section = None
			sensorimotorConnections = None
			sensorimotorMatrix = None
			for line in open("../nnStructures/"+self._inputFile,"r"):
				if line[0] == "#" or line[0] == "\n": continue
				elif line[0] == "@": section = float(line[1])
				elif section == 1: self._infoMuscles.append(line.strip("\n").split())
				elif section == 2: self._infoCommonCellsInMuscles.append(line.strip("\n").split())
				elif section == 3: self._infoSpecialCells.append(line.strip("\n").split())
				elif section == 4: self._infoCommonMuscleConnections.append(line.strip("\n").split())
				elif section == 5:
					if line[0] == "+":
						dictName = line[1:].strip("\n")
						self._infoInterMuscSensorimotorConnections[dictName] = {}
						sensorimotorConnections = False
						sensorimotorMatrix = False
					elif "Connections" in line:
						 sensorimotorConnections = True
						 self._infoInterMuscSensorimotorConnections[dictName]["connections"]=[]
					elif "WeightsMatrix" in line:
						 sensorimotorConnections = False
						 sensorimotorMatrix = True
						 self._infoInterMuscSensorimotorConnections[dictName]["matrix"]=[]
					elif sensorimotorConnections:
						self._infoInterMuscSensorimotorConnections[dictName]["connections"].append(line.strip("\n").split())
					elif sensorimotorMatrix:
						self._infoInterMuscSensorimotorConnections[dictName]["matrix"].append(line.strip("\n").split())
				elif section == 6: self._infoSpecialConnections.append(line.strip("\n").split())

		self._infoMuscles = comm.bcast(self._infoMuscles,root=0)
		self._infoCommonCellsInMuscles = comm.bcast(self._infoCommonCellsInMuscles,root=0)
		self._infoSpecialCells = comm.bcast(self._infoSpecialCells,root=0)
		self._infoCommonMuscleConnections = comm.bcast(self._infoCommonMuscleConnections,root=0)
		self._infoInterMuscSensorimotorConnections = comm.bcast(self._infoInterMuscSensorimotorConnections,root=0)
		self._infoSpecialConnections = comm.bcast(self._infoSpecialConnections,root=0)

	def _init_dictionaries(self):
		""" Initialize all the dictionaries contatining cells, cell ids and the recorded action potentials. """
		# Dictionary contatining all actionPotential
		self.actionPotentials = {}
		# Dictionary containing all cells id.
		# Cells id are used by neuron to communicate synapses between different cells in different hosts. Ids (gids) can be any integer, they just need to be unique.
		self.cellsId = {}
		# Dictionary containing all cells
		self.cells = {}

		self._nMuscles = len(self._infoMuscles)
		for muscle,muscAfferentDelay in self._infoMuscles:
			# Create sub-dictionaries for all DoF
			self.actionPotentials[muscle]={}
			self.cellsId[muscle]={}
			self.cells[muscle]={}
			for cellInfo in self._infoCommonCellsInMuscles:
				# add lists containing cell ids/cells/ap
				cellClass = cellInfo[0]
				cellName = cellInfo[1]
				self.cellsId[muscle][cellName]=[]
				self.cells[muscle][cellName]=[]
				if (cellClass=="Motoneuron" or cellClass=="IntFireMn") and self.recordMotoneurons:
					self.actionPotentials[muscle][cellName]=[]
				elif cellClass=="AfferentFiber" and self.recordAfferents:
					self.actionPotentials[muscle][cellName]=[]
				elif cellClass=="IntFire" and self.recordIntFire:
					self.actionPotentials[muscle][cellName]=[]

		# Add special cells (specifc for some muscles or not muscle related)
		for cellInfo in self._infoSpecialCells:
			groupOrMuscle = cellInfo[0]
			cellClass = cellInfo[1]
			cellName = cellInfo[2]
			if not groupOrMuscle in self.cellsId.keys():
				self.actionPotentials[groupOrMuscle]={}
				self.cellsId[groupOrMuscle]={}
				self.cells[groupOrMuscle]={}

			self.cellsId[groupOrMuscle][cellName]=[]
			self.cells[groupOrMuscle][cellName]=[]
			if (cellClass=="Motoneuron" or cellClass=="IntFireMn") and self.recordMotoneurons:
				self.actionPotentials[groupOrMuscle][cellName]=[]
			elif cellClass=="AfferentFiber" and self.recordAfferents:
				self.actionPotentials[groupOrMuscle][cellName]=[]
			elif cellClass=="IntFire" and self.recordIntFire:
				self.actionPotentials[groupOrMuscle][cellName]=[]

	def _create_cells(self):
		""" Create the desired cells and assign them a unique cell Id. """
		cellId=0
		# Iterate over all dictionaries
		for muscle,muscAfferentDelay in self._infoMuscles:
			for cellInfo in self._infoCommonCellsInMuscles:
				cellClass = cellInfo[0]
				cellName = cellInfo[1]
				cellNumber = cellInfo[2]
				if len(cellInfo)>=4: neuronParam = cellInfo[3]
				else: neuronParam = None
				cellId = self._create_cell_population(cellId,muscle,muscAfferentDelay,cellClass,cellName,cellNumber,neuronParam)
		# Add special cells
		for cellInfo in self._infoSpecialCells:
			groupOrMuscle = cellInfo[0]
			cellClass = cellInfo[1]
			cellName = cellInfo[2]
			cellNumber = cellInfo[3]
			if len(cellInfo)>=5: neuronParam = cellInfo[4]
			else: neuronParam = None
			muscAfferentDelay = None
			cellId = self._create_cell_population(cellId,groupOrMuscle,muscAfferentDelay,cellClass,cellName,cellNumber,neuronParam)

		self._motoneuronsNames = self._intMotoneuronsNames+self._realMotoneuronsNames
		self._afferentsNames = self._primaryAfferentsNames+self._secondaryAfferentsNames

	def _create_cell_population(self,cellId,muscle,muscAfferentDelay,cellClass,cellName,cellNumber,neuronParam=None):
		""" Create cells populations. """
		for n in range(int(cellNumber)):
			# Lets divide equally the cells between the different hosts
			if n%sizeComm==rank:
				# Assign a cellId to the new cell
				self.cellsId[muscle][cellName].append(cellId)
				# Tell this host it has this cellId
				self._pc.set_gid2node(cellId, rank)
				# Create the cell
				if cellClass=="IntFireMn":
					#List containing all integrate and fire motoneurons names
					if not cellName in self._intMotoneuronsNames: self._intMotoneuronsNames.append(cellName)
					self.cells[muscle][cellName].append(IntFireMn())
				elif cellClass=="Motoneuron":
					#List containing all realistic motoneurons names
					if not cellName in self._realMotoneuronsNames: self._realMotoneuronsNames.append(cellName)
					# durg - parameter specific to the Mn
					drug=False
					if neuronParam=="drug":drug=True
					self.cells[muscle][cellName].append(Motoneuron(drug))
				elif cellClass=="AfferentFiber":
					#Lists containing all primary or secondary afferent fibers names
					if "II" in cellName:
						if not cellName in self._secondaryAfferentsNames: self._secondaryAfferentsNames.append(cellName)
					else:
						if not cellName in self._primaryAfferentsNames: self._primaryAfferentsNames.append(cellName)
					# delay - parameter specific for the Afferent fibers
					if neuronParam is not None: delay = int(neuronParam)
					elif muscAfferentDelay is not None: delay = int(muscAfferentDelay)
					else: raise Exception("Please specify the afferent fiber delay")
					self.cells[muscle][cellName].append(AfferentFiber(delay))
				elif cellClass=="IntFire":
					#List containing all interneurons names
					if not cellName in self._interNeuronsNames: self._interNeuronsNames.append(cellName)
					self.cells[muscle][cellName].append(IntFire())
				else:
					raise Exception("Unkown cell in the netowrk instructions.... ("+str(cellClass)+")")
				# Associate the cell with this host and id, the nc is also necessary to use this cell as a source for all other hosts
				nc = self.cells[muscle][cellName][-1].connect_to_target(None)
				self._pc.cell(cellId, nc)
				# Record cells APs
				if (cellClass=="Motoneuron" or cellClass=="IntFireMn") and self.recordMotoneurons:
					self.actionPotentials[muscle][cellName].append(h.Vector())
					nc.record(self.actionPotentials[muscle][cellName][-1])
				elif cellClass=="AfferentFiber" and self.recordAfferents:
					self.actionPotentials[muscle][cellName].append(h.Vector())
					nc.record(self.actionPotentials[muscle][cellName][-1])
				elif cellClass=="IntFire" and self.recordIntFire:
					self.actionPotentials[muscle][cellName].append(h.Vector())
					nc.record(self.actionPotentials[muscle][cellName][-1])
			cellId+=1
		return cellId

	def _connect(self,sourcesId,targetsId,conRatio,conNum,conWeight,synType, conDelay=1):
		""" Connect source cells to target cells.

		Keyword arguments:
		sourcesId -- List with the id of the source cells.
		targetsId -- List with the id of the target cells.
		conRatio -- Define how the source cells are connected to the target cells;
		It can be either "unique"  or "random". With "unique" every source cell is connected
		to every target cell, while with "random" every target cell is connected to n=conNum
		randomly selected source cells.
		conNum -- Number of source cells connected to every target cell. Note that with "unique"
		conRation this parameter is still mandatory for double checking.
		conWeight -- Connection weight.
		synType -- Type of synapse that form the connection. It could be either "artificial" for
		artificial cells or "excitatory"/"inhibitory" for realistic cell models.
		conDelay -- Delay of the synapse in ms (default = 1).
		"""
		noisePerc = 0.2
		for targetId in targetsId:
			# check whether this id is associated with a cell in this host
			if not self._pc.gid_exists(targetId): continue
			if conRatio == "unique" and len(sourcesId)!=conNum:
				raise Exception("Wrong connections number parameter. If the synapses ratio is 'unique' the number of synapses has to be the same as the number of source cells")
			# retrieve the target for artificial cells
			if synType == "artificial":
				target = self._pc.gid2cell(targetId)
			# retrieve the cell for realistic cells
			elif synType == "excitatory" or synType == "inhibitory":
				cell = self._pc.gid2cell(targetId)
			else: raise Exception("Wrong synType")

			# create the connections
			for i in range(conNum):
				# create the target for realistic cells
				if synType == "excitatory" or synType == "inhibitory": target = cell.create_synapse(synType)
				if conRatio == "unique": source = sourcesId[i]
				elif conRatio == "random": source = rnd.choice(sourcesId)
				else : raise Exception("Wrong connections ratio parameter")
				nc = self._pc.gid_connect(source,target)
				nc.weight[0] = rnd.normalvariate(conWeight,conWeight*noisePerc)
				nc.delay = conDelay+rnd.normalvariate(0.25,0.25*noisePerc)
				self._connections.append(nc)
		comm.Barrier()

	def _create_common_connections(self):
		""" Connect network cells within the same degree of freedom. """
		for muscle,muscAfferentDelay in self._infoMuscles:
			for connection in self._infoCommonMuscleConnections:
				# List of source cells ids
				sourcesId = self.cellsId[muscle][connection[0]]
				# gather the sources all together
				sourcesId = comm.gather(sourcesId,root=0)
				if rank==0: sourcesId = sum(sourcesId,[])
				sourcesId = comm.bcast(sourcesId,root=0)
				# List of taget cells ids
				targetsId = self.cellsId[muscle][connection[1]]
				# Ratio of connection
				conRatio = connection[2]
				# Number of connections
				conNum = int(connection[3])
				# Weight of connections
				conWeight = float(connection[4])
				# Type of synapse
				synType = connection[5]
				# connect sources to targets
				self._connect(sourcesId,targetsId,conRatio,conNum,conWeight,synType)

	def _create_inter_muscles_sensorimotor_connections(self):
		""" Create sensorimotor connections between muscles."""

		for pathway in self._infoInterMuscSensorimotorConnections:
			connections = self._infoInterMuscSensorimotorConnections[pathway]["connections"]
			matrix = self._infoInterMuscSensorimotorConnections[pathway]["matrix"]
			if not len(matrix)-1 == len(matrix[0])-1 == len(self._infoMuscles):
				raise(Exception("The weight matrix has to be nMuscles x nMuscles."))
			# The first raw is a header
			for M2weights,M1 in zip(matrix[1:],self._infoMuscles):
				for weight,M2 in zip(M2weights[1:],self._infoMuscles):
					if not float(weight) == 0:
						if M1[0] is M2[0]: raise(Exception("Intra muscle sensorimotor conncetions have to be implemented in section 4."))
						for connection in connections:
							# List of source cells ids
							sourcesId = self.cellsId[M1[0]][connection[1]]
							# gather the sources all together
							sourcesId = comm.gather(sourcesId,root=0)
							if rank==0: sourcesId = sum(sourcesId,[])
							sourcesId = comm.bcast(sourcesId,root=0)
							# List of taget cells ids
							targetsId = self.cellsId[M2[0]][connection[3]]
							# Ratio of connection
							conRatio = connection[4]
							# Number of connections
							conNum = int(int(connection[5])*float(weight))
							# Weight of connections
							conWeight = float(connection[6])
							# Type of synapse
							synType = connection[7]
							# connect sources to targets
							self._connect(sourcesId,targetsId,conRatio,conNum,conWeight,synType)

	def _create_special_connections(self):
		""" Create connections specific to single muscles or cell groups. """
		for connection in self._infoSpecialConnections:
			# List of source cells ids
			sourcesId = self.cellsId[connection[0]][connection[1]]
			# gather the sources all together
			sourcesId = comm.gather(sourcesId,root=0)
			if rank==0: sourcesId = sum(sourcesId,[])
			sourcesId = comm.bcast(sourcesId,root=0)
			# List of taget cells ids
			targetsId = self.cellsId[connection[2]][connection[3]]
			# Ratio of connection
			conRatio = connection[4]
			# Number of connections
			conNum = int(connection[5])
			# Weight of connections
			conWeight = float(connection[6])
			# Type of synapse
			synType = connection[7]
			# connect sources to targets
			self._connect(sourcesId,targetsId,conRatio,conNum,conWeight,synType)

	def update_afferents_ap(self,time):
		""" Update all afferent fibers ation potentials. """
		# Iterate over all dictionaries
		for muscle in self.cells:
			for cellName in self.cells[muscle]:
				if cellName in self._afferentsNames:
					for cell in self.cells[muscle][cellName]:
						cell.update(time)

	def set_afferents_fr(self,fr):
		""" Set the firing rate of the afferent fibers.

		Keyword arguments:
		fr -- Dictionary with the firing rate in Hz for the different cellNames.
		"""
		# Iterate over all dictionaries
		for muscle in self.cells:
			for cellName in self.cells[muscle]:
				if cellName in self._afferentsNames:
					for cell in self.cells[muscle][cellName]:
						cell.set_firing_rate(fr[muscle][cellName])

	def initialise_afferents(self):
		""" Initialise cells parameters. """
		# Iterate over all dictionaries
		for muscle in self.cells:
			for cellName in self.cells[muscle]:
				if cellName in self._afferentsNames:
					for cell in self.cells[muscle][cellName]:cell.initialise()

	def get_ap_number(self, cellNames):
		""" Return the number of action potentials fired for the different recorded cells.

		The number of Ap is returned only to the main process (rank=0).
		Keyword arguments:
		cellNames -- List of cell names from wich we want to get the number of action potentials. """

		apNumber = {}
		for muscle in self.cells:
			apNumber[muscle] = {}
			for cellName in cellNames:
				if (cellName in self._afferentsNames and self.recordAfferents) \
				or (cellName in self._motoneuronsNames and self.recordMotoneurons) \
				or (cellName in self._interNeuronsNames and self.recordIntFire):
					apNumber[muscle][cellName] = []
					for apVector in self.actionPotentials[muscle][cellName]:
						apNumber[muscle][cellName].append(apVector.size())
				else: raise(Exception("Cell name not found in the NeuralNetwork"))

				if sizeComm<=1: continue

				tempApNumberAll = comm.gather(apNumber[muscle][cellName], root=0)
				if rank==0:
					apNumber[muscle][cellName] = np.concatenate([tempApNumberAll[0],tempApNumberAll[1]])
					for i in xrange(2,sizeComm):
						apNumber[muscle][cellName] = np.concatenate([apNumber[muscle][cellName],tempApNumberAll[i]])
				else: apNumber[muscle][cellName]=None

		return apNumber

	def get_afferents_names(self):
		""" Return the afferents name. """
		return self._afferentsNames

	def get_primary_afferents_names(self):
		""" Return the primary afferents name. """
		return self._primaryAfferentsNames

	def get_secondary_afferents_names(self):
		""" Return the secondary afferents name. """
		return self._secondaryAfferentsNames

	def get_real_motoneurons_names(self):
		""" Return the real motoneurons name. """
		return self._realMotoneuronsNames

	def get_intf_motoneurons_names(self):
		""" Return the int fire name. """
		return self._intMotoneuronsNames

	def get_motoneurons_names(self):
		""" Return the motoneurons names. """
		return self._motoneuronsNames

	def get_interneurons_names(self):
		""" Return the inteurons names. """
		return self._interNeuronsNames

	def get_mn_info(self):
		""" Return the connection informations. """
		return self._infoCommonMuscleConnections, self._infoSpecialConnections
