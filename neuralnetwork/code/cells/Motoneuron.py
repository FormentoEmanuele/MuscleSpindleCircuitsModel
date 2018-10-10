from Cell import Cell
from neuron import h
import random as rnd
import time
import numpy as np
from tools import seed_handler as sh
sh.set_seed()

class Motoneuron(Cell):
	""" Neuron Biophysical rat motoneuron model.

	The model integrates an axon model developed by Richardson et al. 2000,
	a model of the soma and of the dendritic tree developed by McIntyre 2002.
	The geometry is scaled to match the rat dimension.
	This model offers also the possibility to simulate the effect of 5-HT as in Booth et al. 1997.
	"""
	__eesWeight = 20 # Weight of a connection between an ees object and this cell

	def __init__(self,drug=True):
		""" Object initialization.

		Keyword arguments:
		drug -- A boolean flag that is used to decide whether 5-HT is
		inserted in the model or not (default = True).
		"""

		Cell.__init__(self)

		# Define parameters
		self._drug = drug
		self.synapses = []
		self._nNodes = 41
		self._nDendrites = 12
		self._diamAxon = 8.74

		# dendrite diameters scaled for the rat
		self._diamDend = [1.5277386855568675e+01, 1.7434194411648960e+01, 1.9770735930735935e+01,
						1.7973396300669030e+01, 1.7613928374655650e+01, 1.6355790633608819e+01,
						1.4378717040535225e+01, 1.2940845336481701e+01, 1.0424569854388038e+01,
						8.5373632428177881e+00, 6.7400236127508864e+00, 3.1453443526170801e+00]
		# dendrite length
		self._lengthDend = [5.0e+02, 5.0e+02, 7.0e+02,
							9.0e+02, 5.0e+02, 4.0e+02,
							6.0e+02, 4.0e+02, 5.0e+02,
							5.0e+02, 5.0e+02, 5.0e+02]

		self._create_sections()
		self._define_biophysics()
		self._build_topology()


	"""
	Specific Methods of this class
	"""

	def _create_sections(self):
		""" Create the sections of the cell. """
		# NOTE: cell=self is required to tell NEURON of this object.
		self.soma = h.Section(name='soma',cell=self)
		self.dendrite = [h.Section(name='dendrite',cell=self) for x in range(self._nDendrites)]
		self.initSegment = h.Section(name='soma',cell=self)
		self.node = [h.Section(name='node',cell=self) for x in range(self._nNodes)]
		self.paranode = [h.Section(name='paranode',cell=self) for x in range(self._nNodes)]

	def _define_biophysics(self):
		""" Assign geometry and membrane properties across the cell. """
		self.soma.nseg = 1
		self.soma.L = 36
		self.soma.diam = 36
		self.soma.cm = 2
		self.soma.Ra = 200
		self.soma.insert('motoneuron') # Insert the Neuron motoneuron mechanism developed by McIntyre 2002
		if self._drug: self.soma.gcak_motoneuron *= 0.6 # Add the drug effect as in Booth et al 1997

		self.initSegment.nseg = 5
		self.initSegment.L = 1000
		self.initSegment.diam = 10
		self.initSegment.insert('initial') # Insert the Neuorn initial mechanism developed by McIntyre
		self.initSegment.gnap_initial = 0
		self.initSegment.Ra = 200
		self.initSegment.cm = 2

		for dendrite,i in zip(self.dendrite,range(self._nDendrites)):
			dendrite.nseg = 11
			dendrite.diam = self._diamDend[i]
			dendrite.L = self._lengthDend[i]
			dendrite.Ra = 200
			dendrite.cm = 2
			dendrite.insert('pas') # Insert the Neuron pass mechanism
			dendrite.g_pas = 7.7e-6 # Real data from Fleshman 1988 cell 35/4
			dendrite.e_pas = -70.0

		for node,paranode in zip(self.node,self.paranode):
			node.nseg = 1
			node.diam = 0.32*self._diamAxon+0.056
			node.L = 1
			node.Ra = 70
			node.cm = 2
			node.insert('axnode') # Insert the axnode mechanism developed by McIntyre/Richardson
			node.gnapbar_axnode = 0

			paranode.nseg = 5
			paranode.diam = self._diamAxon
			paranode.L = 100*self._diamAxon
			paranode.Ra = 70
			paranode.cm = 0.1/(2*9.15*paranode.diam+2*30)
			paranode.insert('pas')
			paranode.g_pas = 0.001/(2*9.15*paranode.diam+2*30)
			paranode.e_pas = -85


	def _build_topology(self):
		""" Connect the sections together. """
		#childSection.connect(parentSection, [parentX], [childEnd])

		self.initSegment.connect(self.soma,0,0)
		self.node[0].connect(self.initSegment,1,0)
		self.dendrite[0].connect(self.soma,1,0)

		for i in range(self._nDendrites-1):
			self.dendrite[i+1].connect(self.dendrite[i],1,0)

		for i in range(self._nNodes-1):
			self.paranode[i].connect(self.node[i],1,0)
			self.node[i+1].connect(self.paranode[i],1,0)
		self.paranode[i+1].connect(self.node[i+1],1,0)

	def create_synapse(self,type):
		""" Create and return a synapse that links motoneuron state variables to external events.

		The created synapse is also appended to a list containg all synapses the this motoneuron has.

		Keyword arguments:
		type -- type of synapse to be created. This could be:
		1) "excitatory" to create an excitatory synapse positioned on the dendritic tree
		2) "inhibitory" to create an inhibitory synapse positioned on the soma
		3) "ees" to create a synapse that mimic the recruitmend induced by electrical
		stimulation; in this case the synapse is positioned on the axon.
		"""

		if type=="excitatory":
			# from Iaf to Mn we usually have 5 boutons for synapse
			nBoutonsXsyn = 5
			n = 0
			for i in range(nBoutonsXsyn): n += np.random.poisson(4)
			n = round(n/nBoutonsXsyn) - 2 # mean and shift to 0
			if n<0: n=0
			elif n>self._nDendrites: n=self._nDendrites-1
			x = rnd.random()
			syn = h.ExpSyn(self.dendrite[int(n)](x))
			syn.tau = 0.5
			syn.e = 0
			self.synapses.append(syn)
		elif type=="inhibitory":
			syn = h.Exp2Syn(self.soma(0.5))
			syn.tau1 = 1.5
			syn.tau2 = 2
			syn.e = -75
			self.synapses.append(syn)
		elif type=="ees":
			syn = h.ExpSyn(self.node[3](0.5))
			syn.tau = 0.1
			syn.e = 50
			self.synapses.append(syn)

		return syn

	"""
	Redefinition of inherited methods
	"""

	def connect_to_target(self,target,weight=0,delay=1):
		""" Connect the current cell to a target cell and return the netCon object.

		Keyword arguments:
		target -- the target object to which we want to connect
		weight -- the weight of the connection (default 0)
		delay -- communication time delay in ms (default 1)
		"""

		nc = h.NetCon(self.node[-1](1)._ref_v,target,sec=self.node[-1])
		nc.delay = delay
		nc.weight[0] = weight
		nc.threshold = -30
		return nc

	def is_artificial(self):
		""" Return a flag to check whether the cell is an integrate-and-fire or artificial cell. """
		return 0


	@classmethod
	def get_ees_weight(cls):
		""" Return the weight of a connection between an ees object and this cell. """
		return Motoneuron.__eesWeight
