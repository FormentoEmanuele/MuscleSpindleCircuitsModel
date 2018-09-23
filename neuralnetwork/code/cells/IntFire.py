from Cell import Cell
from neuron import h
import time

class IntFire(Cell):
	""" Integrate and Fire cell.

	This class implement and IntFire4 Neuron object.
	Taus are set as in Moraud et al 2016.
	"""

	def __init__(self):
		""" Object initialization. """
		Cell.__init__(self)

		#Create IntFire4
		self.cell = h.IntFire4()
		self.cell.taue= 0.5
		self.cell.taui1=5
		self.cell.taui2=10
		self.cell.taum= 30 #0.2 nF divided g 20 nS
