import argparse
import sys
import pickle
sys.path.append('../code')
import time
from mpi4py import MPI
from neuron import h
from tools import general_tools  as gt
from tools import load_data_tools as ldt
from tools import seed_handler as sh
from parameters import HumanParameters as hp


comm = MPI.COMM_WORLD
sizeComm = comm.Get_size()
rank = comm.Get_rank()

def main():
	""" This program launches a ForSimMuscleSpindles simulation with a predefined NeuralNetwork structure and
	senory-eencoding spatiotemporal EES profiles with amplitude and frequency given by the user as argument.

	This program can be executed both with and without MPI. In case MPI is used the cells
	of the NeuralNetwork are shared between the different hosts in order to speed up the
	simulation.


	Examples of how to run this script from the terminal:
	mpiexec -np 4 python scripts/runForSimMuscleSpindlesStimModulation.py 60 280 human sensory frwSimHuman.txt testHumanSpatiotemporalEES

	"""

	parser = argparse.ArgumentParser(description="launch a ForSimMuscleSpindles simulation")
	parser.add_argument("eesFrequency", help="ees frequency", type=float, choices=[gt.Range(0,1000)])
	parser.add_argument("eesAmplitude", help="ees amplitude (0-600] or %%Ia_II_Mn")
	parser.add_argument("species", help="simulated species", choices=["rat","human"])
	parser.add_argument("modulation", help="type of stimulation modulation", choices=["sensory"])
	parser.add_argument("inputFile", help="neural network structure file")
	parser.add_argument("name", help="name to add at the output files")
	parser.add_argument("--simTime", help="simulation time", type=int, default=-1)
	parser.add_argument("--noPlot", help="no plots flag", action="store_true")
	parser.add_argument("--burstingEes", help="flag to use burst stimulation", action="store_true")
	parser.add_argument("--nPulsesPerBurst", help="number of pulses per burst", type=int, default=5)
	parser.add_argument("--burstsFrequency", help="stimulation frequency within bursts",type=float, default=600, choices=[gt.Range(0,1000)])
	parser.add_argument("--seed", help="positive seed used to initialize random number generators (default = time.time())", type=int, choices=[gt.Range(0,999999)])
	args = parser.parse_args()

	if args.seed is not None: sh.save_seed(args.seed)
	else: sh.save_seed(int(time.time()))

	# Import simulation specific modules
	from simulations import ForSimMuscleSpindles
	from NeuralNetwork import NeuralNetwork
	from EES import EES
	from BurstingEES import BurstingEES


	# Initialze variables...
	if args.eesAmplitude[0]=="%": eesAmplitude = [float(x) for x in args.eesAmplitude[1:].split("_")]
	else: eesAmplitude = float(args.eesAmplitude)
	name = "_"+args.species+"_"+args.name
	if args.species == "rat":
		raise(Exception("Spatiotemporal modulation is not implemented for the rat model"))
	elif args.species == "human":
		muscles = hp.get_muscles_dict()
		if args.modulation == "sensory": fileStimMod = "../inputFiles/spatiotemporalSensoryProfile.p"
		if 'fileStimMod' in locals():
			with open(fileStimMod, 'r') as pickle_file:
				eesModulation = pickle.load(pickle_file)

	pc = h.ParallelContext()
	nn=NeuralNetwork(pc,args.inputFile)
	if not args.burstingEes: ees = EES(pc,nn,eesAmplitude,args.eesFrequency,pulsesNumber=100000,species=args.species)
	else: ees = BurstingEES(pc,nn,eesAmplitude,args.eesFrequency,args.burstsFrequency,args.nPulsesPerBurst,species=args.species)
	ees.get_amplitude(True)
	afferentsInput = ldt.load_afferent_input(args.species,muscles)


	simulation = ForSimMuscleSpindles(pc,nn, afferentsInput, ees, eesModulation, args.simTime)

	# Run simulation, plot results and save them
	simulation.run()
	if not args.noPlot: simulation.plot(muscles["flex"],muscles["ext"],name,False)
	comm.Barrier()
	simulation.save_results(muscles["flex"],muscles["ext"],name)


if __name__ == '__main__':
	main()
