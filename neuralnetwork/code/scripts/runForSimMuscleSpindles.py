import argparse
import sys
sys.path.append('../code')
import time as time
from mpi4py import MPI
from neuron import h
from tools import load_data_tools as ldt
from tools import seed_handler as sh
from tools import general_tools  as gt
from parameters import HumanParameters as hp
from parameters import RatParameters as rp

comm = MPI.COMM_WORLD
sizeComm = comm.Get_size()
rank = comm.Get_rank()

def main():
	""" This program launches a ForSimMuscleSpindles simulation with a NeuralNetwork structure,
	EES amplitude and EES frequency given by the user as argument. The NeuralNetwork
	needs to conatain the structure of a muscle spindle sensorimotor circuitry for 2
	antagonist muscles, 'TA' and 'GM' for a rat model and 'TA' and 'SOL' for a human model.
	Precomputed senosry information of the Ia and II fibers is used to drive the NN.
	The plots resulting from this simulation are saved in the results folder.

	This program can be executed both with and without MPI. In case MPI is used the cells
	of the NeuralNetwork are shared between the different hosts in order to speed up the
	simulation.

	Examples of how to run this script from the terminal:
	mpiexec -np 4 python scripts/runForSimMuscleSpindles.py 60 260 rat frwSimRat.txt testRat
	mpiexec -np 4 python scripts/runForSimMuscleSpindles.py 60 260 human frwSimHuman.txt testHuman
	mpiexec -np 4 python scripts/runForSimMuscleSpindles.py 60 260 human frwSimHumanRat.txt testHumanRat
	mpiexec -np 4 python scripts/runForSimMuscleSpindles.py 60 %0.2_0_0 human frwSimHuman.txt testHumanHFLA --burstingEes
	"""
	parser = argparse.ArgumentParser(description="launch a ForSimMuscleSpindles simulation")
	parser.add_argument("eesFrequency", help="ees frequency", type=float, choices=[gt.Range(0,1000)])
	parser.add_argument("eesAmplitude", help="ees amplitude (0-600] or %%Ia_II_Mn")
	parser.add_argument("species", help="simulated species", choices=["rat","human"])
	parser.add_argument("inputFile", help="neural network structure file")
	parser.add_argument("name", help="name to add at the output files")
	parser.add_argument("--simTime", help="simulation time", type=int, default=-1)
	parser.add_argument("--noPlot", help="no plot flag", action="store_true")
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
	if args.species == "human": muscles = hp.get_muscles_dict()
	elif args.species == "rat": muscles = rp.get_muscles_dict()
	pc = h.ParallelContext()
	nn=NeuralNetwork(pc,args.inputFile)
	if not args.burstingEes: ees = EES(pc,nn,eesAmplitude,args.eesFrequency,pulsesNumber=100000,species=args.species)
	else: ees = BurstingEES(pc,nn,eesAmplitude,args.eesFrequency,args.burstsFrequency,args.nPulsesPerBurst,species=args.species)
	ees.get_amplitude(True)
	afferentsInput = ldt.load_afferent_input(args.species,muscles)
	simulation = ForSimMuscleSpindles(pc,nn, afferentsInput, ees, None, args.simTime)

	# Run simulation, plot results and save them
	simulation.run()
	if not args.noPlot: simulation.plot(muscles["flex"],muscles["ext"],name,False)
	comm.Barrier()
	simulation.save_results(muscles["flex"],muscles["ext"],name)


if __name__ == '__main__':
	main()
