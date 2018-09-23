import argparse
import time
import sys
sys.path.append('../code')
from mpi4py import MPI
from neuron import h
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import pickle
from tools import general_tools  as gt
from tools import seed_handler as sh

comm = MPI.COMM_WORLD
sizeComm = comm.Get_size()
rank = comm.Get_rank()

def main():
	""" This program launches a ForwardSimulation simulation with a predefined NeuralNetwork structure,
	different stimulation amplitudes are tested to evealuate the muslce recruitment curve.
	The plots resulting from this simulation are saved in the results folder.

	This program can be executed both with and without MPI. In case MPI is used the cells
	of the NeuralNetwork are shared between the different hosts in order to speed up the
	simulation.
	"""

	parser = argparse.ArgumentParser(description="Estimate the reflex responses induced by a range of stimulation amplitudes")
	parser.add_argument("inputFile", help="neural network structure file")
	parser.add_argument("--name", help="name to add at the output files", type=str, default="")
	parser.add_argument("--noPlot", help=" no plot flag", action="store_true")
	parser.add_argument("--mnReal", help=" real mn flag", action="store_true")
	parser.add_argument("--burstingEes", help="flag to use burst stimulation", action="store_true")
	parser.add_argument("--nPulsesPerBurst", help="number of pulses per burst", type=int, default=5)
	parser.add_argument("--burstsFrequency", help="stimulation frequency within bursts",type=float, default=600, choices=[gt.Range(0,1000)])
	parser.add_argument("--seed", help="positive seed used to initialize random number generators (default = time.time())", type=int, choices=[gt.Range(0,999999)])
	parser.add_argument("--membranePotential", help="flag to compute the membrane potential", action="store_true")
	parser.add_argument("--muscleName", help="flag to compute the membrane potential", type=str, default="GM")
	args = parser.parse_args()

	if args.seed is not None: sh.save_seed(args.seed)
	else: sh.save_seed(int(time.time()))

	# Import simulation specific modules
	from simulations import ForwardSimulation
	from simulations import ForSimSpinalModulation
	from NeuralNetwork import NeuralNetwork
	from EES import EES
	from BurstingEES import BurstingEES

	# Initialize parameters
	eesAmplitudes = [[x,0,0] for x in np.arange(0.05,1.05,0.05)]
	eesFrequency = 10
	simTime = 150
	plotResults = True


	# Create a Neuron ParallelContext object to support parallel simulations
	pc = h.ParallelContext()
	nn=NeuralNetwork(pc,args.inputFile)

	if not args.burstingEes: ees = EES(pc,nn,eesAmplitudes[0],eesFrequency)
	else: ees = BurstingEES(pc,nn,eesAmplitudes[0],eesFrequency,args.burstsFrequency,args.nPulsesPerBurst)
	afferentsInput = None
	eesModulation = None

	if args.membranePotential:
		if args.mnReal:
			cellsToRecord = {"MnReal":[mn.soma for mn in nn.cells[args.muscleName]['MnReal']]}
			modelTypes = {"MnReal":"real"}
		else:
			cellsToRecord = {"Mn":nn.cells[args.muscleName]['Mn']}
			modelTypes = {"Mn":"artificial"}
		simulation = ForSimSpinalModulation(pc,nn,cellsToRecord,modelTypes, afferentsInput, ees, eesModulation, simTime)
		membranePotentials = []
	else: simulation = ForwardSimulation(pc,nn, afferentsInput, ees, eesModulation, simTime)

	mEmg = []
	mSpikes = []
	mStatTemp = []
	nSamplesToAnalyse = -100 # last 100 samples

	if not args.noPlot:
		fig, ax = plt.subplots(3,figsize=(16,9),sharex='col',sharey='col')
		fig2, ax2 = plt.subplots(1,figsize=(16,3))

	for eesAmplitude in eesAmplitudes:
		ees.set_amplitude(eesAmplitude)
		percFiberActEes = ees.get_amplitude(True)
		simulation.run()

		# Extract emg responses
		try: mEmg.append(simulation.get_estimated_emg(args.muscleName)[nSamplesToAnalyse:])
		except (ValueError, TypeError) as error: mEmg.append(np.zeros(abs(nSamplesToAnalyse)))

		# Extract mn spikes
		try: mSpikes.append(simulation.get_mn_spikes_profile(args.muscleName)[nSamplesToAnalyse:])
		except (ValueError, TypeError) as error: mSpikes.append(np.zeros(abs(nSamplesToAnalyse)))

		# plot mn membrane potentials
		if args.membranePotential:
			title = "%s_amp_%d_Ia_%f_II_%f_Mn_%f"%(args.name,percFiberActEes[0],percFiberActEes[1],percFiberActEes[2],percFiberActEes[3])
			try: fileName = "%s_amp_%d"%(args.name,eesAmplitude)
			except: fileName = "%s_amp_%.2f"%(args.name,eesAmplitude[0])
			simulation.plot_membrane_potatial(fileName,title)

		# Compute statistics
		mStatTemp.append(np.abs(mEmg[-1]).sum())

		if rank==0 and not args.noPlot:
			ax[1].plot(mEmg[-1])
			ax[0].plot(mSpikes[-1])
		comm.Barrier()

	if rank==0:
		resultsFolder = "../../results/"
		generalFileName = time.strftime("%Y_%m_%d_recCurve"+args.name)
		mStat = np.array(mStatTemp).sum()
		if not args.noPlot:
			ax[2].bar(1, mStat, 0.2)
			ax[0].set_title('Mn action potentials')
			ax[1].set_title('EMG response')
			ax[2].set_title('Statistic')

			ax2.plot([np.sum(x) for x in mSpikes])

			fileName = generalFileName+".pdf"
			pp = PdfPages(resultsFolder+fileName)
			pp.savefig(fig)
			pp.close()
			plt.show()

		fileName = generalFileName+".p"
		with open(resultsFolder+fileName, 'w') as pickle_file:
			pickle.dump(mSpikes, pickle_file)
			pickle.dump(mEmg, pickle_file)
			pickle.dump(mStat, pickle_file)

	comm.Barrier()


if __name__ == '__main__':
	main()
