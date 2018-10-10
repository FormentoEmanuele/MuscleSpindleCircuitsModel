import argparse
import sys
sys.path.append('../code')
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import os
import fnmatch
import time
import pickle
from tools import general_tools as gt

pathToResults = "../../results"

def main():
	""" This program launches several static reflex recordings with different random seeds

	Example of how to run this script from the terminal:
	python scripts/runBatchOfRecCurve.py 1 fsMnMod.txt test --mnReal --burstingEes
	"""

	parser = argparse.ArgumentParser(description="launch several static reflex recordings with different random seeds")
	parser.add_argument("nSim", help="number of simulations to run", type=int)
	parser.add_argument("inputFile", help="neural network structure file")
	parser.add_argument("outFileName", help="name to add at the output files")
	parser.add_argument("--mnReal", help=" real mn flag", action="store_true")
	parser.add_argument("--burstingEes", help="flag to use burst stimulation", action="store_true")
	parser.add_argument("--nPulsesPerBurst", help="number of pulses per burst", type=int, default=5)
	parser.add_argument("--burstsFrequency", help="stimulation frequency within bursts",type=float, default=600, choices=[gt.Range(0,1000)])
	parser.add_argument("--muscleName", help="flag to compute the membrane potential", type=str, default="GM")
	args = parser.parse_args()

	nProc = 4
	count=0.
	percLastPrint=0.
	printPeriod = 0.05

	programExtra=[]
	figNameExtra = ""
	if args.burstingEes:
		args.outFileName+="_BurstEES_%dHz_%dPulses"%(args.burstsFrequency,args.nPulsesPerBurst)
		figNameExtra += "_BurstEES_%dHz_%dPulses"%(args.burstsFrequency,args.nPulsesPerBurst)
		programExtra += ["--burstingEes","--nPulsesPerBurst",str(args.nPulsesPerBurst),"--burstsFrequency",str(args.burstsFrequency)]

	if args.mnReal:
		args.outFileName += "_realMn"
		figNameExtra += "_realMn"
		programExtra += ["--mnReal"]

	for n in xrange(args.nSim):

		resultName = "batchRecCurve_seedn_"+str(n)+"_"+args.outFileName
		resultFile = gt.find("*"+resultName+".p",pathToResults)
		if not resultFile:
			program = ['mpiexec','-np',str(nProc),'python','./scripts/runRecCurve.py',\
					args.inputFile,"--name",resultName,"--seed",str(n),"--muscleName",args.muscleName,"--noPlot"]
			gt.run_subprocess(program+programExtra)

		count+=1
		if count/args.nSim-percLastPrint>=printPeriod:
			percLastPrint=count/args.nSim
			print str(round(count/args.nSim*100))+"% of simulations performed..."

	figName = time.strftime("/%Y_%m_%d_Batch_RecCurve_nSim_"+str(args.nSim)+figNameExtra)
	plot_rec_curve(args.nSim,figName,args.outFileName)

def plot_rec_curve(nSimulations,figName,outFileName,showPlot=False):
	""" Plots the recruitment curves. """

	mStat = []
	mSpikes = []
	for n in xrange(nSimulations):
		name = "batchRecCurve_seedn_"+str(n)+"_"+outFileName
		resultFile = gt.find("*"+name+".p",pathToResults)
		if len(resultFile)>1: print "Warning: multiple result files found!!!"
		with open(resultFile[0], 'r') as pickle_file:
			mSpikesOneSeed = pickle.load(pickle_file)
			mEmg = pickle.load(pickle_file)
			mStatOneSeed = pickle.load(pickle_file)
		mStat.append(mStatOneSeed)
		mSpikes.append(mSpikesOneSeed)

	fig, ax = plt.subplots(1,figsize=(16,3))
	fig2, ax2 = plt.subplots(1,figsize=(16,3))
	recCruve = []
	delay = []
	for oneSeed in mSpikes:
		delay.append([np.argmax(amp)*1000/1000. if amp.max()>0 else np.nan for amp in oneSeed])
		recCruve.append([np.sum(amp) for amp in oneSeed])

	meanDelay = np.mean(delay,axis=0)
	sem = np.std(delay,axis=0)/np.sqrt(len(delay)-1)

	ax2.plot(np.arange(0.05,1.05,0.05),meanDelay,color="#49ab4d")
	ax2.fill_between(np.arange(0.05,1.05,0.05), meanDelay-sem, meanDelay+sem, facecolor='#49ab4d', alpha=0.5)
	ax2.set_title('Response delays - with different seeds')
	fig2.savefig(pathToResults+figName+"_delays.pdf", format="pdf",transparent=True)


	meanRecCurve = np.mean(recCruve,axis=0)
	sem = np.std(recCruve,axis=0)/np.sqrt(len(recCruve)-1)

	ax.plot(np.arange(0.05,1.05,0.05),meanRecCurve,color="#49ab4d")
	ax.fill_between(np.arange(0.05,1.05,0.05), meanRecCurve-sem, meanRecCurve+sem, facecolor='#49ab4d', alpha=0.5)
	ax.set_title('Recruitment curve - with different seeds')
	fig.savefig(pathToResults+figName+"_rc.pdf", format="pdf",transparent=True)
	if showPlot: plt.show()

if __name__ == '__main__':
	main()
