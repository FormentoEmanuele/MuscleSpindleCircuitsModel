from mpi4py import MPI
import numpy as np
import os
import fnmatch
import subprocess
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from time import sleep

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

class Range(object):
	def __init__(self, start, end):
		self.start = start
		self.end = end
	def __repr__(self):
		return '{0}-{1}'.format(self.start, self.end)
	def __eq__(self, other):
		return self.start <= other <= self.end

def load_txt_mpi(fileName):
	""" Load txt data files from one process and broadcast them to the odther processes.
	This loader is implemented to avoid race conditions.
	"""
	data = None
	if rank == 0: data = np.loadtxt(fileName)
	data = comm.bcast(data,root=0)
	return data

def naive_string_clustering(stringList):
	clusters = []
	for i,string1 in enumerate(stringList):
		clusters.append([])
		clusters[-1].append(string1)
		for j,string2 in enumerate(stringList):
			if i==j:continue
			if len(set(list(string1)).intersection(list(string2)))>=2:
				clusters[-1].append(string2)
	found = []
	for stringList in clusters:
		foundFlag=0
		for foundList in found:
			if set(foundList) == set(stringList):
				foundFlag=1
		if not foundFlag:found.append(stringList)
	return found

def find(pattern, path):
	" Finds the files in a path with a given pattern. "
	result = []
	for root, dirs, files in os.walk(path):
		for name in files:
			if fnmatch.fnmatch(name, pattern):
				result.append(os.path.join(root, name))
	return result

def run_subprocess(program):
	""" Runs a given program as a subrocess. """
	print "\tRunning subprocess: %s"%(" ".join(program))
	returnCode = None
	while not returnCode==0:
		p = subprocess.Popen(program, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		returnCode = None
		while returnCode is None:
			message =  p.stdout.readline().rstrip("\n").split()
			if message != None:print "\t\t"+" ".join(message)+"\t\t"
			sleep(0.1)
			returnCode = p.poll()
		if returnCode != 0: print "\t\t\t\t Error n: ",p.poll()," resetting simulation..."

def resample(dataDict,keys,ratio):
	for key in keys:
		dataDict[key] = np.interp(np.arange(0, len(dataDict[key]), ratio), np.arange(0, len(dataDict[key])), dataDict[key])
	return dataDict

def make_video(data,dt,fileName,windowLength):
	"""
		data: dict containing the np.array to plot
		dt: period of time between two values
		fileName: fileName
		windowLength: in ms
	"""

	fps = 30
	FFMpegWriter = animation.writers['ffmpeg']
	writer = FFMpegWriter(fps=fps)

	size = 0.5
	fig = plt.figure(figsize=(16*size,9*size))
	gs = gridspec.GridSpec(len(data.keys()),1)
	gs.update(left=0.05, right=0.95, hspace=0.6, wspace=0.1)

	ax = []
	p = []
	for i,key in enumerate(data):
		ax.append(plt.subplot(gs[i,0]))
		p.append(ax[-1].plot([], [], color='#152a57'))
		ax[-1].set_title(key)
		ax[-1].set_xlim(0,windowLength)
		ax[-1].set_ylim(data[key].min(),data[key].max())
		ax[-1].axis("off")

	time = np.arange(0,windowLength,dt)
	startInd = 0
	endInd = len(time)
	nSamplesToShift = int(1000./fps/dt)
	lengthData = data[key].size

	with writer.saving(fig, fileName+".mp4", 100):
		while endInd<lengthData:
			for i,key in enumerate(data):
				dataToPlot = data[key][startInd:endInd]
				p[i][0].set_data(time, dataToPlot)
			startInd += nSamplesToShift
			endInd += nSamplesToShift
			writer.grab_frame()
