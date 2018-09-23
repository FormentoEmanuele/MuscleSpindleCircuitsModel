from mpi4py import MPI
from neuron import h
from ForwardSimulation import ForwardSimulation
from cells import AfferentFiber
import random as rnd
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tools import firings_tools as tlsf
import pickle
from tools import seed_handler as sh
sh.set_seed()

comm = MPI.COMM_WORLD
sizeComm = comm.Get_size()
rank = comm.Get_rank()

class ForSimMuscleSpindles(ForwardSimulation):
	""" Integration of a NeuralNetwork object of two antagonist muscles over time given an input.
		muscle EMGs and neural firing rates are the results extracted at the end of simulation.
	"""

	def plot(self,flexorMuscle,extensorMuscle,name="",showFig=True):
		""" Plot and save the simulation results.

		Keyword arguments:
		flexorMuscle -- flexor muscle to plot
		extensorMuscle -- extensor muscle to plot
		name -- string to add at predefined file name of the saved pdf.
		"""
		meanPercErasedApFlexIaf,percErasedApFlexIaf = self._get_perc_aff_ap_erased(flexorMuscle,self._Iaf)
		meanPercErasedApExtIaf,percErasedApExtIaf = self._get_perc_aff_ap_erased(extensorMuscle,self._Iaf)
		meanPercErasedApFlexIIf,percErasedApFlexIIf = self._get_perc_aff_ap_erased(flexorMuscle,self._IIf)
		meanPercErasedApExtIIf,percErasedApExtIIf = self._get_perc_aff_ap_erased(extensorMuscle,self._IIf)
		meanPerEraserApIaf = np.mean([meanPercErasedApFlexIaf,meanPercErasedApExtIaf])
		meanPerEraserApIIf = np.mean([meanPercErasedApFlexIIf,meanPercErasedApExtIIf])

		#should be on rank 0?
		if not self._ees == None:
			percFiberActEes = self._ees.get_amplitude()
			title = 'EES _ {0:.0f}uA _ {1:.0f}Hz _ Delay _ {2:.0f}ms '.format(percFiberActEes[0],self._ees.get_frequency(),\
				self._nn.cells[flexorMuscle][self._Iaf][0].get_delay())
			plt.suptitle(title+"\n Iaf = {0:.0f}%, IIf = {1:.0f}%, Mn = {2:.0f}%, PercErasedApIaf = {3:.0f}%, prasedApIIf = {4:.0f}% ".format(\
				100*percFiberActEes[1],100*percFiberActEes[2],100*percFiberActEes[3],meanPerEraserApIaf,meanPerEraserApIIf))
		else:
			title = ' No EES _ Delay _ {2:.0f} ms '.format(self._nn.cells[flexorMuscle][self._Iaf][0].get_delay())
			plt.suptitle(title+"\n PercErasedApIaf = {3:.0f}%, prasedApIIf = {4:.0f}% ".format(meanPerEraserApIaf,meanPerEraserApIIf))


		if rank == 0:
			if not self._nn.recordMotoneurons and not self._nn.recordMotoneurons:
				raise(Exception("To plot the results it is necessary to have the NeuralNetwork recordMotoneurons and recordAfferents flags set to True"))

			flexAfferentModel = self._meanFr[flexorMuscle][self._Iaf]
			extAfferentModel = self._meanFr[extensorMuscle][self._Iaf]
			flexMnAll = [self._meanFr[flexorMuscle][mnName] for mnName in self._Mn]
			extMnAll = [self._meanFr[extensorMuscle][mnName] for mnName in self._Mn]
			flexMn = np.mean(flexMnAll,axis=0)
			extMn = np.mean(extMnAll,axis=0)

			if self._afferentModulation:
				flexAfferentInput = self._afferentInput[flexorMuscle][self._Iaf]
				extAfferntInput = self._afferentInput[extensorMuscle][self._Iaf]
			info,temp = self._nn.get_mn_info()
			strInfo = []
			for line in info: strInfo.append(" ".join(line))
			tStop = self._get_tstop()

			fig1, ax1 = plt.subplots(2, 4,figsize=(24,10),sharex='col',sharey='col')
			""" afferents fr subplots """
			ax1[0,0].plot(flexAfferentModel,color='b',label='model')
			if self._afferentModulation:
				ax1[0,0].plot(np.arange(0,tStop,self._dtUpdateAfferent),flexAfferentInput[:np.ceil(float(tStop)/self._dtUpdateAfferent).astype(int)],color='r',label='input')
			ax1[0,0].set_title('Ia afferents mean firing rate')
			ax1[0,0].legend(loc='upper right')
			ax1[0,0].set_ylabel("Ia firing rate (Hz) - flex")
			ax1[1,0].plot(extAfferentModel,color='b',label='model')
			if self._afferentModulation:
				ax1[1,0].plot(np.arange(0,tStop,self._dtUpdateAfferent),extAfferntInput[:np.ceil(float(tStop)/self._dtUpdateAfferent).astype(int)],color='r',label='input')

			ax1[1,0].legend(loc='upper right')
			ax1[1,0].set_ylabel("Ia firing rate (Hz) - ext")
			ax1[1,0].set_xlabel("time (ms)")

			""" afferents percAperased subplots """
			ax1[0,1].hist(percErasedApFlexIaf, bins=range(0,101,2), facecolor='green', alpha=0.75)
			ax1[0,1].set_title("Histogram of the Ap perc erased in Iaf")
			ax1[0,1].set_ylabel("Frquency - flex")
			ax1[0,1].axis([0, 100, 0, 60])
			ax1[1,1].hist(percErasedApExtIaf, bins=range(0,101,2), facecolor='green', alpha=0.75)
			ax1[1,1].set_ylabel("Frquency - ext")
			ax1[1,1].set_xlabel("Percentage %")
			ax1[1,1].axis([0, 100, 0, 60])

			""" Motoneurons subplots """
			ax1[0,2].plot(flexMn)
			ax1[0,2].set_title('Motoneurons mean firing rate')
			ax1[0,2].set_ylabel("Mn firing rate (Hz) - flex")
			ax1[1,2].plot(extMn)
			ax1[1,2].set_ylabel("Mn firing rate (Hz) - ext")
			ax1[1,2].set_xlabel("time (ms)")
			if len(extMnAll)>1:
				for extMnType in extMnAll: ax1[1,2].plot(extMnType)
				for flexMnType in flexMnAll: ax1[0,2].plot(flexMnType)

			""" EMG plot """

			ax1[0,3].plot(self.get_estimated_emg(flexorMuscle),color='b',label='model')
			ax1[0,3].set_ylabel("amplutide (a.u.) - flex")
			ax1[0,3].set_xlabel("time (ms)")
			ax1[0,3].set_title('Estimated EMG')
			ax1[1,3].plot(self.get_estimated_emg(extensorMuscle),color='b',label='model')
			ax1[1,3].set_ylabel("amplutide (a.u.) - ext")
			ax1[1,3].set_xlabel("time (ms)")

			title = title.replace(" ","")
			if showFig:	plt.show()

			fig2 = plt.figure()
			ax2 = fig2.add_subplot(111)
			ax2.text(0.5, 0.5,"\n".join(strInfo), horizontalalignment='center',verticalalignment='center',transform=ax2.transAxes)
			ax2.xaxis.set_visible(False)
			ax2.yaxis.set_visible(False)

			""" IaInt Figure """
			fig3 = plt.figure()
			ax3 = fig3.add_subplot(111)
			pl1=ax3.plot(self._meanFr[flexorMuscle]["IaInt"],label='flex')
			ax3.set_title('IaInt mean firing rate')
			ax3.set_ylabel("IaInt firing rate (Hz) - flex")
			pl2=ax3.plot(self._meanFr[extensorMuscle]["IaInt"],label='ext')
			ax3.set_ylabel("IaInt firing rate (Hz) - ext")
			ax3.set_xlabel("time (ms)")
			plt.legend()

			fileName = time.strftime("%Y_%m_%d_FS_"+title+name+".pdf")
			pp = PdfPages(self._resultsFolder+fileName)
			pp.savefig(fig1)
			pp.savefig(fig3)
			pp.savefig(fig2)
			pp.close()


	def save_results(self,flexorMuscle,extensorMuscle,name=""):
		""" Save the resulting motoneurons mean firings and EMG.

		Keyword arguments:
		name -- string to add at predefined file name of the saved pdf (default = "").
		"""

		meanPercErasedApFlexIaf,percErasedApFlexIaf = self._get_perc_aff_ap_erased(flexorMuscle,self._Iaf)
		meanPercErasedApExtIaf,percErasedApExtIaf = self._get_perc_aff_ap_erased(extensorMuscle,self._Iaf)
		meanPerEraserApIaf = np.mean([meanPercErasedApFlexIaf,meanPercErasedApExtIaf])

		if rank == 0:
			if not self._ees == None:
				percFiberActEes = self._ees.get_amplitude()
				title = 'EES_{0:.0f}uA_{1:.0f}Hz_Delay_{2:.0f}ms'.format(percFiberActEes[0],self._ees.get_frequency(),\
					self._nn.cells[flexorMuscle][self._Iaf][0].get_delay())
			else: title = 'NoEES_Delay_{2:.0f}ms'.format(self._nn.cells[flexorMuscle][self._Iaf][0].get_delay())
			fileName = time.strftime("%Y_%m_%d_FS_"+title+name+".p")

			with open(self._resultsFolder+fileName, 'w') as pickle_file:
				pickle.dump(self._estimatedEMG, pickle_file)
				pickle.dump(self._meanFr, pickle_file)
				pickle.dump(meanPerEraserApIaf,pickle_file)
