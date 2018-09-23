import numpy as np
import pandas as pd
from tools import general_tools  as gt
from parameters import HumanParameters as hp
from parameters import RatParameters as rp

def readCsvGeneral(file2read,headerLines,outLabels,signalsName,sep='\t'):
    outDict = {}
    dataFrame = pd.read_csv(file2read,header=headerLines,sep=sep)
    for label,muscle in zip(outLabels,signalsName):
        outDict[label] = dataFrame[muscle].values
    return outDict

def load_afferent_input(species='rat',muscles=None,exp="locomotion"):
    """ Load previously computed affarent inputs """
    afferentsInput = None
    if species == 'rat':
        muscles = {"ext":"GM","flex":"TA"}
        afferents = {}
        afferents[muscles["flex"]] = {}
        afferents[muscles["ext"]] = {}
        if exp == "locomotion":
            afferents[muscles["flex"]]['Iaf'] = list(gt.load_txt_mpi('../inputFiles/meanFr_Ia_TA_rat.txt'))
            afferents[muscles["flex"]]['IIf'] = list(gt.load_txt_mpi('../inputFiles/meanFr_II_TA_rat.txt'))
            afferents[muscles["ext"]]['Iaf'] = list(gt.load_txt_mpi('../inputFiles/meanFr_Ia_GM_rat.txt'))
            afferents[muscles["ext"]]['IIf'] = list(gt.load_txt_mpi('../inputFiles/meanFr_II_GM_rat.txt'))
        dtUpdateAfferent = 5
        afferentsInput = [afferents,dtUpdateAfferent]
    elif species == 'human':
        muscles = {"ext":"SOL","flex":"TA"}
        afferents = {}
        afferents[muscles["flex"]] = {}
        afferents[muscles["ext"]] = {}
        if exp == "locomotion":
            afferents[muscles["flex"]]['Iaf'] = list(gt.load_txt_mpi("../inputFiles/meanFr_Ia_TA_human.txt"))
            afferents[muscles["flex"]]['IIf'] = list(gt.load_txt_mpi("../inputFiles/meanFr_II_TA_human.txt"))
            afferents[muscles["ext"]]['Iaf'] = list(gt.load_txt_mpi("../inputFiles/meanFr_Ia_SOL_human.txt"))
            afferents[muscles["ext"]]['IIf'] = list(gt.load_txt_mpi("../inputFiles/meanFr_II_SOL_human.txt"))
        dtUpdateAfferent = 5
        afferentsInput = [afferents,dtUpdateAfferent]
    return afferentsInput
