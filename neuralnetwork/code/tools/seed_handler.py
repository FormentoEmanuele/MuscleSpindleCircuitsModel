from mpi4py import MPI
import random as rnd
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
sizeComm = comm.Get_size()
seedValFile = "tools/seedVal.txt"

def save_seed(val):
    """ saves val. Called once in simulation1.py """
    if rank == 0:
        with open(seedValFile, "wb") as f:
            f.write(str(int(val)))

def load_seed():
    """ loads val. Called by all scripts that need the shared seed value """
    seed = None
    if rank == 0:
        with open(seedValFile, "rb") as f:
            seed = int(f.read())
        with open(seedValFile, "wb") as f:
            f.write(str(seed+1))
    seed = comm.bcast(seed,root=0)
    return seed

def set_seed():
    seed = load_seed()
    for i in range(sizeComm):
        if i==rank:
            rnd.seed(seed+rank)
            np.random.seed(seed+rank)
