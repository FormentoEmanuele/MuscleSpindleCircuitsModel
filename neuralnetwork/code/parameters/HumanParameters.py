from mpi4py import MPI
comm = MPI.COMM_WORLD
sizeComm = comm.Get_size()
rank = comm.Get_rank()

class HumanParameters():

    @classmethod
    def get_tot_sim_time(cls):
        return 9995

    @classmethod
    def get_gait_cycles_file(cls):
        return "../inputFiles/humanGaitCyclesB13.p"

    @classmethod
    def get_muscles(cls):
        return ["SOL","TA"]

    @classmethod
    def get_muscles_dict(cls):
        return {"ext":"SOL","flex":"TA"}
