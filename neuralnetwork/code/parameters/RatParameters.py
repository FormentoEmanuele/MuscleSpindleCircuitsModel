class RatParameters():

    @classmethod
    def get_tot_sim_time(cls):
        return 12385

    @classmethod
    def get_gait_cycles_file(cls):
        return "../inputFiles/ratGaitCycles.p"

    @classmethod
    def get_muscles(cls):
        return ["TA","GM"]

    @classmethod
    def get_muscles_dict(cls):
        return {"ext":"GM","flex":"TA"}
