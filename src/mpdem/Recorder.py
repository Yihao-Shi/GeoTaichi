import os

from src.dem.Recorder import WriteFile as DEMWriteFile
from src.dem.Simulation import Simulation as DEMSimulation
from src.mpdem.contact.ContactModelBase import ContactModelBase
from src.mpdem.Simulation import Simulation
from src.mpm.Recorder import WriteFile as MPMWriteFile


class WriteFile:
    drecorder: DEMWriteFile
    mrecorder: MPMWriteFile
    physpp: ContactModelBase
    physpw: ContactModelBase

    def __init__(self, sims, msims, dsims, drecorder: DEMWriteFile, mrecorder: MPMWriteFile, physpp, physpw, pcontact): 
        self.drecorder = drecorder
        self.mrecorder  = mrecorder
        self.physpp = physpp
        self.physpw = physpw
        self.pcontact = pcontact
        self.contact_path = None

        self.save_ppcontact = self.no_operation
        self.save_pwcontact = self.no_operation

        self.mkdir(sims)
        self.mrecorder.manage_function(msims)
        self.drecorder.manage_function(dsims)
        self.manage_function(sims, dsims)

    def no_operation(self, sims, dscene):
        pass

    def mkdir(self, sims: Simulation):
        if not os.path.exists(sims.path):
            os.makedirs(sims.path)

        self.contact_path = sims.path + '/DEMPMcontacts'
        if not os.path.exists(self.contact_path):
            os.makedirs(self.contact_path)

    def manage_function(self, sims: Simulation, dsims: DEMSimulation):
        if sims.particle_interaction and dsims.max_particle_num > 0. and 'ppcontact' in dsims.monitor_type:
            self.save_ppcontact = self.MonitorPPContact
        if sims.wall_interaction and dsims.max_wall_num > 0. and 'pwcontact' in dsims.monitor_type:
            self.save_pwcontact = self.MonitorPWContact

    def output(self, sims, msims, mscene, dsims, dscene):
        self.mrecorder.output(msims, mscene)
        self.drecorder.output(dsims, dscene)
        self.save_ppcontact(sims, mscene)
        self.save_pwcontact(sims, mscene)

    def MonitorPPContact(self, sims: Simulation, mscene): 
        self.physpp.get_ppcontact_output(self.contact_path+'/DEMPMContactPP', sims.current_time, sims.current_print, mscene, self.pcontact)

    def MonitorPWContact(self, sims: Simulation, mscene): 
        self.physpw.get_pwcontact_output(self.contact_path+'/DEMPMContactPW', sims.current_time, sims.current_print, mscene, self.pcontact)
