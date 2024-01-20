from src.dem.ContactManager import ContactManager
from src.dem.engines.EngineKernel import *
from src.dem.engines.ExplicitEngine import ExplicitEngine
from src.dem.neighbor.NeighborBase import NeighborBase
from src.dem.SceneManager import myScene
from src.dem.Simulation import Simulation
from src.utils.ObjectIO import DictIO


class ElementTest(ExplicitEngine):
    def __init__(self, contactor: ContactManager):
        super().__init__(contactor)
        self.compression = "HydroCompression"
        self.container = "Cube"
        self.status = None

        self.get_stiffness = None
        self.get_area = None

    def set_servo_mechanism(self, sims: Simulation, callback=None):    
        if sims.max_servo_wall_num > 0 and sims.servo_status == "On":
            self.update_servo_wall = self.update_servo_motion
        else:
            self.update_servo_wall = self.no_operation_other

    def check(self, scene: myScene, option):
        self.compression = DictIO.GetAlternative(option, "CompressionType", "HydroCompression")
        self.container = DictIO.GetAlternative(option, "Container", "Cube")
        self.status = DictIO.GetEssential(option, "Status")

    def update_servo_motion(self, sims: Simulation, scene: myScene, neighbor: NeighborBase):
        get_contact_stiffness(sims.max_material_num, scene.particleNum[0], scene.particle, scene.wall, self.physpw.surfaceProps, self.physpw.cplist, neighbor.particle_wall)
        self.get_area()
        get_gain(sims.dt, int(scene.servoNum[0]), scene.servo, scene.wall)
        servo(int(scene.servoNum[0]), scene.wall, scene.servo)

    def integration(self, sims: Simulation, scene: myScene, neighbor: NeighborBase):
        self.update_servo_wall(sims, scene, neighbor)
        self.calcu_sphere_position(sims, scene)
        self.calcu_clump_position(sims, scene)