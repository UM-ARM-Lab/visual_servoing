from visual_servoing.pbvs import PBVS
from visual_servoing.camera import Camera
from visual_servoing.arm_robot import ArmRobot
import pybullet as p
import time

class PBVSLoop:

    def __init__(self, pbvs: PBVS, camera : Camera, robot : ArmRobot, side : str, pbvs_hz : float, sim_hz : float ):
        self.pbvs = pbvs
        self.camera = camera
        p.setTimeStep(1/sim_hz)
        self.sim_steps_per_pbvs = int(sim_hz/pbvs_hz) 
        self.pbvs_dt = 1/pbvs_hz
        self.robot = robot
        self.side = side

    def run(self, target):
        while True:
            self.on_before_step_pbvs()
            ctrl, Twe = self.step_pbvs()
            self.on_after_step_pbvs(ctrl, Twe)
            self.step_simulation()
            self.on_after_step_sim()
            if(self.terminating_condition(ctrl, Twe)):
                break

    def step_simulation(self):
        for _ in range(self.sim_steps_per_pbvs):
            p.stepSimulation()

    def step_pbvs(self):
        rgb, depth = camera.get_image()
        rgb_edit = rgb[..., [2, 1, 0]].copy()

        ctrl, Twe = self.pbvs.do_pbvs(rgb_edit, depth, Two, Tae, val.get_arm_jacobian("left"), val.get_jacobian_pinv("left"), sim_dt)

        val.set_velo(val.get_jacobian_pinv("left") @ ctrl)

        return ctrl, Twe


    def on_before_step_sim(self, Two):
        pass

    def on_after_step_sim(self):
        pass

    def on_before_step_pbvs(self):
        pass

    def on_after_step_pbvs(self, ctrl, Twe):
        self.robot.psuedoinverse_ik_controller(ctrl)

    def terminating_condition(self, ctrl, Twe):
        pass