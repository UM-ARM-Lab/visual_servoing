from visual_servoing.pbvs import PBVS
from visual_servoing.camera import Camera
from visual_servoing.arm_robot import ArmRobot
import pybullet as p
import numpy as np
import time

class PBVSLoop:

    def __init__(self, pbvs: PBVS, camera : Camera, robot : ArmRobot, side : str, pbvs_hz : float, sim_hz : float ):
        self.pbvs = pbvs
        self.camera = camera
        p.setTimeStep(1/sim_hz)
        self.sim_steps_per_pbvs = int(sim_hz/pbvs_hz) 
        self.pbvs_dt = 1/pbvs_hz
        self.sim_dt = 1/sim_hz
        self.robot = robot
        self.side = side

    def run(self, target):
        self.target = target
        self.start_time = time.time()
        while True:
            self.on_before_step_pbvs()
            self.step_pbvs()
            self.on_after_step_pbvs()
            self.step_simulation()
            self.on_after_step_sim()
            if(self.terminating_condition()):
                break

    def step_simulation(self):
        for _ in range(self.sim_steps_per_pbvs):
            p.stepSimulation()

    def step_pbvs(self):
        self.ctrl, self.Twe = self.pbvs.do_pbvs(self.rgb, self.depth, self.target, np.eye(4), self.robot.get_arm_jacobian(self.side), self.robot.get_jacobian_pinv(self.side), self.pbvs_dt)

    def on_after_step_sim(self):
        pass

    def on_before_step_pbvs(self):
        self.rgb, self.depth = self.camera.get_image()

    def on_after_step_pbvs(self):
        self.robot.psuedoinverse_ik_controller(self.ctrl)

    def terminating_condition(self):
        pass