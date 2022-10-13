import time
from typing import Dict

import cv2
import numpy as np
import pybullet as p

import rospy
import time 
from visual_servoing.arm_robot import ArmRobot
from visual_servoing.camera import Camera
from visual_servoing.pbvs import PBVS


class AbstractPBVSLoop:

    def __init__(self, pbvs: PBVS, camera: Camera, robot: ArmRobot, side: str, config: Dict):
        self.pbvs = pbvs
        self.camera = camera
        self.robot = robot
        self.side = side
        self.config = config

    def run(self, target):
        self.on_before_run()
        start_time = self.get_time()
        last_t = self.get_time()

        while True:
            rgb, depth = self.get_camera_image()

            self.on_before_step_pbvs(rgb, depth)

            current_t = self.get_time()
            pbvs_dt = self.get_pbvs_dt(current_t, last_t)

            # Twe is the current transform of the end effector in the world frame
            # NOTE: Twe and twist should probably be in camera frame, defined by the cv2 standard (z-forward) ???
            twist, Twe = self.step_pbvs(rgb, depth, target, pbvs_dt)

            self.on_after_step_pbvs(Twe)

            # NOTE: names are meh
            q_dot = self.servoing_controller(twist)
            self.robot.velocity_control(self.side, q_dot)  # FIXME: implement this

            last_t = current_t

            total_time = self.get_time() - start_time
            is_timed_out = total_time > self.config['timeout']
            # check if error is low enough to terminate
            pos_error = np.linalg.norm(Twe[0:3, 3] - target[0:3, 3])
            rot_error = np.linalg.norm(cv2.Rodrigues(Twe[0:3, 0:3].T @ target[0:3, 0:3])[0])
            target_reached = pos_error < self.config['max_pos_error'] and rot_error < self.config['max_rot_error']

            self.on_check_is_done(is_timed_out, target_reached)

            if self.is_done(is_timed_out, target_reached):
                break

    def is_done(self, is_timed_out, target_reached):
        return is_timed_out or target_reached

    def on_check_is_done(self, is_timed_out, target_reached):
        pass

    def get_time(self):
        return time.time()

    def on_before_run(self):
        pass

    def get_camera_image(self):
        rgb, depth = self.camera.get_image()
        return rgb, depth

    def get_pbvs_dt(self, current_t, last_t):
        return current_t - last_t

    def step_pbvs(self, rgb, depth, target, pbvs_dt: float):
        twist, Twe = self.pbvs.do_pbvs(rgb, depth, target, np.eye(4),
                                       self.robot.get_arm_jacobian(self.side),
                                       self.robot.get_jacobian_pinv(self.side), pbvs_dt)
        return twist, Twe

    def on_before_step_pbvs(self, rgb, depth):
        pass

    def on_after_step_pbvs(self, Twe):
        pass

    def terminating_condition(self):
        pass

    def servoing_controller(self, twist):
        J = self.robot.get_arm_jacobian(self.side)
        lmda = 0.0000001
        J_pinv = np.linalg.inv(J.T @ J + lmda * np.eye(7)) @ J.T
        q_dot = J_pinv @ twist
        return q_dot


class RealVictorPBVSLoop(AbstractPBVSLoop):
    pass

class RealValPBVSLoop(AbstractPBVSLoop):
    pass

class PybulletPBVSLoop(AbstractPBVSLoop):

    def __init__(self, pbvs: PBVS, camera: Camera, robot: ArmRobot, side: str, pbvs_hz: float, sim_hz: float,
                 config: Dict):
        super().__init__(pbvs, camera, robot, side, config)
        p.setTimeStep(1 / sim_hz)
        self.sim_steps_per_pbvs = int(sim_hz / pbvs_hz)
        self.pbvs_dt = 1 / pbvs_hz
        self.sim_dt = 1 / sim_hz

    def get_time(self):
        return time.time()

    def get_pbvs_dt(self, current_t, last_t):
        return self.pbvs_dt

    def on_after_step_pbvs(self, Twe):
        for _ in range(self.sim_steps_per_pbvs):
            p.stepSimulation()
