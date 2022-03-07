from abc import ABC, abstractmethod
from types import new_class
import matplotlib
import numpy as np
import scipy.stats
import scipy.cluster
from visual_servoing.utils import draw_sphere_marker, draw_pose
import pybullet as p
import matplotlib.pyplot as plt
import matplotlib
import cv2

# Convert between homogenous transform and 6 vec SE3
def SE3(se3):
    # convert TO homogenous TF
    if(se3.shape[0] == 6):
        rot_3x3, _ = cv2.Rodrigues(se3[3:6]) 
        T = np.zeros((4,4)) 
        T[0:3, 0:3] = rot_3x3 
        T[0:3, 3] = se3[0:3] 
        T[3, 3] = 1
        return T
    else:
        rvec, _ = cv2.Rodrigues(se3[0:3, 0:3])
        tvec = se3[0:3, 3].squeeze() 
        return np.hstack((tvec, rvec.squeeze()))

# Note, this filter is modified from some particle filter code that I jointly
# developed with rlybrdgs@umich.edu
class ParticleFilter():
    def __init__(self):
        self.num_samples = 1000
        self.resampling_noise = 0.0008
        self.sensor_pos_variance = 0.005
        self.sensor_rot_variance = 0.01 
        self.is_setup = False
        self.sensor_cov = np.array([
                [self.sensor_pos_variance, 0, 0, 0, 0, 0 ], 
                [0, self.sensor_pos_variance, 0, 0, 0, 0], 
                [0, 0, self.sensor_pos_variance, 0, 0, 0],
                [0, 0, 0, self.sensor_rot_variance, 0,  0], 
                [0, 0, 0, 0, self.sensor_rot_variance, 0], 
                [0, 0, 0, 0, 0, self.sensor_rot_variance]
            ]) 

    def setup(self, mu_start):
        # particles are N x 6 where each 6 vec represents SE3 pose via <xyz, rod>
        mu_start = SE3(mu_start)
        self.particles = np.vstack([mu_start for _ in range(self.num_samples)])
        self.is_setup = True

    # sample the sensor reading from a distribution centered at position to get its probability
    def particle_weight(self, particle_pose, sensor_pose):
        w_pos = scipy.stats.multivariate_normal.pdf(sensor_pose, mean=particle_pose, cov=self.sensor_cov)
        return w_pos


    # take action (Rod) and sensor_pose (Rod)
    def get_next_state(self, action, dt, sensor_pose=None):
        #print(dt)
        # make action into matrix via Exp  
        action_mat = SE3(action * dt)
        # convert particles into matrix and apply action, then convert back to Rod
        new_particles = np.array([SE3(action_mat @ SE3(particle) ) for particle in self.particles])# self.particles 

        # find weight of each particle by seeing how well it matches the sensor reading
        weights = np.ones(new_particles.shape[0])
        if(sensor_pose is not None):
            for i in range(new_particles.shape[0]):
                weights[i] = self.particle_weight(
                    new_particles[i], SE3(sensor_pose))

        # normalize particle weights so they sum to 1
        weights /= np.sum(weights)

        # use the weighted average of all particles as the state estimate
        #best_estimate = np.average(new_particles, axis=0, weights=weights)
        # use max weight particle as state estimate        
        best_estimate = new_particles[np.argmax(weights)]

        # resample particles with probability equal to their weights
        idx = np.random.choice(
            weights.shape[0], new_particles.shape[0], list(weights))

        # add a small amount of gaussian noise to sampled particles to avoid duplicates
        #noises = np.zeros((self.num_samples, 6)) # 
        noises = np.random.multivariate_normal(mean=np.zeros(6), cov=np.eye(6)*self.resampling_noise, size=(self.num_samples))  
        resampled_particles = np.vstack(
            [new_particles[i] for i in idx]) + noises

        self.particles = resampled_particles 
        #for particle in resampled_particles:
        #    particle_mat = SE3(particle)
        #    rot = particle_mat[0:3, 0:3]
        #    trans = particle_mat[0:3, 3]
        #    #draw_pose(trans, rot, mat=True)
        #    draw_sphere_marker(trans, 0.05, (1.0, 0.0, 0.0, 1.0)) 
        #    print(particle)

        return SE3(best_estimate)
