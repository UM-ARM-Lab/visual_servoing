import hjson
import time
import pybullet as p
import cv2

def run_servoing(pbvs, camera, victor, config):

    p.setTimeStep(1/config['sim_hz'])
    sim_steps_per_pbvs = int(config['sim_hz']/config['pbvs_hz']) 
    start_time = time.time()

    while(True):
        # check if timeout exceeded
        if(time.time() - start_time > config['timeout']):
            return 0

        # check if error is low enough to terminate

        # do visual servo
        rgb, depth, seg = camera.get_image(True)
        rgb_edit = rgb[..., [2, 1, 0]].copy()
        if(config['vis']):
            cv2.imshow("Camera", cv2.resize(rgb_edit, (1280 // 5, 800 // 5))) 
        
        ctrl, Twe = pbvs.do_pbvs(depth, seg, to_homogenous(target), victor.get_arm_jacobian('left'),
                                victor.get_jacobian_pinv('left'))
        victor.psuedoinv_ik_controller("left", ctrl)
        
        for _ in range(sim_steps_per_pbvs):
            p.stepSimulation()
 


def main():
    # Loads hjson config to do visual servoing with
    config_file = open('config.hjson', 'r')
    config_text = config_file.read()
    config = hjson.loads(config_text)

    servo_configs = config['servo_configs']
    for servo_config in servo_configs:
        


main()