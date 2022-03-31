import hjson
import time
import pybullet as p

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
        rgb, 
        
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