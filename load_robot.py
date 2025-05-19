import os
import mujoco
import mujoco.viewer 
import numpy as np
import transformations as tf 
import time





class Drone: 
    def __init__(self):
        self.xml_path = os.path.join('skydio_x2', 'scene.xml')
        print(f"xml_path: {self.xml_path} ")
        
        self.m = mujoco.MjModel.from_xml_path(self.xml_path)
        self.d = mujoco.MjData(self.m)

drone = Drone()

with mujoco.viewer.launch_passive(drone.m, drone.d) as viewer:
    while viewer.is_running():
        # print time
        print(f"Time: {drone.d.time:.2f}s")
        
        mujoco.mj_step(drone.m, drone.d)
        viewer.sync()



        time.sleep(drone.m.opt.timestep)  # Sleep to limit the update rate

