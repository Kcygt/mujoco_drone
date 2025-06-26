import os
import mujoco
import mujoco.viewer
import numpy as np
import transformations as tf
import time
import pygame 
from pid_controller import PIDController
from se3controller import SE3Controller
from state_estimator import StateEstimator
from user_command import UserCommand


class Drone:
    def __init__(self):
        self.xml_path = os.path.join("skydio_x2", "scene_x2.xml")
        print(f"xml_path: {self.xml_path} ")

        self.m = mujoco.MjModel.from_xml_path(self.xml_path)
        self.d = mujoco.MjData(self.m)

        self.user_cmd = UserCommand()
        self.state_estimator = StateEstimator(self.m, self.d)

        self.stabilisation_controller = PIDController(
            z_des=0.5, rpy_setpoint=[0, 0, 0], state_estimator=self.state_estimator
        )
        self.se3_controller = SE3Controller(
            state_estimator=self.state_estimator, user_cmd=self.user_cmd
        )

        # Initialize pygame and joystick
        pygame.init()
        pygame.joystick.init()

        self.joystick = None
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print("Xbox Controller Connected")
        else:
            print("No controller found")

    def set_pos(self, pos):
        self.d.qpos[:3] = pos

    def set_quat(self, quat):
        self.d.qpos[3:7] = quat

    def update_user_command_from_joystick(self):
        pygame.event.pump()

        if self.joystick:
            deadzone = 0.1

            axis_x = self.joystick.get_axis(0)  # Left stick X
            axis_y = self.joystick.get_axis(1)  # Left stick Y
            axis_z = self.joystick.get_axis(4)  # Right stick Y

            if abs(axis_x) > deadzone:
                self.user_cmd.x += axis_x * 0.01
                self.user_cmd.x = np.clip(self.user_cmd.x, -1.0, 1.0)

            if abs(axis_y) > deadzone:
                self.user_cmd.y += -axis_y * 0.01
                self.user_cmd.y = np.clip(self.user_cmd.y, -1.0, 1.0)

            if abs(axis_z) > deadzone:
                self.user_cmd.z += -axis_z * 0.01
                self.user_cmd.z = np.clip(self.user_cmd.z, 0.1, 2.0)

            # Buttons for rotation (yaw)
            lb = self.joystick.get_button(4)  # LB
            rb = self.joystick.get_button(5)  # RB

            yaw_rate = 1  # adjust sensitivity

            if lb:
                self.user_cmd.yaw -= yaw_rate

            if rb:
                self.user_cmd.yaw += yaw_rate



    def __call__(self):
        self.update_user_command_from_joystick()

        # Use yaw as heading direction vector in XY plane
        heading_des = [np.cos(self.user_cmd.yaw)*0.01, np.sin(self.user_cmd.yaw)*0.01, 0]

        f, M = self.se3_controller.tracking_control(
            pos_des=[self.user_cmd.x, self.user_cmd.y, self.user_cmd.z],
            heading_des=heading_des,
        )

        motor_cmd = self.cal_motor_cmd(f, M[0], M[1], M[2])
        self.set_motor_cmd(motor_cmd)


    def cal_motor_cmd(self, T, tau_x, tau_y, tau_z):
        dx, dy, k = 0.14, 0.18, 0.1
        gain = 4
        f1 = (T - tau_x / dx + tau_y / dy + tau_z / k) / gain
        f2 = (T - tau_x / dx - tau_y / dy - tau_z / k) / gain
        f3 = (T + tau_x / dx + tau_y / dy - tau_z / k) / gain
        f4 = (T + tau_x / dx - tau_y / dy + tau_z / k) / gain
        return np.array([f1, f2, f3, f4])

    def set_motor_cmd(self, motor_cmd):
        self.d.ctrl[:4] = motor_cmd


def main():
    drone = Drone()

    with mujoco.viewer.launch_passive(drone.m, drone.d) as viewer:
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
        
        # cam_id = mujoco.mj_name2id(drone.m, mujoco.mjtObj.mjOBJ_CAMERA, "drone_cam")
        # viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        # viewer.cam.fixedcamid = cam_id

        while viewer.is_running():
            print(f"Time: {drone.d.time:.2f}s, Altitude Command: {drone.user_cmd.z:.2f}")

            drone()

            mujoco.mj_step(drone.m, drone.d)
            viewer.sync()
            time.sleep(drone.m.opt.timestep)


if __name__ == "__main__":
    main()
