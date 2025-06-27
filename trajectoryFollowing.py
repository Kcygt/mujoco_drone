import os
import mujoco
import mujoco.viewer
import numpy as np
import time
import pygame
from pid_controller import PIDController
from se3controller import SE3Controller
from state_estimator import StateEstimator
from user_command import UserCommand
from trajectory_generator import TrajectoryGenerator  # <- Add this import


class Drone:
    def __init__(self):
        self.xml_path = os.path.join("skydio_x2", "scene_x2.xml")
        print(f"xml_path: {self.xml_path}")

        # Define waypoints BEFORE using them
        self.waypoints = [[0, 0, 1.0, 0.0]]  # takeoff point

        # Now safe to pass waypoints to the generator
        self.trajectory_generator = TrajectoryGenerator(self.waypoints, total_time=20.0)
        self.start_time = time.time()
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

        self.current_wp_idx = 0
        self.waypoint_start_time = time.time()

    def set_pos(self, pos):
        self.d.qpos[:3] = pos

    def set_quat(self, quat):
        self.d.qpos[3:7] = quat

    def __call__(self, dt):
        current_time = time.time() - self.start_time
        pos_des, vel_des, acc_des, heading_des = (
            self.trajectory_generator.get_desired_state(current_time)
        )

        thrust, moments = self.se3_controller.tracking_trajectory(
            pos_des=pos_des,
            vel_des=vel_des,
            heading_des=heading_des,
        )

        motor_cmd = self.cal_motor_cmd(thrust, moments[0], moments[1], moments[2])
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

        while viewer.is_running():
            dt = drone.m.opt.timestep
            drone(dt)
            mujoco.mj_step(drone.m, drone.d)
            viewer.sync()
            time.sleep(dt)


if __name__ == "__main__":
    main()
