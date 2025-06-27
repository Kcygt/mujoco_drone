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


class Drone:
    def __init__(self):
        self.xml_path = os.path.join("skydio_x2", "scene_x2.xml")
        print(f"xml_path: {self.xml_path}")

        self.m = mujoco.MjModel.from_xml_path(self.xml_path)
        self.d = mujoco.MjData(self.m)
        # trajectory generator is not used in this version
        self.trajectory_log = (
            []
        )  # To store [timestamp, position, velocity, acceleration]
        self.last_pos = np.zeros(3)
        self.last_vel = np.zeros(3)
        self.last_time = time.time()

        self.user_cmd = UserCommand()
        self.state_estimator = StateEstimator(self.m, self.d)

        self.stabilisation_controller = PIDController(
            z_des=0.5, rpy_setpoint=[0, 0, 0], state_estimator=self.state_estimator
        )
        self.se3_controller = SE3Controller(
            state_estimator=self.state_estimator, user_cmd=self.user_cmd
        )

        # Waypoints: [x, y, z, yaw]
        self.waypoints = [
            [0, 0, 1.0, 0.0],  # takeoff point
            [1, 0, 1.0, 0.0],  # waypoint 1
            [1, 1, 1.0, 0.0],  # waypoint 2
            [0, 1, 1.0, 0.0],  # waypoint 3
            [0, 0, 1.0, 0.0],  # home
        ]
        self.current_wp_idx = 0
        self.waypoint_start_time = time.time()

    def set_pos(self, pos):
        self.d.qpos[:3] = pos

    def set_quat(self, quat):
        self.d.qpos[3:7] = quat

    def __call__(self, dt):
        # Get current estimated position
        pos = self.state_estimator.get_position()  # make sure this returns [x, y, z]

        if self.current_wp_idx < len(self.waypoints):
            wp = self.waypoints[self.current_wp_idx]
            pos_error = np.linalg.norm(pos - wp[:3])

            if pos_error < 0.3 or time.time() - self.waypoint_start_time > 10:
                print(f"Reached waypoint {self.current_wp_idx}: {wp}")
                self.current_wp_idx += 1
                self.waypoint_start_time = time.time()

            if self.current_wp_idx < len(self.waypoints):
                # Continue to next waypoint
                wp = self.waypoints[self.current_wp_idx]
                pos_des = wp[:3]
                yaw_des = wp[3]
            else:
                # Final hover at last waypoint
                pos_des = wp[:3]
                yaw_des = wp[3]
        else:
            # Mission complete: hover at last waypoint
            wp = self.waypoints[-1]
            pos_des = wp[:3]
            yaw_des = wp[3]

        # Update desired position and heading
        heading_des = [np.cos(yaw_des), np.sin(yaw_des), 0.0]

        f, M = self.se3_controller.tracking_waypoint(
            pos_des=pos_des,
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

        while viewer.is_running():
            dt = drone.m.opt.timestep
            drone(dt)
            mujoco.mj_step(drone.m, drone.d)
            viewer.sync()
            time.sleep(dt)


if __name__ == "__main__":
    main()
