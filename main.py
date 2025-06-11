import os
import mujoco
import mujoco.viewer 
import numpy as np
import transformations as tf 
import time

from simple_pid import PID

from gamepad_reader import GamepadReader

def hat(omega):
    """Convert a 3D vector to a 3x3 skew-symmetric matrix."""
    return np.array([
        [0, -omega[2], omega[1]],
        [omega[2], 0, -omega[0]],
        [-omega[1], omega[0], 0]
    ])

def vee(omega_hat):
    """Convert a 3x3 skew-symmetric matrix to a 3D vector."""
    return np.array([
        omega_hat[2,1],
        omega_hat[0,2],
        omega_hat[1,0]
    ])


class UserInput:
    def __init__(self, update_freq=20):
        self.update_freq = update_freq
        self.gamepad_reader = GamepadReader(update_freq=update_freq)
        self.gamepad_reader.start()

    def get_input(self):
        axes = self.gamepad_reader.get_axes()
        buttons = self.gamepad_reader.get_buttons()
        return axes, buttons

    def stop(self):
        self.gamepad_reader.stop()

    def roll(self, lim=1.0):
        axes, _ = self.get_input()
        return -axes[0] * lim  
    
    def pitch(self, lim=1.0):
        axes, _ = self.get_input()
        return axes[1] * lim  
    
    def yaw_rate(self, lim=1.0):
        axes, _ = self.get_input()
        return -axes[3] * lim


class StateEstimator:
    def __init__(self, d):
        self.d = d  # Store the data object for later access

    @property
    def mass(self):
        return 0.325  # Mass of the drone in kg

    @property
    def base_pos(self):
        return self.d.qpos[:3]

    @property
    def base_quat(self):
        return self.d.qpos[3:7]

    @property
    def base_vel(self):
        return self.d.qvel[:6]

    @property
    def base_omega(self):
        return self.base_vel[3:6]

    @property
    def base_acc(self):
        return self.d.qacc[:6]

    @property
    def base_rpy(self):
        return tf.euler_from_quaternion(self.base_quat)
    
    @property
    def alt(self):
        return self.d.qpos[2]
    
    @property
    def roll(self):
        return self.base_rpy[0]
    
    @property
    def pitch(self):
        return self.base_rpy[1]
    
    @property
    def yaw(self):
        return self.base_rpy[2]
    
    @property
    def rotation_matrix(self):
        return tf.quaternion_matrix(self.base_quat)[:3, :3]
    
    @property
    def R(self):
        return self.rotation_matrix


class PIDController:
    def __init__(self, z_des=0.5, rpy_setpoint=[0,0,0], state_estimator=None):
        self.state_estimator = state_estimator
        self.pid_alt = PID(6, 0.5, 1.25, z_des)
        self.pid_roll = PID(6, 0.5, 1.25, setpoint=rpy_setpoint[0], output_limits = (-1,1))
        self.pid_pitch = PID(6, 0.5, 1.25, setpoint=rpy_setpoint[1], output_limits = (-1,1))
        self.pid_yaw = PID(6, 0, 1.25, setpoint=rpy_setpoint[2], output_limits = (-3,3))


    def compute_control(self):
        # Compute control signals based on the current state
        thrust_total = (self.pid_alt(self.state_estimator.alt) + 9.81) * self.state_estimator.mass/(np.cos(self.state_estimator.roll) * np.cos(self.state_estimator.pitch))
        torque_roll = self.pid_roll(self.state_estimator.roll)
        torque_pitch = self.pid_pitch(self.state_estimator.pitch)
        torque_yaw = self.pid_yaw(self.state_estimator.yaw)
        return thrust_total, torque_roll, torque_pitch, torque_yaw

class SE3Controller:
    def __init__(self, state_estimator=None, user_input=None):
        if user_input is not None:
            self.user_input = user_input
        self.se = state_estimator
        # Initialize any necessary parameters for SE(3) control
        self.k_pos = 15.0  # Position gain
        self.k_vel = 5.0   # Velocity gain
        self.k_rot = 30.0   # Rotation gain
        self.k_omega = 3.0 # Angular velocity gain


        self.m = 0.325  # Mass of the drone in kg
        self.g = np.array([0, 0, -9.81])  # Gravitational acceleration in m/s^2

        self.base_inertia = np.array([[5e-2, 0, 0],
                                      [0, 5e-2, 0],
                                      [0, 0, 1e-3]])
  

    def tracking_control(self, pos_des=[0.5, 0.5, 0.5], 
                         vel_des = np.array([0, 0, 0]), 
                         acc_des = np.array([0, 0, 0]), 
                         heading_des=[1,0,0], 
                         omega_des = np.array([0, 0, 0]),
                         omega_dot_des = np.array([0, 0, 0])):


        
        acc_cmd = self.k_pos * (pos_des - self.se.base_pos) + \
                  self.k_vel * (vel_des - self.se.base_vel[:3]) + \
                  acc_des
          
        f_cmd  = self.m * acc_cmd - self.m * self.g  # Total force command
        
        thrust =  np.dot(f_cmd, self.se.R[:, 2])  # Project the force onto the Z-axis of the drone's frame
        
        b3d = f_cmd / (np.linalg.norm(f_cmd) + 1e-6)  # Normalize to get the direction of thrust
     
        b1d = heading_des / np.linalg.norm(heading_des)  # Desired direction of thrust
        b2d = np.cross(b3d, b1d)  # Orthogonal vector to b3d and b1d
        b2d = b2d / (np.linalg.norm(b2d) + 1e-6)
        Rd = np.column_stack((np.cross(b2d, b3d), b2d, b3d))  # Desired rotation matrix
        

        err_rot = 1/2 * vee(Rd.T @ self.se.R - self.se.R.T @ Rd)
        err_omega = self.se.base_vel[3:6] - self.se.R.T @ Rd @ omega_des

        M = - self.k_rot * err_rot - self.k_omega * err_omega \
            + hat(self.se.base_omega) @ self.base_inertia @ self.se.base_omega \
            - self.base_inertia @ (hat(self.se.base_omega) @ self.se.R.T @ Rd @ omega_des - self.se.R.T @ Rd @ omega_dot_des)
        return thrust, M




class Drone: 
    def __init__(self):
        self.xml_path = os.path.join('skydio_x2', 'scene.xml')
        print(f"xml_path: {self.xml_path} ")
        
        self.m = mujoco.MjModel.from_xml_path(self.xml_path)
        self.d = mujoco.MjData(self.m)
        
        self.user_input = UserInput(update_freq=20)
        self.state_estimator = StateEstimator(self.d)
        self.stabilisation_controller = PIDController(z_des=0.5, rpy_setpoint=[0,0,0], state_estimator=self.state_estimator)
        self.se3_controller = SE3Controller(state_estimator=self.state_estimator, user_input=self.user_input)


    def set_pos(self, pos):
        # Set the position of the drone's base
        self.d.qpos[:3] = pos
    def set_quat(self, quat):
        # Set the orientation of the drone's base
        self.d.qpos[3:7] = quat


    def __call__(self):
        print("Base Position:", [f"{x:.3f}" for x in self.state_estimator.base_pos])
        print("Base Quaternion:", [f"{x:.3f}" for x in self.state_estimator.base_quat])
        print("Base RPY:", [f"{x:.3f}" for x in self.state_estimator.base_rpy])

        # thrust_total, torque_roll, torque_pitch, torque_yaw = self.stabilisation_controller.compute_control()

        # thrust_total, torque_roll, torque_pitch, torque_yaw = self.se3_controller.compute_control()
        f, M = self.se3_controller.tracking_control(pos_des=[0.5, 0.5, 0.5], heading_des=[1, 0, 0])


        motor_cmd = self.cal_motor_cmd(f, M[0], M[1], M[2])
        print(f"Motor Commands: {motor_cmd}")

        self.set_motor_cmd(motor_cmd)

    def cal_motor_cmd(self, thrust_total, torque_roll, torque_pitch, torque_yaw):
        motor_cmd = np.array([thrust_total - torque_roll + torque_pitch + torque_yaw,
                            thrust_total - torque_roll - torque_pitch - torque_yaw,
                            thrust_total + torque_roll + torque_pitch - torque_yaw,
                            thrust_total + torque_roll - torque_pitch + torque_yaw])

        return motor_cmd
    
    def set_motor_cmd(self, motor_cmd):
        # Set the motor commands to the actuators
        self.d.ctrl[:4] = motor_cmd


drone = Drone()


def main():
    with mujoco.viewer.launch_passive(drone.m, drone.d) as viewer:

        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY        # Show body frames
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True  # Enable site visualization

        while viewer.is_running():

            print(f"Time: {drone.d.time:.2f}s")
            
            drone()

            mujoco.mj_step(drone.m, drone.d)

            viewer.sync()

            time.sleep(drone.m.opt.timestep) 


if __name__ == "__main__":
    main()

