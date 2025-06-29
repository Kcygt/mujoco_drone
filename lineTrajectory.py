import time
import numpy as np
import mujoco
import mujoco.viewer
from simple_pid import PID
import transformations as tf
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Utility functions for SE(3) controller
# -----------------------------------------------------------------------------
def hat(omega):
    return np.array([
        [0, -omega[2], omega[1]],
        [omega[2], 0, -omega[0]],
        [-omega[1], omega[0], 0]
    ])

def vee(omega_hat):
    return np.array([
        omega_hat[2, 1],
        omega_hat[0, 2],
        omega_hat[1, 0]
    ])

# -----------------------------------------------------------------------------
# State estimation
# ------------------------a-----------------------------------------------------
class StateEstimator:
    def __init__(self, m, d, base_name="x2"):
        self.m = m
        self.d = d
        self.base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, base_name)
        if self.base_id == -1:
            raise ValueError(f"No body named '{base_name}' in model.")

    @property
    def base_pos(self):
        return self.d.qpos[:3].copy()

    @property
    def base_quat(self):
        return self.d.qpos[3:7].copy()

    @property
    def R(self):
        return tf.quaternion_matrix(self.base_quat)[:3, :3]

    @property
    def base_vel_lin_global(self):
        return self.d.qvel[:3].copy()

    @property
    def base_vel_ang_local(self):
        return self.d.qvel[3:6].copy()

# -----------------------------------------------------------------------------
# PID controller (altitude + attitude)
# -----------------------------------------------------------------------------
class PIDController:
    def __init__(self, z_des=0.5, rpy_setpoint=[0,0,0], state_estimator=None):
        self.se = state_estimator
        # self.pid_alt = PID(20, 0.5, 1.25, setpoint=z_des)
        # self.pid_roll = PID(20, 0.5, 1.25, setpoint=rpy_setpoint[0], output_limits=(-1,1))
        # self.pid_pitch = PID(20, 0.5, 1.25, setpoint=rpy_setpoint[1], output_limits=(-1,1))
        # self.pid_yaw = PID(6, 0, 1.25, setpoint=rpy_setpoint[2], output_limits=(-3,3))

    def compute_control(self):
        z = self.se.base_pos[2]
        roll = tf.euler_from_quaternion(self.se.base_quat)[0]
        pitch = tf.euler_from_quaternion(self.se.base_quat)[1]
        yaw = tf.euler_from_quaternion(self.se.base_quat)[2]

        T = (self.pid_alt(z) - 9.81) * 1.0 \
            / (np.cos(roll)*np.cos(pitch))
        tau_roll = self.pid_roll(roll)
        tau_pitch = self.pid_pitch(pitch)
        tau_yaw = self.pid_yaw(yaw)

        return T, tau_roll, tau_pitch, tau_yaw

# -----------------------------------------------------------------------------
# SE(3) controller
# -----------------------------------------------------------------------------
class SE3Controller:
    def __init__(self, state_estimator, user_cmd=None):
        self.se = state_estimator
        self.user_cmd = user_cmd
        self.m = 1.325
        self.g = np.array([0,0,-9.81])
        self.k_pos, self.k_vel = 5.0, 4
        self.k_rot, self.k_omega = 30.0, 6.0
        self.base_I = np.diag([0.06,0.04,0.02])

    def tracking_waypoint(self, pos_des, heading_des,
                          vel_des=None, acc_des=None,
                          omega_des_local=None, omega_dot_des_local=None):
        vel_des = np.zeros(3) if vel_des is None else vel_des
        acc_des = np.zeros(3) if acc_des is None else acc_des
        omega_des_local = np.zeros(3) if omega_des_local is None else omega_des_local
        omega_dot_des_local = np.zeros(3) if omega_dot_des_local is None else omega_dot_des_local

        pos = self.se.base_pos
        vel = self.se.base_vel_lin_global
        R = self.se.R
        omega = self.se.base_vel_ang_local

        acc_cmd = self.k_pos*(pos_des-pos) + self.k_vel*(vel_des-vel) + acc_des
        T_cmd = self.m*acc_cmd - self.m*self.g
        T_body = R.T @ T_cmd
        f = T_body[2]

        zd = T_cmd / (np.linalg.norm(T_cmd)+1e-6)
        heading_des = np.array(heading_des)
        if np.linalg.norm(np.cross(zd, heading_des)) < 1e-3:
            heading_des = np.array([1.0,0.0,0.0])
        heading_des /= np.linalg.norm(heading_des)
        yd = np.cross(zd, heading_des)
        yd /= (np.linalg.norm(yd)+1e-6)
        xd = np.cross(yd, zd)
        Rd = np.column_stack([xd, yd, zd])

        err_rot = 0.4 * vee(R.T @ Rd - Rd.T @ R)
        err_omega = omega_des_local - omega
        M = self.k_rot*err_rot + self.k_omega*err_omega + hat(omega)@self.base_I@omega

        return f, M

# -----------------------------------------------------------------------------
# UserCommand (hook for manual inputs)
# -----------------------------------------------------------------------------
class UserCommand:
    def __init__(self):
        pass
    def yaw(self):
        return 0.0  #

# -----------------------------------------------------------------------------
# Trajectory generation: straight line from (0,0,1) to (1,1,1)
# -----------------------------------------------------------------------------
def generate_straight_trajectory(duration, dt):
    t = np.arange(0, duration, dt)
    start = np.array([0., 0., 1.])
    end = np.array([1., 1., 1.])
    pos = np.outer(t/duration, end - start) + start
    vel = np.tile((end - start) / duration, (len(t),1))
    acc = np.zeros_like(pos)
    return t, pos, vel, acc

# -----------------------------------------------------------------------------
# Main Drone class
# -----------------------------------------------------------------------------
class Drone:
    def __init__(self, xml_path="skydio_x2/scene_x2.xml"):
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.d = mujoco.MjData(self.m)
        self.est = StateEstimator(self.m, self.d)
        self.user = UserCommand()
        self.pid = PIDController(z_des=1.0, rpy_setpoint=[0,0,0], state_estimator=self.est)
        self.se3 = SE3Controller(self.est, self.user)
        
        # Trajectory generation
        self.traj_t, self.traj_pos, self.traj_vel, self.traj_acc = generate_straight_trajectory(duration=3, dt=self.m.opt.timestep)
        
        # Timing
        self.start_time = None
        self.takeoff_height = 1.0
        
        # Logs for plotting after sim
        self.log_time = []
        self.log_pos = []
        self.log_pos_des = []

    def cal_motor_cmd(self, T, tx, ty, tz):
        dx, dy, k = 0.14, 0.18, 0.1
        g = 4.0
        f1 = (T - tx/dx + ty/dy + tz/k)/g
        f2 = (T - tx/dx - ty/dy - tz/k)/g
        f3 = (T + tx/dx + ty/dy - tz/k)/g
        f4 = (T + tx/dx - ty/dy + tz/k)/g
        return np.clip([f1, f2, f3, f4], 0, None)

    def __call__(self, dt):
        if self.start_time is None:
            self.start_time = time.time()
        elapsed = time.time() - self.start_time
        
        pos = self.est.base_pos
        
        # Log current time and position
        self.log_time.append(elapsed)
        self.log_pos.append(pos.copy())

        if elapsed < 5.0:
            # Takeoff phase: ascend vertically to takeoff_height
            pos_des = np.array([pos[0], pos[1], self.takeoff_height])
            yaw_des = 0.0
            heading = [np.cos(yaw_des), np.sin(yaw_des), 0]
            vel_des = np.zeros(3)
            acc_des = np.zeros(3)
            f, M = self.se3.tracking_waypoint(pos_des, heading, vel_des, acc_des)
            motor = self.cal_motor_cmd(f, *M)
            self.d.ctrl[:4] = motor
        
        elif elapsed < 7.0:
            # Hover phase at takeoff height
            pos_des = np.array([pos[0], pos[1], self.takeoff_height])
            yaw_des = 0.0
            heading = [np.cos(yaw_des), np.sin(yaw_des), 0]
            vel_des = np.zeros(3)
            acc_des = np.zeros(3)
            f, M = self.se3.tracking_waypoint(pos_des, heading, vel_des, acc_des)
            motor = self.cal_motor_cmd(f, *M)
            self.d.ctrl[:4] = motor
        
        else:
            # Trajectory following phase
            t_now = elapsed - 7.0
            idx = np.searchsorted(self.traj_t, t_now)
            if idx >= len(self.traj_t):
                idx = len(self.traj_t) - 1
            
            pos_des = self.traj_pos[idx]
            vel_des = self.traj_vel[idx]
            acc_des = self.traj_acc[idx]
            yaw_des = 0.0
            heading = [np.cos(yaw_des), np.sin(yaw_des), 0]
            
            # Log desired position
            self.log_pos_des.append(pos_des.copy())
            
            f, M = self.se3.tracking_waypoint(pos_des, heading, vel_des, acc_des)
            motor = self.cal_motor_cmd(f, *M)
            self.d.ctrl[:4] = motor
        
        # For takeoff & hover phases also log desired position for plot consistency
        if elapsed < 7.0:
            self.log_pos_des.append(pos_des.copy())
        
        # Print elapsed time on the same console line
        print(f"Elapsed simulation time: {elapsed:.2f} s", end='\r')

    def run(self):
        with mujoco.viewer.launch_passive(self.m, self.d) as vw:
            vw.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
            vw.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
            while vw.is_running():
                dt = self.m.opt.timestep
                self(dt)
                mujoco.mj_step(self.m, self.d)
                vw.sync()
                time.sleep(dt)

        # After simulation ends, plot results
        self.plot_results()

    def plot_results(self):
        t = np.array(self.log_time)
        pos = np.array(self.log_pos)
        pos_des = np.array(self.log_pos_des)
        
        plt.figure(figsize=(10, 8))
        
        labels = ['X', 'Y', 'Z']
        for i in range(3):
            plt.subplot(3,1,i+1)
            plt.plot(t, pos[:,i], label='Actual')
            plt.plot(t, pos_des[:,i], '--', label='Desired')
            plt.ylabel(f'{labels[i]} (m)')
            plt.grid()
            plt.legend()
        plt.xlabel('Time (s)')
        plt.suptitle('Drone Position: Actual vs Desired')
        plt.show()

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    drone = Drone()
    drone.run()
