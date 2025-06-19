
import transformations as tf

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
