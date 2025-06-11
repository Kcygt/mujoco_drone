class PDController:
  def __init__(self, kp, kd, setpoint):
    self.kp = kp
    self.kd = kd
    self.setpoint = setpoint
    self.prev_error = 0

  def compute(self, measured_value, setpoint=None):
    if setpoint is not None:
      self.setpoint = setpoint
    error = self.setpoint - measured_value
    derivative = error - self.prev_error
    output = (self.kp * error) + (self.kd * derivative)
    self.prev_error = error
    return output

class PIDController:
  def __init__(self, kp, ki, kd, setpoint):
    self.kp = kp
    self.ki = ki
    self.kd = kd
    self.setpoint = setpoint
    self.prev_error = 0
    self.integral = 0

  def compute(self, measured_value, setpoint=None):
    if setpoint is not None:
      self.setpoint = setpoint
    error = self.setpoint - measured_value
    self.integral += error
    derivative = error - self.prev_error
    output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
    self.prev_error = error
    return output