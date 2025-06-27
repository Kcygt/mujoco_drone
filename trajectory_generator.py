import numpy as np
from scipy.interpolate import CubicSpline


class TrajectoryGenerator:
    def __init__(self, waypoints, total_time=20.0):
        self.waypoints = np.array(waypoints)
        self.total_time = total_time
        self.num_segments = len(waypoints) - 1
        self.segment_times = np.linspace(0, total_time, len(waypoints))

        # Separate waypoints
        self.positions = self.waypoints[:, :3]
        self.yaws = self.waypoints[:, 3]

        # Create splines for x, y, z
        self.spline_x = CubicSpline(self.segment_times, self.positions[:, 0])
        self.spline_y = CubicSpline(self.segment_times, self.positions[:, 1])
        self.spline_z = CubicSpline(self.segment_times, self.positions[:, 2])
        self.spline_yaw = CubicSpline(self.segment_times, self.yaws)

    def get_desired_state(self, t):
        t = min(max(t, 0.0), self.total_time)  # Clamp time within bounds

        pos = np.array(
            [
                self.spline_x(t),
                self.spline_y(t),
                self.spline_z(t),
            ]
        )

        vel = np.array(
            [
                self.spline_x.derivative()(t),
                self.spline_y.derivative()(t),
                self.spline_z.derivative()(t),
            ]
        )

        acc = np.array(
            [
                self.spline_x.derivative(nu=2)(t),
                self.spline_y.derivative(nu=2)(t),
                self.spline_z.derivative(nu=2)(t),
            ]
        )

        yaw = self.spline_yaw(t)
        heading = np.array([np.cos(yaw), np.sin(yaw), 0])

        return pos, vel, acc, heading
