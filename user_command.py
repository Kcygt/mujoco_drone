from gamepad_reader import GamepadReader


class UserCommand:
    def __init__(self):
        self.gamepad_reader = GamepadReader()
        self.gamepad_reader.start()
        self.x = 0.0  # start roughly centered
        self.y = 0.0
        self.z = 0.1
        self.yaw = 0.0  # desired yaw rate or angle

    def get_input(self):
        axes = self.gamepad_reader.get_axes()
        buttons = self.gamepad_reader.get_buttons()
        return axes, buttons

    def stop(self):
        self.gamepad_reader.stop()

    def throttle(self, lim=1.0, deadzone=0.05):
        axes, _ = self.get_input()
        val = -axes[1] * lim
        return 0 if abs(val) < deadzone else val

    def yaw(self, lim=1.0):
        axes, _ = self.get_input()
        return -axes[0] * lim

    def roll(self, lim=1.0):
        axes, _ = self.get_input()
        return -axes[3] * lim

    def pitch(self, lim=1.0):
        axes, _ = self.get_input()
        return -axes[4] * lim
