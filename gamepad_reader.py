import pygame
import threading
import time

class GamepadReader:
    def __init__(self, update_freq=50):
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No joystick detected.")
        
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"Joystick name: {self.joystick.get_name()}")

        # Store states
        self.num_axes = self.joystick.get_numaxes()
        self.num_buttons = self.joystick.get_numbuttons()
        self.axis_values = [0.0] * self.num_axes
        self.button_states = [False] * self.num_buttons

        # Thread control
        self.update_freq = update_freq
        self._running = False
        self._thread = threading.Thread(target=self._update_loop, daemon=True)

                # Define labels (based on typical F710 layout in XInput mode)
        self.button_labels = [
            "A", "B", "X", "Y",  # 0-3
            "LB", "RB",          # 4-5
            "Back", "Start",     # 6-7
            "Left Stick", "Right Stick"  # 8-9
        ]

        self.axis_labels = [
            "Left Stick X", "Left Stick Y", "LT",
            "Right Stick X", "Right Stick Y",
             "RT"  # These may show up as axis or button depending on mode
        ]

    def start(self):
        self._running = True
        self._thread.start()

    def stop(self):
        self._running = False
        self._thread.join()

    def _update_loop(self):
        clock = pygame.time.Clock()
        while self._running:
            pygame.event.pump()
            for i in range(self.num_axes):
                self.axis_values[i] = self.joystick.get_axis(i)
            for i in range(self.num_buttons):
                self.button_states[i] = self.joystick.get_button(i)
            clock.tick(self.update_freq)

    def get_axes(self):
        return self.axis_values.copy()

    def get_buttons(self):
        return self.button_states.copy()

    def print_state(self):
        print("=== Axis States ===")
        for i, val in enumerate(self.axis_values):
            label = self.axis_labels[i] if i < len(self.axis_labels) else f"Axis {i}"
            print(f"{label}: {val:.3f}")

        print("=== Button States ===")
        for i, pressed in enumerate(self.button_states):
            label = self.button_labels[i] if i < len(self.button_labels) else f"Button {i}"
            state = "Pressed" if pressed else "Released"
            print(f"{label}: {state}")
        
if __name__ == "__main__":
    reader = GamepadReader(update_freq=20)
    reader.start()

    try:
        while True:
            reader.print_state()
            time.sleep(1/50)
    except KeyboardInterrupt:
        print("Stopping...")
        reader.stop()
        pygame.quit()
