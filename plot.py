import matplotlib.pyplot as plt
import numpy as np

# Segment 1: 0 < x < 2
x1 = np.linspace(0, 2, 100)
M1 = -5 * x1

# Segment 2: 2 < x < 4
x2 = np.linspace(2, 4, 100)
M2 = -x2**2 - x2 - 4

# Segment 3: 4 < x < 6
x3 = np.linspace(4, 6, 100)
M3 = 12 * x3 - 72

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(x1, M1, label="0 < x < 2", color='blue')
plt.plot(x2, M2, label="2 < x < 4", color='green')
plt.plot(x3, M3, label="4 < x < 6", color='red')

# Mark key points
plt.scatter([0, 2, 4, 6], [0, -10, -24, 0], color='black')

# Format plot
plt.title("Bending Moment Diagram (0 < x < 6)")
plt.xlabel("x (m)")
plt.ylabel("Bending Moment M(x) [kNm]")
plt.grid(True)
plt.axhline(0, color='black', linewidth=1)
plt.legend()
plt.tight_layout()
plt.show()