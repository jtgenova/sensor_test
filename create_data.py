import numpy as np
import matplotlib.pyplot as plt

# Randomize parameters within given ranges
slope = np.random.uniform(0.0001, 0.002)  # Just an example range
depth = np.random.normal(300, 50)  # Mean at 300, with a standard deviation of 50
crack_width = np.random.uniform(1, 10)  # Between 5mm and 30mm
crack_depth = np.random.uniform(5, 10)  # Between 1mm and 10mm
rect_width = np.random.uniform(50, 75)  # Between 50mm and 100mm
crack_starting_point = np.random.uniform(0, 20)  # Between 0mm and 20mm
starting_point = np.random.uniform(-100, 100)  # Between 40mm and 60mm
noise_std_dev = np.random.uniform(0.005, 0.01)  # Between 0.005 and 0.02

# Create an array of x values from -10 to 10 (3200 points)
x = np.linspace(-150, 150, 3000)

# Calculate the corresponding y values using the equation of the line
y = slope * x + depth

# Add noise before applying the conditions
y_noise = y + np.random.normal(0, noise_std_dev, x.shape)

# Apply the conditions with the noise included
y_noise = np.where((x >= starting_point) & (x < starting_point + rect_width), y_noise - crack_depth, y_noise)
y_noise = np.where((x >= starting_point + crack_starting_point) & 
                   (x < starting_point + crack_starting_point + crack_width), y_noise + crack_depth, y_noise)

# Calculate the x value of the center of the crack width
crack_center_x = starting_point + crack_starting_point + (crack_width / 2)
crack_center_y = slope * crack_center_x + depth  # This assumes the center of the crack is not in the dipped part
print(f"Crack Center: {crack_center_x}, Crack Depth: {crack_center_y}")

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y_noise, label='Noisy Data')
plt.xlabel('x')
plt.ylabel('y')

# Mark the center of the crack width with 'X'
plt.plot(crack_center_x, crack_center_y, 'rx', label='Crack Center', markersize=10)

plt.legend()
plt.grid(True)
plt.show()