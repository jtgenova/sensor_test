import numpy as np
import matplotlib.pyplot as plt

# define parameters
slope = 0.001
depth = 350 # mm
crack_width = 20 # mm
crack_depth = 8 # mm
rect_width = 80 # mm
crack_starting_point = 10 # mm
starting_point = 50 # mm
noise_std_dev = 0.01  # Standard deviation of the noise

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
print(crack_center_x, crack_center_y)
print(y)


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