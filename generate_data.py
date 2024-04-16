import numpy as np
import matplotlib.pyplot as plt


class DataGenerator:
    def __init__(self, num_samples=100, x_range=(-150, 150), num_points=1024):
        self.num_samples = num_samples
        self.x_range = x_range
        self.num_points = num_points

    def generate_data(self):
            x_data = []
            y_data = []
            centers = []
            center_indices = []
            output = []
            crack_width_list = []
            crack_depth_list = []

            for _ in range(self.num_samples):
                slope = np.random.uniform(0.0001, 0.002)
                depth = np.random.normal(300, 50)
                crack_width = np.random.uniform(1, 10)
                crack_depth = np.random.uniform(5, 10)
                rect_width = np.random.uniform(50, 100)
                crack_starting_point = np.random.uniform(5, 50)
                starting_point = np.random.uniform(-100, 100)
                noise_std_dev = np.random.uniform(0.005, 0.01)

                x = np.linspace(self.x_range[0], self.x_range[1], self.num_points)
                y = slope * x + depth

                y_noise = y + np.random.normal(0, noise_std_dev, x.shape)
                y_noise = np.where((x >= starting_point) & (x < starting_point + rect_width), y_noise - crack_depth, y_noise)
                y_noise = np.where((x >= starting_point + crack_starting_point) &
                                (x < starting_point + crack_starting_point + crack_width), y_noise + crack_depth, y_noise)
                
                crack_center_x = starting_point + crack_starting_point + (crack_width / 2)
                crack_center_y = slope * crack_center_x + depth

                # Calculate the index of the center_x within the x array
                center_index = np.argmin(np.abs(x - crack_center_x))  # Find the closest x index to the crack_center_x

                # need return list of y_data with all zeros except the center index
                out = np.zeros_like(y_noise)
                out[center_index] = 1

                x_data.append(x)
                y_data.append(y_noise)
                centers.append([crack_center_x, crack_center_y])
                center_indices.append(center_index)  # Append the index of the center
                output.append(out)
                crack_width_list.append(crack_width)
                crack_depth_list.append(crack_depth)



            return np.array(x_data), np.array(y_data), np.array(centers), np.array(output), np.array(crack_width_list), np.array(crack_depth_list)

    

# if __name__=="__main__":
#     generator = DataGenerator(num_samples=5)
#     x_data, y_data, centers, idx, out = generator.generate_data()
#     x = x_data[0]
#     y = y_data[0]
#     crack_center_x, crack_center_y = centers[0]
#     index = idx[0]
#     output = out[0]
#     print(f"Index: {index}, x: {x[index]}, z: {y[index]}")
    
#     print(f"Center: {centers[0]}")
#     # Create the plot
#     plt.figure(figsize=(10, 6))
#     plt.plot(x, y, label='Noisy Data')
#     # plt.plot(x, output, label='just center data', color='red')
#     plt.xlabel('x')
#     plt.ylabel('z')

#     # Mark the center of the crack width with 'X'
#     plt.plot(crack_center_x, crack_center_y, 'rx', label='Crack Center', markersize=10)

#     plt.legend()
#     plt.grid(True)
#     plt.show()