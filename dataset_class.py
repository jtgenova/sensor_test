import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CrackDataset(Dataset):
    def __init__(self, num_samples=100):
        self.x_data, self.y_data, self.centers = self.generate_data(num_samples)
        self.x = torch.tensor(self.x_data, dtype=torch.float32)  # Convert x_data to tensor
        self.y = torch.tensor(self.y_data, dtype=torch.float32)  # Convert y_data to tensor
        self.centers = torch.tensor(self.centers, dtype=torch.float32)  # Convert centers to tensor

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.centers[idx]

    def generate_data(self, num_samples):
        x_data = []
        y_data = []
        centers = []
        for _ in range(num_samples):
            slope = np.random.uniform(0.0001, 0.002)
            depth = np.random.normal(300, 50)
            crack_width = np.random.uniform(1, 10)
            crack_depth = np.random.uniform(5, 10)
            rect_width = np.random.uniform(50, 75)
            crack_starting_point = np.random.uniform(0, 20)
            starting_point = np.random.uniform(-100, 100)
            noise_std_dev = np.random.uniform(0.005, 0.01)

            x = np.linspace(-150, 150, 3000)
            y = slope * x + depth
            y_noise = y + np.random.normal(0, noise_std_dev, x.shape)
            y_noise = np.where((x >= starting_point) & (x < starting_point + rect_width), y_noise - crack_depth, y_noise)
            y_noise = np.where((x >= starting_point + crack_starting_point) &
                               (x < starting_point + crack_starting_point + crack_width), y_noise + crack_depth, y_noise)
            
            crack_center_x = starting_point + crack_starting_point + (crack_width / 2)
            crack_center_y = slope * crack_center_x + depth

            x_data.append(x)
            y_data.append(y_noise)
            centers.append([crack_center_x, crack_center_y])

        return np.array(x_data), np.array(y_data), np.array(centers)
    
if __name__=="__main__":
    dataset = CrackDataset(num_samples=100)
    print(dataset[0])