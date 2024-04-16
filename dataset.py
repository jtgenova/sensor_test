import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from generate_data import DataGenerator


class CrackDataset(Dataset):
    def __init__(self, y_data, outputs, widths, depths):
        """
        Initialization method for the dataset.
        
        Parameters:
        y_data (numpy.ndarray): The array of input data.
        outputs (numpy.ndarray): Binary segmentation maps for the crack locations.
        widths (numpy.ndarray): Continuous values indicating the width of the cracks.
        depths (numpy.ndarray): Continuous values indicating the depth of the cracks.
        """
        self.y_data = torch.from_numpy(y_data).float()
        self.outputs = torch.from_numpy(outputs).float()
        self.widths = torch.from_numpy(widths).float()
        self.depths = torch.from_numpy(depths).float()

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset at the specified index.
        
        Returns:
        Tuple[Tensor]: Tuple containing the input data, binary outputs, widths, and depths.
        """
        return self.y_data[idx].unsqueeze(0), self.outputs[idx], self.widths[idx], self.depths[idx]


# if __name__ == "__main__":
#     generator = DataGenerator(num_samples=100)
#     x_data, y_data, centers, center_indices, output = generator.generate_data()

#     y_train, y_val, output_train, output_val = train_test_split(
#         y_data, output, test_size=0.2, random_state=42)

#     train_dataset = CrackDataset(y_train, output_train)
#     val_dataset = CrackDataset(y_val, output_val)

#     train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

#     print(f"Train Dataset: {len(train_dataset)} samples")
#     print(f"Validation Dataset: {len(val_dataset)} samples")
