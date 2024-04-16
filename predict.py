import torch
import numpy as np
from generate_data import DataGenerator
import matplotlib.pyplot as plt



class ModelPredictor:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model = self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()  # Set the model to evaluation mode

    def load_model(self, model_path):
        """
        Load the saved model from a specified path.
        """
        model = torch.load(model_path, map_location=self.device)  # Ensure model loads on the right device
        return model

    def preprocess(self, input_data):
        """
        Preprocess the input data to be suitable for the model.
        Assuming input_data is a 1D NumPy array of length 1024.
        """
        # Ensure input_data is a float32 NumPy array, reshape, normalize if necessary
        # input_tensor = torch.from_numpy(input_data.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 1024)
        input_tensor = torch.from_numpy(input_data.astype(np.float32))
        input_tensor = input_tensor.to(self.device)  # Send data to the appropriate device
        return input_tensor

    def predict(self, input_data):
        """
        Perform a prediction using preprocessed data.
        """
        input_tensor = self.preprocess(input_data)
        with torch.no_grad():  # Ensure no gradients are calculated during prediction
            segmentation, width, depth = self.model(input_tensor)
        return segmentation, width, depth

    def postprocess(self, segmentation, width, depth):
        """
        Convert model outputs into a human-readable or application-specific format.
        Example: Convert segmentation to binary mask, extract width and depth values.
        """
        # segmentation = torch.sigmoid(segmentation).squeeze().cpu().numpy()  # Convert to probabilities, then to numpy array
        # segmentation = segmentation.squeeze().cpu().numpy()  # Simply remove singleton dimensions and convert to numpy
        segmentation = segmentation.argmax(dim=1).cpu().numpy()
        # print(segmentation)
        width = width.item()  # Get single scalar value
        depth = depth.item()  # Get single scalar value
        return segmentation, width, depth

    def run_prediction(self, input_data):
        """
        Complete pipeline from input data to processed results.
        """
        segmentation, width, depth = self.predict(input_data)
        return self.postprocess(segmentation, width, depth)
    
if __name__ == "__main__":
    # Generate or load data
    generator = DataGenerator(num_samples=1)
    x_data, y_data, centers, output, crack_width, crack_depth = generator.generate_data()
    x = x_data[0]
    y = y_data[0]
    centers = centers[0]
    output = np.nonzero(output)[1][0]
    crack_width = crack_width[0]
    crack_depth = crack_depth[0]
    # print(f"Centers: {centers}, Output: {output}, Crack Width: {crack_width}, Crack Depth: {crack_depth}")
    # print(f"X: {x[output]}, Y: {y[output]}")
    
    print("#" * 50)
    print("Predicting...")
    predictor = ModelPredictor(model_path='best_full_model.pth', device='cuda')  # Adjust path and device as needed
    input_data = y_data  # Example random input data array
    segmentation, width, depth = predictor.run_prediction(input_data)

    index = segmentation[0]

    print(f"GT Index: {output}, Predicted Index: {index}")
    print(f"GT Center: {x[output], y[output]}, Predicted Center: {x[index]}, {y[index]}")
    print(f"GT Crack Width: {crack_width}, Predicted Crack Width: {width}")
    print(f"GT Crack Depth: {crack_depth}, Predicted Crack Depth: {depth}")
    

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Noisy Data')
    # plt.plot(x, output, label='just center data', color='red')
    plt.xlabel('x')
    plt.ylabel('z')
    # Mark the center of the crack width with 'X'
    plt.plot(x[output], y[output], 'rx', label='Grouth Truth Center', markersize=10)
    plt.plot(x[index], y[index], 'bx', label='Predicted Center', markersize=10)

    plt.legend()
    plt.grid(True)
    plt.show()