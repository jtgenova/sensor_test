import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import matplotlib.pyplot as plt
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import os

# Assuming dataset.py and model.py are in the same directory or properly added to the Python path
from dataset import CrackDataset
from generate_data import DataGenerator
from model import MultiOutputCNN, FocalLoss

def main(batch_size, lr, epochs):
    # Generate or load data
    generator = DataGenerator(num_samples=10000)
    x_data, y_data, centers, output, crack_width, crack_depth = generator.generate_data()
    
    # Split data into training and validation sets
    y_train, y_val, output_train, output_val, width_train, width_val, depth_train, depth_val = train_test_split(
        y_data, output, crack_width, crack_depth, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = CrackDataset(y_train, output_train, width_train, depth_train)
    val_dataset = CrackDataset(y_val, output_val, width_val, depth_val)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train Dataset: {len(train_dataset)} samples")
    print(f"Validation Dataset: {len(val_dataset)} samples")

    # Model setup
    model = MultiOutputCNN()
    criterion_segmentation = FocalLoss()
    criterion_width = nn.MSELoss()
    criterion_depth = nn.MSELoss()
    best_val_loss = float('inf')  # Initialize best validation loss for checkpoint

    optimizer = Adam(model.parameters(), lr=lr)

    # Train the model
    epochs = epochs   
    train_losses, val_losses = [], []

    # Initialize lists to keep track of losses
    train_losses_segmentation, val_losses_segmentation = [], []
    train_losses_width, val_losses_width = [], []
    train_losses_depth, val_losses_depth = [], []

    print(f"Model with batch_size = {batch_size}, learning_rate = {lr}, epochs = {epochs}")
    for epoch in range(epochs):
        # scheduler.step()
        model.train()
        total_train_loss, total_train_loss_segmentation, total_train_loss_width, total_train_loss_depth = 0, 0, 0, 0
        
        for inputs, segmentation_targets, width_targets, depth_targets in train_loader:
            optimizer.zero_grad()
            segmentation_pred, width_pred, depth_pred = model(inputs)  # Inputs are already properly shaped
            loss_segmentation = criterion_segmentation(segmentation_pred, segmentation_targets)
            loss_width = criterion_width(width_pred.squeeze(-1), width_targets)
            loss_depth = criterion_depth(depth_pred.squeeze(-1), depth_targets)

            total_loss = loss_segmentation + loss_width + loss_depth  # Aggregate losses from all tasks
            total_train_loss += total_loss.item()
            total_loss.backward()
            optimizer.step()

            total_train_loss_segmentation += loss_segmentation.item()
            total_train_loss_width += loss_width.item()
            total_train_loss_depth += loss_depth.item()

        # Calculate average losses for the current epoch
        avg_train_loss_segmentation = total_train_loss_segmentation / len(train_loader)
        avg_train_loss_width = total_train_loss_width / len(train_loader)
        avg_train_loss_depth = total_train_loss_depth / len(train_loader)

        train_losses_segmentation.append(avg_train_loss_segmentation)
        train_losses_width.append(avg_train_loss_width)
        train_losses_depth.append(avg_train_loss_depth)

        train_losses.append(total_train_loss / len(train_loader))

        model.eval()
        total_val_loss, total_val_loss_segmentation, total_val_loss_width, total_val_loss_depth = 0, 0, 0, 0
        
        with torch.no_grad():
            for inputs, segmentation_targets, width_targets, depth_targets in val_loader:
                segmentation_pred, width_pred, depth_pred = model(inputs)  # Inputs are already properly shaped
                loss_segmentation = criterion_segmentation(segmentation_pred, segmentation_targets)
                loss_width = criterion_width(width_pred.squeeze(-1), width_targets)
                loss_depth = criterion_depth(depth_pred.squeeze(-1), depth_targets)

                total_loss = loss_segmentation + loss_width + loss_depth
                total_val_loss += total_loss.item()

                total_val_loss_segmentation += loss_segmentation.item()
                total_val_loss_width += loss_width.item()
                total_val_loss_depth += loss_depth.item()

        avg_val_loss_segmentation = total_val_loss_segmentation / len(val_loader)
        avg_val_loss_width = total_val_loss_width / len(val_loader)
        avg_val_loss_depth = total_val_loss_depth / len(val_loader)

        val_losses_segmentation.append(avg_val_loss_segmentation)
        val_losses_width.append(avg_val_loss_width)
        val_losses_depth.append(avg_val_loss_depth)

        print(f'Epoch {epoch+1}: Train Loss Seg: {avg_train_loss_segmentation:.4f}, Val Loss Seg: {avg_val_loss_segmentation:.4f}')
        print(f'Epoch {epoch+1}: Train Loss Width: {avg_train_loss_width:.4f}, Val Loss Width: {avg_val_loss_width:.4f}')
        print(f'Epoch {epoch+1}: Train Loss Depth: {avg_train_loss_depth:.4f}, Val Loss Depth: {avg_val_loss_depth:.4f}')
            
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f'Epoch {epoch+1}: Train Loss: {total_train_loss / len(train_loader):.4f}, Val Loss: {total_val_loss / len(val_loader):.4f}')

        # Save model checkpoint if it has the best validation loss so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'model/model_-bs{batch_size}_-lr{lr}_-epochs{epochs}.pth')
            torch.save(model, f'model/full_model_-bs{batch_size}_-lr{lr}_-epochs{epochs}.pth')
            print("Saved new best model.")
        
        print("#####################################################################")

        # Plotting
    save_folder = f"figs/bs{batch_size}_-lr{lr}_-epochs{epochs}"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plot_losses(train_losses_segmentation, val_losses_segmentation, "Segmentation Loss", f"{save_folder}/seg_loss_bs{batch_size}_-lr{lr}_-epochs{epochs}.png")
    plot_losses(train_losses_width, val_losses_width, "Width Loss", f"{save_folder}/width_loss_bs{batch_size}_-lr{lr}_-epochs{epochs}.png")
    plot_losses(train_losses_depth, val_losses_depth, "Depth Loss", f"{save_folder}/depth_loss_bs{batch_size}_-lr{lr}_-epochs{epochs}.png")
    plot_losses(train_losses, val_losses, "Total Loss", 15, f"{save_folder}/total_loss_bs{batch_size}_-lr{lr}_-epochs{epochs}.png")



def plot_losses(train_losses, val_losses, title, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)  # Save the plot to a file
    plt.close()  # Close the plot figure to free up memory


if __name__ == "__main__":
    # batch_size = [100, 500, 1000]
    # lr = [0.01, 0.001, 0.0001]
    # epochs = [100, 500, 1000, 5000]
    # for i in range(len(batch_size)):
    #     for j in range(len(lr)):
    #         for k in range(len(epochs)):
    #             main(batch_size[i], lr[j], epochs[k])
    
    batch_size = 100
    lr = 0.0001
    epochs = 1000
    main(batch_size, lr, epochs)

