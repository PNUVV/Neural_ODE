import torch
import matplotlib.pyplot as plt
import os
def plot_radius_deviation_histogram(x_data, x_pred, data_type, hidden_dim, n_samples, epochs, save_path):
    # Calculate the radius for actual and predicted data
    actual_radius = torch.sqrt(x_data[:, 0]**2 + x_data[:, 1]**2)
    pred_radius = torch.sqrt(x_pred[:, 0]**2 + x_pred[:, 1]**2)

    # Calculate the difference in radius (deviation)
    radius_deviation = actual_radius - pred_radius

    # Plotting the histogram of the radius deviation
    plt.hist(radius_deviation.detach().numpy(), bins=20, color="blue", edgecolor="black", alpha=0.7)
    plt.title(f'Radius Deviation Histogram\nHidden Dim: {hidden_dim}, Samples: {n_samples}, Epochs: {epochs}')
    plt.xlabel('Radius Deviation')
    plt.ylabel('Frequency')

    # Calculate and display mean and standard deviation
    mean_deviation = radius_deviation.mean().item()
    std_deviation = radius_deviation.std().item()
    plt.axvline(mean_deviation, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_deviation:.2f}\nStd Dev: {std_deviation:.2f}')
    plt.legend()

    # Saving the plot
    plot_path = os.path.join(save_path, f'{data_type}_radius_deviation_histogram.png')
    plt.savefig(plot_path)
    plt.close() # Close the plot to avoid displaying it in the notebook
