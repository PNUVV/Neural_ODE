import torch
import matplotlib.pyplot as plt

def euclidean_distance(x1, x2):
    if x1.requires_grad:
        x1 = x1
    if x2.requires_grad:
        x2 = x2
    return torch.sqrt(torch.sum((x1 - x2) ** 2, dim=1))



def save_2d_and_actual_vs_predicted(x_data, x_pred, data_type, save_path):
    file_suffix = f"{data_type}"
    # 2D Motion Plot
    plt.plot(x_data[:, 0].detach().numpy(), x_data[:, 1].detach().numpy(),color='red', label='True trajectory')
    plt.scatter(x_pred[:, 0].detach().numpy(), x_pred[:, 1].detach().numpy(), label='Neural ODE approximation')
    plt.legend()
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('2D Motion')
    if file_suffix == 'Validation':
        plt.xlim(-1.4, -0.4)
        plt.ylim(-0.35, 0.65)
    elif file_suffix == 'Test':
        plt.xlim(-0.9, 0.1)
        plt.ylim(0.3, 1.3)
    plt.savefig(f"{save_path}/{file_suffix}_2D_motion.png")
    plt.close()

    # Actual vs Predicted Plot
    plt.scatter(x_data[:, 0].detach().numpy(), x_pred[:, 0].detach().numpy(), label='X component')
    plt.plot(torch.linspace(-1.1, 1.1, 100), torch.linspace(-1.1, 1.1, 100), color='red', linewidth=1.2,label='True = Predicted')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.title('Actual vs Predicted Plot')
    plt.grid()
    if file_suffix == 'Validation':
        plt.xlim(-1.1, -0.75)
        plt.ylim(-1.1, -0.75)
    elif file_suffix == 'Test':
        plt.xlim(-0.9, 0.1)
        plt.ylim(-0.9, 0.1)
    plt.savefig(f"{save_path}/{file_suffix}_actual_vs_predict_X.png")
    plt.close()

    plt.scatter(x_data[:, 1].detach().numpy(), x_pred[:, 1].detach().numpy(), label='Y component')
    plt.plot(torch.linspace(-1.1, 1.1, 100), torch.linspace(-1.1, 1.1, 100), color='red', linewidth=1.2,label='True = Predicted')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.title('Actual vs Predicted Plot')
    plt.grid()
    if file_suffix == 'Validation':
        plt.xlim(-0.4, 0.75)
        plt.ylim(-0.4, 0.75)
    elif file_suffix == 'Test':
        plt.xlim(0.5, 1.1)
        plt.ylim(0.5, 1.1)
    plt.savefig(f"{save_path}/{file_suffix}_actual_vs_predict_Y.png")
    plt.close()

    # 3. Actual vs Predicted euclidean_distance Distance Plot
    distances = euclidean_distance(x_data, x_pred).detach().numpy()

    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(distances)), distances, label='Euclidean Distance', alpha=0.7)
    plt.xlabel('Index')
    plt.ylabel('Euclidean Distance between Actual and Predicted')
    plt.title(f'Euclidean Distance over Data Points - {data_type}')
    plt.legend()
    plt.grid()
    plt.savefig(f"{save_path}/{file_suffix}_distance_plot.png")
    plt.close()