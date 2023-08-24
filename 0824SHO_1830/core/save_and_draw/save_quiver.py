import matplotlib.pyplot as plt


def save_quiver(x_data, x_pred, data_type, save_path, scaler):
    file_suffix = f"{data_type}"
    actual_positions = x_data.detach().numpy()
    predicted_positions = x_pred.detach().numpy()

    # 벡터를 그리기 위한 시작점 및 방향 설정
    X, Y = actual_positions[:, 0], actual_positions[:, 1]
    U, V = predicted_positions[:, 0] - actual_positions[:, 0], predicted_positions[:, 1] - actual_positions[:, 1]

    # quiver 그래프로 표시
    # plt.scatter(x_data[:, 0].detach().numpy(), x_data[:, 1].detach().numpy(),color='blue',s=5, label='True data')
    plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=scaler, color='red', label='Residual Vectors')
    plt.xlabel('X component')
    plt.ylabel('Y component')
    plt.legend()
    plt.title('Residual Vectors')
    plt.grid()
    if file_suffix == 'Validation':
        plt.xlim(-1.4, -0.4)
        plt.ylim(-0.35, 0.65)
    elif file_suffix == 'Test':
        plt.xlim(-0.9, 0.1)
        plt.ylim(0.3, 1.3)
    plt.savefig(f"{save_path}/{file_suffix}_scaler{scaler}_quiver.png")
    plt.close()
