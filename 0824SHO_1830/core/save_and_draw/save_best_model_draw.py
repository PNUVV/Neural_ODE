import matplotlib.pyplot as plt

def save_best_model_draw(save_path,x_lhd, x_pred_test_best, x_pred_val_best, x_pred_train_best):
    # 2D Motion Plot
    plt.plot(x_lhd[:, 0].detach().numpy(), x_lhd[:, 1].detach().numpy(),color='red', label='True trajectory')
    plt.scatter(x_pred_train_best[:, 0].detach().numpy(), x_pred_train_best[:, 1].detach().numpy(), label='Neural ODE approximation')
    plt.scatter(x_pred_val_best[:, 0].detach().numpy(), x_pred_val_best[:, 1].detach().numpy(), label='Predicted validation trajectory')
    plt.scatter(x_pred_test_best[:, 0].detach().numpy(), x_pred_test_best[:, 1].detach().numpy(), label='Predicted test trajectory')
    plt.legend()
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Best Model 2D Motion Data Prediction')
    plt.savefig(f"{save_path}/best_model_draw.png")
    plt.close()