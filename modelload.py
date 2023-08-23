from pyDOE import lhs
import torch
import os
from torch import nn
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import pandas as pd
from sklearn.metrics import mean_absolute_error, max_error
import datetime
import copy

class ODEFunc(nn.Module):
        def __init__(self, class_num_layers, class_hidden_dim):
            super(ODEFunc, self).__init__()
            layers = []
            layers.append(nn.Linear(2, class_hidden_dim))
            for _ in range(class_num_layers-1):
                layers.append(nn.ReLU() if ReLU_On else nn.ELU())
                layers.append(nn.Linear(class_hidden_dim, class_hidden_dim))
            layers.append(nn.ReLU() if ReLU_On else nn.ELU())
            layers.append(nn.Linear(class_hidden_dim, 2))
            self.net = nn.Sequential(*layers)
            self.nfe = 0
    
        def forward(self, t, x):
            self.nfe += 1
            return self.net(x)
        
####################################################################################################################################################
#2023-08/23 23:00 민준 update 모델 불러오기
n_samples = 1000
num_layers = 3 # 실제 레이어의 수. 코드의 for문에서 -1을 이미 적용함
hidden_dim = 32
learning_rate = 0.005
epochs = 1000
batch_size= 70  # 배치 사이즈
model_try= 3   # 해당 값으로 트라이할 모델 수
min_epoch = 300 # model당 최소 epoch. 해당 값 이전까지는 stop하지않음
ReLU_On = False # True 적용시 레이어의 ReLU 활성화
cuda_On = False # cuDNN 설치 전까지 False 사용. 혹시 사용할 수도 있으니 다들 사전에 설치하면 좋겠음
patience = 1 # 이 epoch동안 val_loss 기록이 단 한 번도 개선되지 않으면 iteration을 종료
scaler = 0.1 # quiver scale 조정 값
amplification_factor = 20 # 증폭계수 적용
####################################################################################################################################################
func = ODEFunc(num_layers,hidden_dim)
model = func
model.load_state_dict(torch.load('best_model.pt')) # 불러올 모델. 해당 가중치로 그래프만 그려줍니다. results 폴더 바깥으로 best_model.pt 파일을 빼주세요
####################################################################################################################################################

print("Model Load")

def euclidean_distance(x1, x2):
    if x1.requires_grad:
        x1 = x1.detach()
    if x2.requires_grad:
        x2 = x2.detach()
    return torch.sqrt(torch.sum((x1 - x2) ** 2, dim=1))


def save_2d_and_actual_vs_predicted_amp(x_data, x_pred, data_type, hidden_dim, n_samples, epochs, save_path, amplification_factor):
    file_suffix = f"{data_type}"
    # Actual vs Predicted Plot
    error_x = (x_data[:, 0].detach().numpy() - x_pred[:, 0].detach().numpy())
    error_y = (x_data[:, 1].detach().numpy() - x_pred[:, 1].detach().numpy())
    total_error = np.sqrt(error_x**2 + error_y**2) * amplification_factor

    plt.plot(x_data[:, 0].detach().numpy(), x_pred[:, 0].detach().numpy() + total_error, linewidth=4, label='X component with Total Error')
    plt.plot(x_data[:, 1].detach().numpy(), x_pred[:, 1].detach().numpy() + total_error, linewidth=4, label='Y component with Total Error')
    plt.plot(torch.linspace(-1, 1, 100), torch.linspace(-1, 1, 100), color='red', linewidth=1.2, label='True = Predicted')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.title('Actual vs Predicted Plot with Total Error Amplification')
    plt.grid()
    plt.savefig(f"{save_path}/{file_suffix}_actual_vs_predict_amp.png")
    plt.close()

    
def save_2d_and_actual_vs_predicted(x_data, x_pred, data_type, hidden_dim, n_samples, epochs, save_path):
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
    distances = euclidean_distance(x_data, x_pred).numpy()

    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(distances)), distances, label='Euclidean Distance', alpha=0.7)
    plt.xlabel('Index')
    plt.ylabel('Euclidean Distance between Actual and Predicted')
    plt.title(f'Euclidean Distance over Data Points - {data_type}')
    plt.legend()
    plt.grid()
    plt.savefig(f"{save_path}/{file_suffix}_distance_plot.png")
    plt.close()
   
def save_quiver(x_data, x_pred, data_type, hidden_dim, n_samples, epochs, save_path,scaler):
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


def ensure_list(*values):
    max_length = max(len(value) if isinstance(value, list) else 1 for value in values)
    return [[value] * max_length if not isinstance(value, list) else value * (max_length // len(value)) for value in values]


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

def return_numerical_validation(x_data, x_pred, data_type, hidden_dim, n_samples, epochs, save_path):
    slope, intercept, r_value, _, _ = linregress(x_data.flatten().detach().numpy(), x_pred.flatten().detach().numpy())
    r_squared = r_value**2
    mean_abs_rel_residual = mean_absolute_error(x_data.detach().numpy(), x_pred.detach().numpy()) / (x_data.abs().mean())
    max_abs_rel_residual = max(np.max(np.abs(x_data.detach().numpy() - x_pred.detach().numpy()), axis=0) / x_data.abs().max())

    return r_squared, mean_abs_rel_residual, max_abs_rel_residual   

def numerical_validation(x_data, x_pred):
    slope, intercept, r_value, _, _ = linregress(x_data.flatten().detach().numpy(), x_pred.flatten().detach().numpy())
    r_squared = r_value**2
    mean_abs_rel_residual = mean_absolute_error(x_data.detach().numpy(), x_pred.detach().numpy()) / (x_data.abs().mean())
    max_abs_rel_residual = max(np.max(np.abs(x_data.detach().numpy() - x_pred.detach().numpy()), axis=0) / x_data.abs().max())
    
    return r_squared, mean_abs_rel_residual, max_abs_rel_residual

def save_best_model_draw(save_path, x_pred_test_best, x_pred_val_best, x_pred_train_best):
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

# Latin Hypercube Design (LHD) sample generation
lhd_samples = lhs(2, samples=n_samples)
t_lhd = torch.tensor(np.sort(lhd_samples[:, 0]) * 2 * np.pi, dtype=torch.float32)
x_lhd = torch.cat((torch.sin(t_lhd).reshape(-1, 1), torch.cos(t_lhd).reshape(-1, 1)), dim=1)

# Data splitting
train_size = int(0.7 * len(x_lhd))
val_size = int(0.15 * len(x_lhd))
x_train = x_lhd[:train_size]
t_train = t_lhd[:train_size]
x_val = x_lhd[train_size:train_size + val_size]
t_val = t_lhd[train_size:train_size + val_size]
x_test = x_lhd[train_size + val_size:]
t_test = t_lhd[train_size + val_size:]


x_pred_train_best = odeint(model, x_train[0], t_train).squeeze() # odeint to be defined

x_pred_val_best = odeint(model, x_val[0], t_val).squeeze() # odeint to be defined

x_pred_test_best = odeint(model, x_test[0], t_test).squeeze() # odeint to be defined

r_squared_train, mean_abs_rel_residual_train, max_abs_rel_residual_train = numerical_validation(x_train, x_pred_train_best)
r_squared_val, mean_abs_rel_residual_val, max_abs_rel_residual_val = numerical_validation(x_val, x_pred_val_best)
r_squared_test, mean_abs_rel_residual_test, max_abs_rel_residual_test = numerical_validation(x_test, x_pred_test_best)

validation_results = pd.DataFrame(columns=['Samples', 'Hidden_Dim', 'Learning_Rate', 'Epochs', 'Data_Type', 'R_Squared', 'Mean_Abs_Rel_Residual', 'Max_Abs_Rel_Residual'])
results_df_train = pd.DataFrame(columns=['Samples', 'Hidden_Dim', 'Learning_Rate', 'Epochs', 'Data_Type', 'R_Squared', 'Mean_Abs_Rel_Residual', 'Max_Abs_Rel_Residual'])
results_df_test = pd.DataFrame(columns=['Samples', 'Hidden_Dim', 'Learning_Rate', 'Epochs', 'Data_Type', 'R_Squared', 'Mean_Abs_Rel_Residual', 'Max_Abs_Rel_Residual'])
results_df_train.loc[0] = [n_samples, hidden_dim, learning_rate, epochs, 'Train', r_squared_train, mean_abs_rel_residual_train, max_abs_rel_residual_train]
validation_results.loc[0] = [n_samples, hidden_dim, learning_rate, epochs, 'Validation', r_squared_val, mean_abs_rel_residual_val, max_abs_rel_residual_val]
results_df_test.loc[0] = [n_samples, hidden_dim, learning_rate, epochs, 'Test', r_squared_test, mean_abs_rel_residual_test, max_abs_rel_residual_test]

n_samples_list, hidden_dim_list, learning_rate_list, epochs_list = ensure_list(n_samples, hidden_dim,learning_rate, epochs)
save_path_csv = "results"
if not os.path.exists(save_path_csv):
        os.makedirs(save_path_csv)

timestamp = datetime.datetime.now().strftime("%m-%d_%H%M")

for n_samples, hidden_dim, learning_rate, epochs in zip(n_samples_list, hidden_dim_list, learning_rate_list, epochs_list):
    # 결과 저장 경로
    save_path = f"{save_path_csv}/{timestamp}_sam{n_samples}_layer{num_layers}_dim{hidden_dim}_lr{learning_rate}_epoch{epochs}_batch{batch_size}_minepoch{min_epoch}_patience{patience}"
    # 디렉토리가 없으면 생성
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # if not os.path.exists(save_path2):
    #     os.makedirs(save_path2)

    save_2d_and_actual_vs_predicted_amp(x_train, x_pred_train_best, 'Train', hidden_dim, n_samples, epochs, save_path, amplification_factor)
    save_2d_and_actual_vs_predicted_amp(x_val, x_pred_val_best, "Validation", hidden_dim, n_samples, epochs, save_path, amplification_factor)
    save_2d_and_actual_vs_predicted_amp(x_test, x_pred_test_best, 'Test', hidden_dim, n_samples, epochs, save_path, amplification_factor)
    # train
    plot_radius_deviation_histogram(x_train, x_pred_train_best, 'Train', hidden_dim, n_samples, epochs, save_path)
    save_2d_and_actual_vs_predicted(x_train, x_pred_train_best, 'Train', hidden_dim, n_samples, epochs, save_path)
    save_quiver(x_train, x_pred_train_best, 'Train', hidden_dim, n_samples, epochs, save_path,scaler)
    # val
    plot_radius_deviation_histogram(x_val, x_pred_val_best, "Validation", hidden_dim, n_samples, epochs, save_path)
    save_2d_and_actual_vs_predicted(x_val, x_pred_val_best, "Validation", hidden_dim, n_samples, epochs, save_path)
    save_quiver(x_val, x_pred_val_best, "Validation", hidden_dim, n_samples, epochs, save_path,scaler)
    # test
    plot_radius_deviation_histogram(x_test, x_pred_test_best, 'Test', hidden_dim, n_samples, epochs, save_path)
    save_2d_and_actual_vs_predicted(x_test, x_pred_test_best, 'Test', hidden_dim, n_samples, epochs, save_path)
    save_quiver(x_test, x_pred_test_best, 'Test', hidden_dim, n_samples, epochs, save_path,scaler)
    # 최종 그래프
    save_best_model_draw(save_path, x_pred_test_best, x_pred_val_best, x_pred_train_best)



results_df_train.to_csv(f"{save_path}/Train_results_{hidden_dim}_{n_samples}.csv", index=False)
validation_results.to_csv(f"{save_path}/validation_results_{hidden_dim}_{n_samples}.csv", index=False)
results_df_test.to_csv(f"{save_path}/Test_results_{hidden_dim}_{n_samples}.csv", index=False)