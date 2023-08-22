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

############################################################################################################################
#2023-08/22 13:30 민준 update
n_samples = 1000
num_layers = 3 # 실제 레이어의 수. 코드의 for문에서 -1을 이미 적용함
hidden_dim = 32
learning_rate = 0.005
epochs = 3000
batch_size= 70  # 배치 사이즈
model_try= 5   # 해당 값으로 트라이할 모델 수
min_epoch = 1000 # model당 최소 epoch. 해당 값 이전까지는 stop하지않음
ReLU_On = False # True 적용시 레이어의 ReLU 활성화
cuda_On = False # cuDNN 설치 전까지 False 사용. 혹시 사용할 수도 있으니 다들 사전에 설치하면 좋겠음
patience = 1 #이 epoch동안 val_loss 기록이 단 한 번도 개선되지 않으면 iteration을 종료
############################################################################################################################
# Ex)
# n_samples = 1000
# num_layers = 3 # 실제 레이어의 수. 코드의 for문에서 -1을 이미 적용함
# hidden_dim = 32
# learning_rate = 0.005
# epochs = 110
# batch_size= 70  # 배치 사이즈
# model_try= 5   # 해당 값으로 트라이할 모델 수
# min_epoch = 80 # model당 최소 epoch. 해당 값 이전까지는 stop하지않음
# ReLU_On = False # True 적용시 레이어의 ReLU 활성화
# cuda_On = False # cuDNN 설치 전까지 False 사용. 혹시 사용할 수도 있으니 다들 사전에 설치하면 좋겠음
# patience = 4 #이 epoch동안 val_loss 기록이 단 한 번도 개선되지 않으면 iteration을 종료
############################################################################################################################

#쿠다 설정 cpu가 디폴트
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda_On else 'cpu')
print("you can use cuda" if use_cuda else "you can't use cuda")
print(f'Using device: {device}')


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


def train_ode_model(hidden_dim,num_layers, learning_rate, epochs):
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
    

    start_time = datetime.datetime.now()

    # Neural ODE definition
    func = ODEFunc(num_layers,hidden_dim)
   
    optimizer = torch.optim.Adam(func.parameters(), lr=learning_rate)
    # optimizer = torch.optim.AdamW(func.parameters(), lr=0.004, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.03, amsgrad=False)

    criterion = nn.MSELoss()
    best_loss = float(5) # 임의 정의

    losses = []
    val_losses=[]
    for epoch in range(epochs):
        optimizer.zero_grad()
        func.train()
        for batch_start in range(0, train_size, batch_size):
            batch_x_train = x_train[batch_start:batch_start+batch_size]
            batch_t_train = t_train[batch_start:batch_start+batch_size]
            batch_x_pred_train = odeint(func, batch_x_train[0], batch_t_train).squeeze()
            loss = criterion(batch_x_pred_train, batch_x_train)
            loss.backward()
        optimizer.step()
        
        func.eval() # 모델을 평가 모드로 설정
        with torch.no_grad():
            x_pred_val = odeint(func, x_val[0], t_val).squeeze()
            val_loss = criterion(x_pred_val, x_val)
            val_losses.append(val_loss.item())
        losses.append(loss.item())
    
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
        else:
            counter += 1

        if epoch>min_epoch:
            if counter >= patience:
                print("Early stopping at epoch {:3} | Loss: {:.9f} | Val Loss: {:.9f}".format(epoch, loss.item(), val_loss.item()))
                break
    
        if epoch % 10 == 0:
            print("Epoch: {:3} | Loss: {:.9f} | Val Loss: {:.9f}".format(epoch, loss.item(), val_loss.item()))
    
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print(f'Training finished. Elapsed Time: {elapsed_time}')
    
    return func, losses, val_losses
    

def train_ode_models(n_samples, hidden_dim,num_layers, learning_rate, epochs, save_path):

    best_func = None
    best_r_squared = -float('inf') # Initialize with negative infinity
    validation_results = pd.DataFrame(columns=['Samples', 'Hidden_Dim', 'Learning_Rate', 'Epochs', 'Data_Type', 'R_Squared', 'Mean_Abs_Rel_Residual', 'Max_Abs_Rel_Residual'])
    results_df_train = pd.DataFrame(columns=['Samples', 'Hidden_Dim', 'Learning_Rate', 'Epochs', 'Data_Type', 'R_Squared', 'Mean_Abs_Rel_Residual', 'Max_Abs_Rel_Residual'])
    results_df_test = pd.DataFrame(columns=['Samples', 'Hidden_Dim', 'Learning_Rate', 'Epochs', 'Data_Type', 'R_Squared', 'Mean_Abs_Rel_Residual', 'Max_Abs_Rel_Residual'])
    
    # Model selection and result saving logic
    for idx in range(model_try):
        print(f'Model idx: {idx}')
        
        func, losses, val_losses = train_ode_model(hidden_dim,num_layers, learning_rate, epochs) # Function to be defined

        x_pred_train = odeint(func, x_train[0], t_train).squeeze() # odeint to be defined
        r_squared_train, mean_abs_rel_residual_train, max_abs_rel_residual_train = numerical_validation(x_train, x_pred_train)
        x_pred_val = odeint(func, x_val[0], t_val).squeeze() # odeint to be defined
        r_squared_val, mean_abs_rel_residual_val, max_abs_rel_residual_val = numerical_validation(x_val, x_pred_val)
        x_pred_test = odeint(func, x_test[0], t_test).squeeze() # odeint to be defined
        r_squared_test, mean_abs_rel_residual_test, max_abs_rel_residual_test = numerical_validation(x_test, x_pred_test)

        # Save validation results
        results_df_train.loc[idx] = [n_samples, hidden_dim, learning_rate, epochs, 'Train', r_squared_train, mean_abs_rel_residual_train, max_abs_rel_residual_train]
        validation_results.loc[idx] = [n_samples, hidden_dim, learning_rate, epochs, 'Validation', r_squared_val, mean_abs_rel_residual_val, max_abs_rel_residual_val]
        results_df_test.loc[idx] = [n_samples, hidden_dim, learning_rate, epochs, 'Test', r_squared_test, mean_abs_rel_residual_test, max_abs_rel_residual_test]
        
        # Select best model
        if r_squared_val > best_r_squared:
            best_r_squared = r_squared_val
            best_func = copy.deepcopy(func)
            x_pred_test_best = x_pred_test
            x_pred_val_best = x_pred_val
            x_pred_train_best = x_pred_train
    
        plt.plot(losses,label='Train Losses')
        plt.legend()
        plt.title('Train Loss (MSE)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(f"{save_path}/Train_loss_idx_{idx}.png")
        plt.close()
        
        plt.plot(val_losses,label='Validation Losses')
        plt.legend()
        plt.title('Validation Loss (MSE)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(f"{save_path}/Validation_loss_idx_{idx}.png")
        plt.close()
        
        plt.plot(losses,label='Train Losses')
        plt.plot(val_losses,label='Validation Losses')
        plt.legend()
        plt.title('Train Loss vs Validation Loss (MSE)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.savefig(f"{save_path}/Train vs Validation_loss_idx_{idx}.png")
        plt.close()
    
    # 08/21 01:10 나머지 그래프도 idx따라 다 뽑아야할지 고민. 현재는 loss를 제외하고는 best만 추출 
    model_save = os.path.join(save_path,'best_model.pt')
    torch.save(best_func.state_dict(), model_save)  # Save the best model
    # Save CSV
    results_df_train.to_csv(f"{save_path}/Train_results_{hidden_dim}_{n_samples}.csv", index=False)
    validation_results.to_csv(f"{save_path}/validation_results_{hidden_dim}_{n_samples}.csv", index=False)
    results_df_test.to_csv(f"{save_path}/Test_results_{hidden_dim}_{n_samples}.csv", index=False)
    max_r_squared_model = validation_results['R_Squared'].idxmax()

    return best_func, max_r_squared_model,x_pred_train_best,x_pred_val_best,x_pred_test_best


def save_2d_and_actual_vs_predicted_train_amp(x_data, x_pred, data_type, hidden_dim, n_samples, epochs, save_path, amplification_factor):
    file_suffix = f"{data_type}"

    # Calculate Total Error
    error_x = (x_data[:, 0].detach().numpy() - x_pred[:, 0].detach().numpy())
    error_y = (x_data[:, 1].detach().numpy() - x_pred[:, 1].detach().numpy())
    total_error = np.sqrt(error_x**2 + error_y**2)

    # Actual vs Predicted Plot using Total Error
    plt.plot(total_error, total_error, linewidth=4, label='Total Error')
    plt.plot(total_error, total_error, color='red', linewidth=1.2, label='True = Predicted') # Ideal line
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.title('Actual vs Predicted Plot for Total Error')
    plt.grid()
    plt.savefig(f"{save_path}/{file_suffix}_actual_vs_predict_amp.png")
    plt.close()



def save_2d_and_actual_vs_predicted_train(x_data, x_pred, data_type, hidden_dim, n_samples, epochs, save_path):
    file_suffix = f"{data_type}"
    # 2D Motion Plot
    plt.plot(x_data[:, 0].detach().numpy(), x_data[:, 1].detach().numpy(), linewidth= 5,label='True trajectory')
    plt.plot(x_pred[:, 0].detach().numpy(), x_pred[:, 1].detach().numpy(), label='Neural ODE approximation')
    plt.legend()
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('2D Motion')
    plt.savefig(f"{save_path}/{file_suffix}_2D_motion.png")
    plt.close()

    # Actual vs Predicted Plot
    plt.plot(x_data[:, 0].detach().numpy(), x_pred[:, 0].detach().numpy(),linewidth=5, label='X component')
    plt.plot(x_data[:, 1].detach().numpy(), x_pred[:, 1].detach().numpy(), linewidth=5, label='Y component') 
    plt.plot(torch.linspace(-1, 1, 100), torch.linspace(-1, 1, 100), color='red', linewidth=1.2,label='True = Predicted')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.title('Actual vs Predicted Plot')
    plt.grid()
    plt.savefig(f"{save_path}/{file_suffix}_actual_vs_predict.png")
    plt.close()
    

def save_2d_and_actual_vs_predicted_val(x_data, x_pred, data_type, hidden_dim, n_samples, epochs, save_path):
    file_suffix = f"{data_type}"
    # 2D Motion Plot
    plt.plot(x_data[:, 0].detach().numpy(), x_data[:, 1].detach().numpy(),linewidth=5, label='True trajectory')
    plt.plot(x_pred[:, 0].detach().numpy(), x_pred[:, 1].detach().numpy(), label='Validation trajectory')
    plt.legend()
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.xlim(-1.4, -0.4)
    plt.ylim(-0.35, 0.65)
    plt.title('2D Motion')
    plt.savefig(f"{save_path}/{file_suffix}_2D_motion.png")
    plt.close()

    # Actual vs Predicted Plot
    plt.plot(x_data[:, 0].detach().numpy(), x_pred[:, 0].detach().numpy(),linewidth=5, label='X component')
    plt.plot(x_data[:, 1].detach().numpy(), x_pred[:, 1].detach().numpy(), linewidth=5, label='Y component')
    plt.plot(torch.linspace(-1, 1, 100), torch.linspace(-1, 1, 100), color='red', linewidth=1.2,label='True = Predicted') 
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.title('Actual vs Predicted Plot')
    plt.grid()
    plt.savefig(f"{save_path}/{file_suffix}_actual_vs_predict.png")
    plt.close()
    

def save_2d_and_actual_vs_predicted_test(x_data, x_pred, data_type, hidden_dim, n_samples, epochs, save_path):
    file_suffix = f"{data_type}"
    # 2D Motion Plot
    plt.plot(x_data[:, 0].detach().numpy(), x_data[:, 1].detach().numpy(),linewidth=5, label='True trajectory')
    plt.plot(x_pred[:, 0].detach().numpy(), x_pred[:, 1].detach().numpy(), label='Test trajectory')
    plt.legend()
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.xlim(-0.9, 0.1)
    plt.ylim(0.3, 1.3)
    plt.title('2D Motion')
    plt.savefig(f"{save_path}/{file_suffix}_2D_motion.png")
    plt.close()

    # Actual vs Predicted Plot
    plt.plot(x_data[:, 0].detach().numpy(), x_pred[:, 0].detach().numpy(),linewidth=5, label='X component')
    plt.plot(x_data[:, 1].detach().numpy(), x_pred[:, 1].detach().numpy(), linewidth=5, label='Y component') 
    plt.plot(torch.linspace(-1, 1, 100), torch.linspace(-1, 1, 100), color='red', linewidth=1.2,label='True = Predicted')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.title('Actual vs Predicted Plot')
    plt.grid()
    plt.savefig(f"{save_path}/{file_suffix}_actual_vs_predict.png")
    plt.close()
   
def save_quiver(x_data, x_pred, data_type, hidden_dim, n_samples, epochs, save_path):
    file_suffix = f"{data_type}"
    # X, Y 위치를 벡터로 나타냄
    actual_positions = x_data.detach().numpy()
    predicted_positions = x_pred.detach().numpy()

    # 벡터를 그리기 위한 시작점 및 방향 설정
    X, Y = actual_positions[:, 0], actual_positions[:, 1]
    U, V = predicted_positions[:, 0] - actual_positions[:, 0], predicted_positions[:, 1] - actual_positions[:, 1]

    # quiver 그래프로 표시
    plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=100000, color='red', label='Residual Vectors')
    plt.xlabel('X component')
    plt.ylabel('Y component')
    plt.legend()
    plt.title('Residual Vectors')
    plt.grid()
    plt.savefig(f"{save_path}/{file_suffix}_quiver.png")
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
    plt.plot(x_lhd[:, 0].detach().numpy(), x_lhd[:, 1].detach().numpy(),linewidth= 4, label='True trajectory')
    plt.plot(x_pred_train_best[:, 0].detach().numpy(), x_pred_train_best[:, 1].detach().numpy(), label='Neural ODE approximation')
    plt.plot(x_pred_val_best[:, 0].detach().numpy(), x_pred_val_best[:, 1].detach().numpy(), label='Predicted validation trajectory')
    plt.plot(x_pred_test_best[:, 0].detach().numpy(), x_pred_test_best[:, 1].detach().numpy(), label='Predicted test trajectory')
    plt.legend()
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Best Model 2D Motion Data Prediction')
    plt.savefig(f"{save_path}/best_model_draw.png")
    plt.close()

n_samples_list, hidden_dim_list, learning_rate_list, epochs_list = ensure_list(n_samples, hidden_dim,learning_rate, epochs)
save_path_csv = "results"
if not os.path.exists(save_path_csv):
        os.makedirs(save_path_csv)
save_path_csv2 = "csv"
if not os.path.exists(save_path_csv2):
        os.makedirs(save_path_csv2)

# with torch.no_grad():
#             for name, param in best_func.named_parameters(): 
#                 print(f"{name}: {param.data}")# 레이어의 weight와 bias 출력

# 결과를 저장할 DataFrame 생성

timestamp = datetime.datetime.now().strftime("%m-%d_%H%M")

for n_samples, hidden_dim, learning_rate, epochs in zip(n_samples_list, hidden_dim_list, learning_rate_list, epochs_list):
    # 결과 저장 경로
    save_path = f"{save_path_csv}/{timestamp}_sam{n_samples}_layer{num_layers}_dim{hidden_dim}_lr{learning_rate}_epoch{epochs}_batch{batch_size}_minepoch{min_epoch}_patience{patience}"
    save_path2 = f"{save_path_csv}/{save_path_csv2}/{timestamp}_Numerical_sam{n_samples}_layer{num_layers}_dim{hidden_dim}_lr{learning_rate}_epoch{epochs}_batch{batch_size}_minepoch{min_epoch}_patience_{patience}"
    # 디렉토리가 없으면 생성
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # if not os.path.exists(save_path2):
    #     os.makedirs(save_path2)

    # 최적의 모델 훈련
    best_func, max_r_squared_model,x_pred_train_best,x_pred_val_best,x_pred_test_best = train_ode_models(n_samples, hidden_dim,num_layers, learning_rate, epochs, save_path)
    print(f"Training with n_samples={n_samples}, hidden_dim={hidden_dim}, lr={learning_rate}, epochs={epochs},best model idx={max_r_squared_model}")
    

    # save_2d_and_actual_vs_predicted_train_amp(x_train, x_pred_train_best, 'Train', hidden_dim, n_samples, epochs, save_path, 5000)
    # train
    plot_radius_deviation_histogram(x_train, x_pred_train_best, 'Train', hidden_dim, n_samples, epochs, save_path)
    save_2d_and_actual_vs_predicted_train(x_train, x_pred_train_best, 'Train', hidden_dim, n_samples, epochs, save_path)
    save_quiver(x_train, x_pred_train_best, 'Train', hidden_dim, n_samples, epochs, save_path)
    # val
    plot_radius_deviation_histogram(x_val, x_pred_val_best, "Validation", hidden_dim, n_samples, epochs, save_path)
    save_2d_and_actual_vs_predicted_val(x_val, x_pred_val_best, "Validation", hidden_dim, n_samples, epochs, save_path)
    save_quiver(x_val, x_pred_val_best, "Validation", hidden_dim, n_samples, epochs, save_path)
    # test
    plot_radius_deviation_histogram(x_test, x_pred_test_best, 'Test', hidden_dim, n_samples, epochs, save_path)
    save_2d_and_actual_vs_predicted_test(x_test, x_pred_test_best, 'Test', hidden_dim, n_samples, epochs, save_path)
    save_quiver(x_test, x_pred_test_best, 'Test', hidden_dim, n_samples, epochs, save_path)
    # 최종 그래프
    save_best_model_draw(save_path, x_pred_test_best, x_pred_val_best, x_pred_train_best)
