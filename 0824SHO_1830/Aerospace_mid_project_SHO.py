from pyDOE import lhs
import torch
import os
from torch import nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import numpy as np
import copy
from core import ensure_list
from core.valid import numerical_validation
from core.save_and_draw import save_best_model_draw,save_2d_and_actual_vs_predicted, save_quiver, plot_radius_deviation_histogram

############################################################################################################################
#2023-08/23 21:27 민기 update
n_samples = 1000
num_layers = 3 # 실제 레이어의 수. 코드의 for문에서 -1을 이미 적용함
hidden_dim = 32
learning_rate = 0.005
epochs = 100
batch_size= 70  # 배치 사이즈
model_try= 3   # 해당 값으로 트라이할 모델 수
min_epoch = 30 # model당 최소 epoch. 해당 값 이전까지는 stop하지않음
ReLU_On = False # True 적용시 레이어의 ReLU 활성화
cuda_On = False # cuDNN 설치 전까지 False 사용. 혹시 사용할 수도 있으니 다들 사전에 설치하면 좋겠음
patience = 10 # 이 epoch동안 val_loss 기록이 단 한 번도 개선되지 않으면 iteration을 종료
scaler = 1 # quiver scale 조정 값
amplification_factor = 5 # 증폭계수 적용
############################################################################################################################


def get_true_x():
    class ODEFunc_true(nn.Module):
        omega = torch.tensor(1.0).to(device)  # You can adjust this value as needed

        def __init__(self):
            super(ODEFunc_true, self).__init__()
            self.nfe = 0

        def forward(self, t, y):
            self.nfe += 1
            x, v = y[..., 0], y[..., 1]
            dxdt = v
            dvdt = - ODEFunc_true.omega ** 2 * x
            return torch.stack([dxdt, dvdt], -1)

    func = ODEFunc_true()

    return func


def train_ode_model(hidden_dim, num_layers, learning_rate, epochs):
    class ODEFunc(nn.Module):
        def __init__(self, class_num_layers, class_hidden_dim):
            super(ODEFunc, self).__init__()
            layers = []
            layers.append(nn.Linear(2, class_hidden_dim))
            for _ in range(class_num_layers - 1):
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
        func.train()
        optimizer.zero_grad()
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
    print("Epoch: {:3} | Loss: {:.9f} | Val Loss: {:.9f} | Training finished. Elapsed Time: {:} ".format(epoch, loss.item(), val_loss.item(),elapsed_time))
    
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
        r_squared_train, mean_abs_rel_residual_train, max_abs_rel_residual_train = \
            numerical_validation.numerical_validation(x_train, x_pred_train)
        x_pred_val = odeint(func, x_val[0], t_val).squeeze() # odeint to be defined
        r_squared_val, mean_abs_rel_residual_val, max_abs_rel_residual_val = \
            numerical_validation.numerical_validation(x_val, x_pred_val)
        x_pred_test = odeint(func, x_test[0], t_test).squeeze() # odeint to be defined
        r_squared_test, mean_abs_rel_residual_test, max_abs_rel_residual_test = \
            numerical_validation.numerical_validation(x_test, x_pred_test)

        # Save validation results
        results_df_train.loc[idx] = \
            [n_samples, hidden_dim, learning_rate, epochs, 'Train', r_squared_train, mean_abs_rel_residual_train, max_abs_rel_residual_train]
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


#쿠다 설정 cpu가 디폴트
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda_On else 'cpu')
print("you can use cuda" if use_cuda else "you can't use cuda")
print(f'Using device: {device}')


# Latin Hypercube Design (LHD) sample generation
lhd_samples = lhs(2, samples=n_samples)
t_lhd = torch.tensor(np.sort(lhd_samples[:, 0]) * 2 * np.pi, dtype=torch.float32)

tensor1 = torch.tensor([0])
tensor2 = torch.tensor([0])

# 텐서 합치기
x_initial = torch.cat((tensor1, tensor2), dim=0)
x_initial = x_initial.float()


func_get = get_true_x()
x_lhd = odeint(func_get, x_initial, t_lhd).squeeze()


# Data splitting
train_size = int(0.7 * len(x_lhd))
val_size = int(0.15 * len(x_lhd))
x_train = x_lhd[:train_size]
t_train = t_lhd[:train_size]
x_val = x_lhd[train_size:train_size + val_size]
t_val = t_lhd[train_size:train_size + val_size]
x_test = x_lhd[train_size + val_size:]
t_test = t_lhd[train_size + val_size:]


n_samples_list, hidden_dim_list, learning_rate_list, epochs_list = ensure_list.ensure_list(n_samples, hidden_dim,learning_rate, epochs)
save_path_csv = "./results"
if not os.path.exists(save_path_csv):
        os.makedirs(save_path_csv)
save_path_csv2 = "./csv"
if not os.path.exists(save_path_csv2):
        os.makedirs(save_path_csv2)

# with torch.no_grad():
#             for name, param in best_func.named_parameters(): 
#                 print(f"{name}: {param.data}")# 레이어의 weight와 bias 출력

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
    print(f"best model idx: {max_r_squared_model} | samples: {n_samples} | hidden_dim: {hidden_dim} | lr={learning_rate} | epochs: {epochs}")
    
    # save_2d_and_actual_vs_predicted_amp(x_train, x_pred_train_best, 'Train', hidden_dim, n_samples, epochs, save_path, amplification_factor)
    # train
    plot_radius_deviation_histogram.plot_radius_deviation_histogram(x_train, x_pred_train_best, 'Train', hidden_dim, n_samples, epochs, save_path)
    save_2d_and_actual_vs_predicted.save_2d_and_actual_vs_predicted(x_train, x_pred_train_best, 'Train', save_path)
    save_quiver.save_quiver(x_train, x_pred_train_best, 'Train', save_path, scaler)
    # val
    plot_radius_deviation_histogram.plot_radius_deviation_histogram(x_val, x_pred_val_best, "Validation", hidden_dim, n_samples, epochs, save_path)
    save_2d_and_actual_vs_predicted.save_2d_and_actual_vs_predicted(x_val, x_pred_val_best, "Validation", save_path)
    save_quiver.save_quiver(x_val, x_pred_val_best, "Validation", save_path, scaler)
    # test
    plot_radius_deviation_histogram.plot_radius_deviation_histogram(x_test, x_pred_test_best, 'Test', hidden_dim, n_samples, epochs, save_path)
    save_2d_and_actual_vs_predicted.save_2d_and_actual_vs_predicted(x_test, x_pred_test_best, 'Test', save_path)
    save_quiver.save_quiver(x_test, x_pred_test_best, 'Test', save_path, scaler)
    # 최종 그래프
    save_best_model_draw.save_best_model_draw(save_path, x_lhd, x_pred_test_best, x_pred_val_best, x_pred_train_best)
