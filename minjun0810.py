from pyDOE import lhs
import torch
from torch import nn
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.metrics import mean_absolute_error, max_error
from prettytable import PrettyTable

class ODEF(nn.Module):
    def forward(self, t, x):
        return self.net(x)

    def __init__(self, in_dim, hidden_dim, out_dim):
        super(ODEF, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim) 
        )

# Neural ODE definition
func = ODEF(2, 16, 2)

#최적화
# optimizer = torch.optim.Adam(func.parameters(), lr=0.0035)
optimizer = torch.optim.AdamW(func.parameters(), lr=0.003, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.03, amsgrad=False)

n_samples = 2048
lhd_samples = lhs(2, samples=n_samples)

# t_lhd = torch.linspace(0, 2*np.pi, 1000) #균일 시계열
t_lhd = torch.tensor(np.sort(lhd_samples[:, 0]) * 2 * np.pi, dtype=torch.float32)
x_lhd = torch.cat((torch.sin(t_lhd).reshape(-1, 1), torch.cos(t_lhd).reshape(-1, 1)), dim=1)
train_size = int(0.7 * len(x_lhd))
val_size = int(0.15 * len(x_lhd))
x_train, x_val, x_test = x_lhd[:train_size], x_lhd[train_size:train_size+val_size], x_lhd[train_size+val_size:]
t_train, t_val, t_test = t_lhd[:train_size], t_lhd[train_size:train_size+val_size], t_lhd[train_size+val_size:]
best_loss = float(5) # 임의 정의
patience = 30 #이 epoch동안 val_loss 기록이 단 한 번도 개선되지 않으면 iteration을 종료
batch_size=256 # 배치 크기 정의
losses = []

for epoch in range(2000):
    optimizer.zero_grad()
    for batch_start in range(0, train_size, batch_size):
        batch_x_train = x_train[batch_start:batch_start+batch_size]
        batch_t_train = t_train[batch_start:batch_start+batch_size]
        batch_x_pred_train = odeint(func, batch_x_train[0], batch_t_train).squeeze()
        loss = ((batch_x_pred_train - batch_x_train) ** 2).mean()
        loss.backward()
    optimizer.step()
    
    # Validation
    with torch.no_grad():
        x_pred_val = odeint(func, x_val[0], t_val).squeeze()
        val_loss = ((x_pred_val - x_val) ** 2).mean()
    
    losses.append(loss.item())
    
    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
        torch.save(func.state_dict(), 'best_model.pt')  # Save the best model
    else:
        counter += 1

    if epoch>500:
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    if epoch % 10 == 0:
            print("Epoch: {:3} | Loss: {:.9f} | Val Loss: {:.9f}".format(epoch, loss.item(), val_loss.item()))

with torch.no_grad():
            for name, param in func.named_parameters(): 
                print(f"{name}: {param.data}")# 레이어의 weight와 bias 출력
print("Epoch: {:3} | Loss: {:.6f} | Val Loss: {:.6f}".format(epoch, loss.item(), val_loss.item()))  


# Plotting the training loss and 2D Motion Data Prediction in subplots
fig, (ax1, ax2), = plt.subplots(1, 2, figsize=(12, 6))

# Plotting the training loss
ax1.plot(losses)
ax1.set_title('Training Loss (MSE)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
x_pred_train = odeint(func, x_train[0], t_train)
x_pred_test = odeint(func, x_test[0], t_test).squeeze()
ax2.plot(x_lhd[:, 0].detach().numpy(), x_lhd[:, 1].detach().numpy(), label='True trajectory')
ax2.plot(x_pred_train[:, 0].detach().numpy(), x_pred_train[:, 1].detach().numpy(), label='Neural ODE approximation')
ax2.plot(x_pred_val[:, 0].detach().numpy(), x_pred_val[:, 1].detach().numpy(), label='Predicted validation trajectory')
ax2.plot(x_pred_test[:, 0].detach().numpy(), x_pred_test[:, 1].detach().numpy(), label='Predicted test trajectory')
ax2.legend()
ax2.set_xlabel('X Position')
ax2.set_ylabel('Y Position')
ax2.set_title('2D Motion Data Prediction')
plt.tight_layout()
plt.show()


# Numerical Validation
slope, intercept, r_value, _, _ = linregress(x_val.flatten().detach().numpy(), x_pred_val.flatten().detach().numpy())
r_squared = r_value**2
mean_abs_rel_residual = mean_absolute_error(x_val.detach().numpy(), x_pred_val.detach().numpy()) / (x_val.abs().mean())
max_abs_rel_residual = max(np.max(np.abs(x_val.detach().numpy() - x_pred_val.detach().numpy()), axis=0) / x_val.abs().max())
table = PrettyTable()
table.field_names = ["Metric", "Value"]
table.add_row(["Squared correlation coefficient (r^2)", r_squared])
table.add_row(["Mean absolute relative residual", mean_abs_rel_residual])
table.add_row(["Maximum of absolute relative residuals", max_abs_rel_residual])
print(table)
