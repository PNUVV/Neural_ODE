from sklearn.metrics import mean_absolute_error
from scipy.stats import linregress
import numpy as np
def return_numerical_validation(x_data, x_pred):
    slope, intercept, r_value, _, _ = linregress(x_data.flatten().detach().numpy(), x_pred.flatten().detach().numpy())
    r_squared = r_value**2
    mean_abs_rel_residual = mean_absolute_error(x_data.detach().numpy(), x_pred.detach().numpy()) / (x_data.abs().mean())
    max_abs_rel_residual = max(np.max(np.abs(x_data.detach().numpy() - x_pred.detach().numpy()), axis=0) / x_data.abs().max())

    return r_squared, mean_abs_rel_residual, max_abs_rel_residual
