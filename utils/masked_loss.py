import torch
import numpy as np


def masked_mse_loss(y_pred, y_true, null_val):
    mask = torch.ne(y_true, null_val).float()
    mask = torch.divide(mask, torch.mean(mask))
    loss = torch.square(y_pred - y_true)
    loss = torch.mul(loss, mask)
    loss[torch.isnan(loss)] = 0.
    return torch.mean(loss)


def masked_loss_np(y_pred, y_true, loss_type):
    mask = np.greater(y_true, 1e-1).astype(np.float)
    mask = np.divide(mask, np.mean(mask))
    if loss_type == "mae":
        loss = np.abs(np.subtract(y_pred, y_true))
    elif loss_type == "mape":
        loss = np.divide(np.abs(np.subtract(y_pred, y_true)), y_true)
    elif loss_type == "rmse":
        loss = np.power(np.subtract(y_pred, y_true), 2)
    else:
        raise ValueError("No Such Loss!")
    loss = np.nan_to_num(np.multiply(loss, mask))
    return np.mean(loss)




