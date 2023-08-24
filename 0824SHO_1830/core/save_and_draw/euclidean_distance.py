import torch
def euclidean_distance(x1, x2):
    if x1.requires_grad:
        x1 = x1
    if x2.requires_grad:
        x2 = x2
    return torch.sqrt(torch.sum((x1 - x2) ** 2, dim=1))
