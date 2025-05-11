import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 560  # Must be a multiple of 14
PATCH_SIZE = 14
