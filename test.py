import torch
print(torch.cuda.is_available())  # Doit renvoyer True
print(torch.__version__)          # Version avec cuXXX dans le nom si CUDA est activé
