import wandb

wandb.finish()




# import torch

# # Check CUDA availability
# if torch.cuda.is_available():
#     # Get the cuDNN version
#     cudnn_version = torch.backends.cudnn.version()
#     print(f"cuDNN version: {cudnn_version}")

#     # Optionally, check the CUDA version as well
#     cuda_version = torch.version.cuda
#     print(f"CUDA version: {cuda_version}")

#     # Check PyTorch version
#     pytorch_version = torch.__version__
#     print(f"PyTorch version: {pytorch_version}")

# else:
#     print("CUDA is not available. cuDNN version cannot be determined.")