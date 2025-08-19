import torch

# This command is more reliable for checking the version
print(f"PyTorch version: {torch.__version__}")
print("-" * 30)

# The main check to see if your GPU is detected
if torch.cuda.is_available():
    print("✅ Success! PyTorch can see your GPU.")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
else:
    print("❌ Error: PyTorch cannot see your GPU.")
    print("This confirms the script is using the slow CPU instead.")