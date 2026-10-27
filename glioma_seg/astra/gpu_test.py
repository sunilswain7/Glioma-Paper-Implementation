"""Run as an Astra GPU job: reports what the compute node actually gives us."""
import os, time, torch
print("GPU Availability:", torch.cuda.is_available())
print("Torch:", torch.__version__, "| CUDA runtime:", torch.version.cuda)
if torch.cuda.is_available():
    print("Visible GPUs:", torch.cuda.device_count(), "|", torch.cuda.get_device_name(0))
    print("bf16 supported:", torch.cuda.is_bf16_supported())
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(20): a @ a
    torch.cuda.synchronize(); print(f"bf16 matmul: {20*2*8192**3/(time.time()-t)/1e12:.0f} TFLOPS")
os.system("nvidia-smi; echo CPUS $(nproc); free -g; df -h $HOME; "
          "curl -s -o /dev/null -w 'internet from compute node: %{http_code}\\n' https://pypi.org "
          "|| echo 'internet from compute node: NO'")
