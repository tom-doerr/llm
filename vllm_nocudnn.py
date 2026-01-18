import torch
torch.backends.cudnn.enabled = False
import sys
import subprocess
cmd = ["vllm", "serve"] + sys.argv[1:]
subprocess.call(cmd)
