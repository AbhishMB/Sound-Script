import os

import torch

# cuDNN's kernel modules need a lot of host memory to initialise and can make
# CUDA calls fail ("unknown error" / "out of memory") on machines that are short
# of RAM / commit memory. On a normal GPU machine (Colab, workstation) leave it
# on; set AMT_CUDNN=0 to turn it off on a memory-starved machine.
torch.backends.cudnn.enabled = os.environ.get("AMT_CUDNN", "1") != "0"
