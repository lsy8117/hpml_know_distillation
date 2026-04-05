
# Hardware/Environment

Benchmarking notebook is currently running on Google Colab, using Tesla T4 (A100 does not seem to effectively speedup the inference, may explore it later on if we switch to batch inference)
PyTorch version : 2.10.0+cu128
CUDA available  : True
GPU             : Tesla T4
VRAM            : 15.6 GB

# Benchmark Configuration

- One sample inference at a time
- 0-shot
- TEMPERATURE = None (greedy decoding)
- torch_dtype = torch.float16
- MAX_NEW_TOKENS = 512

