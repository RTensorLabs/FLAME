"""Rewrite a safetensors file in place with every float32 tensor stored as float16.

Halves the browser download. The page converts the weights back to float32 when it loads them,
so the maths still runs in float32; only the stored values are rounded (to about 3 decimal
digits of precision).
"""
import sys

import numpy as np
from safetensors.numpy import load_file, save_file

path = sys.argv[1]
tensors = load_file(path)
for name, t in tensors.items():
    if t.dtype == np.float32:
        if np.abs(t).max() > 65504:  # largest finite float16
            sys.exit(f"{name}: values too large for float16")
        tensors[name] = t.astype(np.float16)
save_file(tensors, path)
print(f"converted {len(tensors)} tensors to float16: {path}")
