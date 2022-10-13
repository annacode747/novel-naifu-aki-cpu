import torch
import os

lowvram = False if os.environ.get('LOWVRAM') == "0" else True
dtype = torch.float32 if os.environ.get('DTYPE', 'float32') else torch.float16


print("lowvram: " + str(lowvram))
print("using dtype: " + os.environ.get('DTYPE', 'float32'))
