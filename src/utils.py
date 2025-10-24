import os, json, random
import numpy as np
import torch

class JsonLogger:
    def __init__(self, outdir):
        os.makedirs(outdir, exist_ok=True)
        self.path = os.path.join(outdir, "log.jsonl")
    def write(self, obj):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True