import os
import json
import sys
from .utils import dict_product, iwt

sys.path.append("../")

with open("../base_config.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "loss": ['GCELoss', 'SCELoss', 'ELRLoss'],
    "weight_decay": [1e-2, 1e-3, 1e-4, 0.0],
    "seed": [123, 231, 312]
}

all_configs = [{**BASE_CONFIG, **p} for p in dict_product(PARAMS)]
if os.path.isdir('multisteps/') or os.path.isdir('cosine/'):
    raise ValueError("Please delete the 'multisteps/' and 'cosine/' directories")
os.makedirs("multisteps/")
os.makedirs("cosine/")

for i, config in enumerate(all_config):
    with open(f"agent_configs/{i}.json", "w") as f:
        json.dump(config, f)