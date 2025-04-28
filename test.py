import yaml
import torch
from fvcore.nn import parameter_count_table, FlopCountAnalysis
from LMLTransformer_mod import LMLTransformer

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

model = LMLTransformer(
        block_type=config["model"]["block_type"],
        n_blocks=config["model"]["n_blocks"],
        n_sub_blocks=config["model"]["n_sub_blocks"],
        levels=config["model"]["levels"],
        window_size=config["model"]["window_size"],
        dim=config["model"]["dim"],
        level_dim=config["model"]["level_dim"],
        n_heads=config["model"]["n_heads"],
        n_heads_fuse=config["model"]["n_heads_fuse"],
        feature_dim=config["model"]["feature_dim"],
        scale_factor=config["model"]["scale_factor"]
)

print(parameter_count_table(model))

tensor = torch.randn(1,3,64,64)
flop_count = FlopCountAnalysis(model, tensor)
flops = flop_count.total()
print(flops)