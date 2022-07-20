import itertools as it
from easydict import EasyDict as edict
import torch

# function to generate descartes mul of hyperparameter
def descartes_mul(configs):
    params = []
    names = []
    for x in configs:
        names.append(x)
        params.append(configs[x])
    return {"names" : names, "params" : torch.cartesian_prod(*params)}



HYPER_PARAM = edict()

# TRAIN
HYPER_PARAM.TRAIN_ALPHA = torch.arange(0, 2)
HYPER_PARAM.TRAIN_BETA = torch.arange(2, 4)
HYPER_PARAM.TRAIN_LAMBDA = torch.arange(4, 6)

# MODEL

# SEED
HYPER_PARAM.SEED = torch.arange(0, 20)



################## EXAMPLE
print(descartes_mul(HYPER_PARAM))
