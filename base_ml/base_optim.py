# -*- coding: utf-8 -*-
# Wrappping all available PyTorch Optimizer


from torch.optim import (
    ASGD,
    LBFGS,
    SGD,
    Adadelta,
    Adagrad,
    Adam,
    Adamax,
    AdamW,
    RAdam,
    RMSprop,
    Rprop,
    SparseAdam,
)

OPTI_DICT = {
    "Adadelta": Adadelta,
    "Adagrad": Adagrad,
    "Adam": Adam,
    "AdamW": AdamW,
    "SparseAdam": SparseAdam,
    "Adamax": Adamax,
    "ASGD": ASGD,
    "LBFGS": LBFGS,
    "RAdam": RAdam,
    "RMSprop": RMSprop,
    "Rprop": Rprop,
    "SGD": SGD,
}
