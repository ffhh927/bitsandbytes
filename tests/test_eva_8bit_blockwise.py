import os
from os.path import join
import shutil
import time
import uuid

import torch
import pytest

import bitsandbytes as bnb
import bitsandbytes.functional as F
from tests.helpers import describe_dtype, id_formatter

# Define a function to assert closeness with a custom tolerance and error count
def assert_most_approx_close(a, b, rtol=1e-3, atol=1e-3, max_error_count=0):
    idx = torch.isclose(a, b, rtol=rtol, atol=atol)
    error_count = (idx == 0).sum().item()
    if error_count > max_error_count:
        print(f"Too many values not close: assert {error_count} < {max_error_count}")
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)

# Define a function to get a temporary directory
def get_temp_dir():
    path = f"/tmp/autoswap/{uuid.uuid4()}"
    os.makedirs(path, exist_ok=True)
    return path

# Define a function to remove a path
def rm_path(path):
    shutil.rmtree(path)

# Define optimizers and their corresponding state names
str2optimizers = {
    "eva8bit_blockwise": (
        torch.optim.Adam,  # For comparison
        lambda model: bnb.optim.Eva8bit(
            model, 
            block_wise=True,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=0.03,
            kl_clip=0.001,
            weight_decay=0,
            kfac_update_freq=1,
            kfac_batch_size=16,
            fac_update_freq=1,
            factor_decay=0.95
        )
    )
}
str2statenames = {
    "eva8bit_blockwise": [
        ("exp_avg", "state1", "qmap1", "absmax1"),
        ("exp_avg_sq", "state2", "qmap2", "absmax2"),
        ("kfac_A", "state3", "qmap3", "absmax3"),
    ]
}

# Test Eva8bit blockwise optimizer
@pytest.mark.parametrize("gtype", [torch.float32, torch.float16, torch.bfloat16], ids=describe_dtype)
@pytest.mark.parametrize("dim2", [32, 1024, 4097], ids=id_formatter("dim2"))
@pytest.mark.parametrize("dim1", [1024], ids=id_formatter("dim1"))
def test_eva8bit_blockwise(dim1, dim2, gtype):
    if gtype == torch.bfloat16:
        pytest.skip()
    if dim1 == 1 and dim2 == 1:
        return
    
    # Create a dummy model instead of passing individual parameters
    model = torch.nn.Sequential(
        torch.nn.Linear(dim2, dim2).to("cuda", dtype=gtype),
        torch.nn.ReLU(),
        torch.nn.Linear(dim2, dim2).to("cuda", dtype=gtype)
    )
    
    p1 = torch.randn(dim1, dim2, device="cuda", dtype=gtype) * 0.1
    p2 = p1.clone()
    p1 = p1.float()

    torch_optimizer = str2optimizers["eva8bit_blockwise"][0](model.parameters())
    bnb_optimizer = str2optimizers["eva8bit_blockwise"][1](model)

    if gtype == torch.float32:
        atol, rtol = 3e-3, 1e-3
        patol, prtol = 1e-5, 1e-3
    elif gtype == torch.bfloat16:
        atol, rtol = 3e-3, 1e-3
        patol, prtol = 1e-4, 1e-2
    else:
        atol, rtol = 3e-3, 1e-3
        patol, prtol = 1e-5, 1e-3

    errors = []
    relerrors = []

    for i in range(100):
        g = torch.randn(dim1, dim2, device="cuda", dtype=gtype) * 0.01
        p1.grad = g.clone().float()
        p2.grad = g.clone()

        bnb_optimizer.step()
        torch_optimizer.step()


