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
    "adam8bit_blockwise": (
        torch.optim.Adam,
        lambda pxx: bnb.optim.Adam8bit(pxx, block_wise=True)
    )
}
str2statenames = {
    "adam8bit_blockwise": [
        ("exp_avg", "state1", "qmap1", "absmax1"),
        ("exp_avg_sq", "state2", "qmap2", "absmax2"),
    ]
}

# Test Adam8bit blockwise optimizer
@pytest.mark.parametrize("gtype", [torch.float32, torch.float16, torch.bfloat16], ids=describe_dtype)
@pytest.mark.parametrize("dim2", [32, 1024, 4097], ids=id_formatter("dim2"))
@pytest.mark.parametrize("dim1", [1024], ids=id_formatter("dim1"))
def test_adam8bit_blockwise(dim1, dim2, gtype):
    if gtype == torch.bfloat16:
        pytest.skip()
    if dim1 == 1 and dim2 == 1:
        return
    p1 = torch.randn(dim1, dim2, device="cuda", dtype=gtype) * 0.1
    p2 = p1.clone()
    p1 = p1.float()
    blocksize = 2048

    torch_optimizer = str2optimizers["adam8bit_blockwise"][0]([p1])
    bnb_optimizer = str2optimizers["adam8bit_blockwise"][1]([p2])

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

        assert_most_approx_close(p1, p2.float(), patol, prtol, max_error_count=5)

        dequant_states = []
        for name1, name2, qmap, max_val in str2statenames["adam8bit_blockwise"]:
            s1 = F.dequantize_blockwise(
                code=bnb_optimizer.state[p2][qmap],
                absmax=bnb_optimizer.state[p2][max_val],
                A=bnb_optimizer.state[p2][name2],
                blocksize=blocksize,
            )
            num_not_close = torch.isclose(torch_optimizer.state[p1][name1], s1, atol=atol, rtol=rtol) == 0
            dequant_states.append(s1.clone())

        err = torch.abs(p1 - p2)
        relerr = err / (torch.abs(p1) + 1e-9)
        if gtype == torch.bfloat16:
            assert err.mean() < 0.00015
            assert relerr.mean() < 0.0016
        else:
            assert err.mean() < 0.00012
            assert relerr.mean() < 0.0012

        errors.append(err.mean().item())
        relerrors.append(relerr.mean().item())

        if i % 10 == 0 and i > 0:
            for (name1, name2, qmap, max_val), s in zip(str2statenames["adam8bit_blockwise"], dequant_states):
                s1cpy = s.clone()
                raws1cpy = bnb_optimizer.state[p2][name2].clone()
                qmap1 = bnb_optimizer.state[p2][qmap].clone()

                path = get_temp_dir()
                torch.save(bnb_optimizer.state_dict(), join(path, "opt.pt"))
                del bnb_optimizer
                bnb_optimizer = None
                bnb_optimizer = str2optimizers["adam8bit_blockwise"][1]([p2])
                bnb_optimizer.load_state_dict(torch.load(join(path, "opt.pt")))
                rm_path(path)
                torch.testing.assert_close(raws1cpy, bnb_optimizer.state[p2][name2])
                torch.testing.assert_close(qmap1, bnb_optimizer.state[p2][qmap])

                s1 = F.dequantize_blockwise(
                    code=bnb_optimizer.state[p2][qmap],
                    absmax=bnb_optimizer.state[p2][max_val],
                    A=bnb_optimizer.state[p2][name2],
                    blocksize=blocksize,
                )
                torch.testing.assert_close(s1cpy, s1)

                num_not_close = torch.isclose(torch_optimizer.state[p1][name1], s1, atol=atol, rtol=rtol) == 0
                assert num_not_close.sum().item() < 20
            assert_most_approx_close(p1, p2.float(), patol, prtol, max_error_count=5)

        p1.data = p1.data.to(gtype).float()
        p2.copy_(p1.data)
        torch.testing.assert_close(p1.to(gtype), p2)
        for (name1, name2, qmap, max_val), s in zip(str2statenames["adam8bit_blockwise"], dequant_states):
            torch_optimizer.state[p1][name1].copy_(s.data)

    # Print error statistics
    print(sum(errors)/len(errors))
    print(sum(relerrors)/len(relerrors))