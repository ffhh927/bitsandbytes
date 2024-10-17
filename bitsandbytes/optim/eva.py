# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch

import bitsandbytes.functional as B_F
from bitsandbytes.optim.optimizer import Optimizer3State

import torch.nn as nn
import torch.nn.functional as F
#import horovod.torch as hvd
#import kfac.backend as backend
#backend.init("Horovod") 
import logging
logger = logging.getLogger()
    
class Eva8bit(Optimizer3State):
    def __init__(
        self,
        model,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=0.03,
        weight_decay=0,
        min_8bit_size=4096,
        block_wise=True,
        fac_update_freq=1,
        kfac_update_freq=1,
        kfac_batch_size=16,
        kl_clip=0.001,
        factor_decay=0.95,
        exclude_vocabulary_size=None,
        hook_enabled=True,
        exclude_parts=''
    ):
       
        super().__init__(
            "eva",
            model.parameters(),
            lr,
            betas,
            eps,
            weight_decay,
            8,
            None,
            min_8bit_size,
            block_wise,
            0.0,
            False,
            kl_clip,
        )

        self.fac_update_freq = fac_update_freq
        self.kfac_batch_size = kfac_batch_size
        self.kl_clip = kl_clip if (kl_clip is not None and kl_clip >= 0) else None
        self.factor_decay = factor_decay
        self.exclude_vocabulary_size = exclude_vocabulary_size
        self.hook_enabled = hook_enabled
        
        # register hooks
        self.supported_modules = {'Linear', 'Conv2d'}
        self.modules = []
        self.module_names = []
        self._register_module_hooks(model)

        # dictionaries keyed by `module` to storing KFs, inverse KFs, etc
        self.m_a, self.m_g = {}, {}
        self.max_ma, self.max_mg = {}, {}
        self.handles = []
        
        # scheduling results
        self.module_ranks = None

        self.steps = 0

        self.code = B_F.create_dynamic_map(signed=True).to(next(model.parameters()).device)  # 创建动态量化映射表

    ### Register hooks
    def set_hook_enabled(self, mode=True):
        self.hook_enabled = mode

    def _forward_hook_event(self, module, input):
        """Default: hook for saving input (a)"""
        if self.hook_enabled and torch.is_grad_enabled() and self.steps % self.fac_update_freq == 0:
            with torch.no_grad():
                new = get_vector_a(input[0].data[0:self.kfac_batch_size], module)
                if module not in self.m_a:
                    self.m_a[module] = new
                else:
                    #self.m_a[module].mul_(self.factor_decay).add_(new, alpha=1-self.factor_decay)
                    self.m_a[module].mul_(1-self.factor_decay).add_(new, alpha=self.factor_decay)
                    #xi =  math.pow(self.steps+1, -self.factor_decay)
                    #self.m_a[module].mul_(1-xi).add_(new, alpha=xi)
                self.m_a[module], quant_state = B_F.quantize_blockwise(self.m_a[module], code=self.code)
                self.max_ma[module] = quant_state.absmax
            #if backend.comm.size() > 1:
            #    self.handles.append(backend.comm.allreduce_async_(self.m_a[module], op=backend.comm.Average))

    def _backward_hook_event(self, module, grad_input, grad_output):
        """Default: hook for saving gradient w.r.t output (g)"""
        if self.hook_enabled and self.steps % self.fac_update_freq == 0:
            with torch.no_grad():
                new = get_vector_g(grad_output[0].data[0:self.kfac_batch_size], module)
                if module not in self.m_g:
                    self.m_g[module] = new
                else:
                    #self.m_g[module].mul_(self.factor_decay).add_(new, alpha=1-self.factor_decay)
                    self.m_g[module].mul_(1-self.factor_decay).add_(new, alpha=self.factor_decay)
                self.m_g[module], quant_state = B_F.quantize_blockwise(self.m_g[module], code=self.code)
                self.max_mg[module] = quant_state.absmax
                    #xi =  math.pow(self.steps+1, -self.factor_decay)
                    #self.m_g[module].mul_(1-xi).add_(new, alpha=xi)
            #if backend.comm.size() > 1:
            #    self.handles.append(backend.comm.allreduce_async_(self.m_g[module], op=backend.comm.Average))

    def _register_module_hooks(self, model):
        """Register forard/backward hooks to supported modules"""
        name_idx = 0
        for module in model.modules():
            classname = module.__class__.__name__
            if classname in self.supported_modules:
                if self.exclude_vocabulary_size is not None and classname == 'Linear' and module.out_features == self.exclude_vocabulary_size:
                    continue # exclude the pre-softmax linear layer in the Transformer model
                self.modules.append(module)
                module.register_forward_pre_hook(self._forward_hook_event)
                module.register_backward_hook(self._backward_hook_event)  # used in pytorch1.4, and pytorch1.8 (full_backward_hook is not fired when its grad_input is None)
                #module.register_full_backward_hook(self._backward_hook_event)  # used in pytorch1.10
                module_name = 'module_name_%s_%d' % (classname, name_idx)
                self.module_names.append(module_name)
                name_idx += 1
        #if backend.comm.rank() == 0:
         #   logger.info("#register modules: %s", len(self.modules))
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Arguments:
            closure (`Callable`, *optional*, defaults to `None`):
                A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        overflows = []

        if not self.initialized:
            self.check_overrides()
            self.to_gpu()  # needed for fairseq pure fp16 training
            self.initialized = True

        all_p = []
        all_p_grad = []
        # if self.is_paged: self.page_mng.prefetch_all()
        for module, (gindex, group) in zip(self.modules, enumerate(self.param_groups)):
            for p_item in module.parameters():
                if p_item.grad is None:
                    continue
                all_p.append(p_item.view(-1))
                all_p_grad.append(p_item.grad.view(-1))
            if len(all_p) == 0:
                continue
            p = torch.cat(all_p)
            p.grad = torch.cat(all_p_grad)
            state = self.state[p]
            if len(state) == 0:
                self.init_state(group, p, gindex, 0, self.m_a[module], self.m_g[module], self.max_ma[module], self.max_mg[module])
                #print(self.state)

            self.prefetch_state(p)
            self.update_step(group, p, gindex, 0)
            torch.cuda.synchronize()


        return loss


def _extract_patches(x, kernel_size, stride, padding):
    """Extract patches from convolutional layer

    Args:
      x: The input feature maps.  (batch_size, in_c, h, w)
      kernel_size: the kernel size of the conv filter (tuple of two elements)
      stride: the stride of conv operation  (tuple of two elements)
      padding: number of paddings. be a tuple of two elements
    
    Returns:
      Tensor of shape (batch_size, out_h, out_w, in_c*kh*kw)
    """
    if padding[0] + padding[1] > 0:
        x = F.pad(x, (padding[1], padding[1], padding[0],
                      padding[0])).data  # Actually check dims
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    x = x.view(
        x.size(0), x.size(1), x.size(2),
        x.size(3) * x.size(4) * x.size(5))
    return x

def get_vector_a(a, layer):
    """Return vectorized input activation (m_a)"""
    if isinstance(layer, nn.Linear): 
        a = torch.mean(a, list(range(len(a.shape)))[0:-1])
        if layer.bias is not None:
            a = torch.cat([a, a.new(1).fill_(1)])
        return a

    elif isinstance(layer, nn.Conv2d):
        # batch averag first
        a = torch.mean(a, dim=0, keepdim=True)
        # extract patch
        a = _extract_patches(a, layer.kernel_size, layer.stride, layer.padding)
        a = torch.mean(a, [0, 1, 2])
        if layer.bias is not None:
            a = torch.cat([a, a.new(1).fill_(1)])
        return a
        
    else:
        raise NotImplementedError("KFAC does not support layer: ".format(layer))

def get_vector_g(g, layer):
    """Return vectorized deviation w.r.t. the pre-activation output (m_g)"""
    if isinstance(layer, nn.Linear):
        g = torch.mean(g, list(range(len(g.shape)))[0:-1])
        return g

    elif isinstance(layer, nn.Conv2d):
        g = torch.mean(g, [0, 2, 3])
        return g

    else:
        raise NotImplementedError("KFAC does not support layer: ".format(layer))