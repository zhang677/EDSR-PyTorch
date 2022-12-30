import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_503460026817608471(nn.Module):
    def __init__(self, c: int, h: int):
        # Configurations
        super(Kernel_503460026817608471, self).__init__()
        self.g = 4
        self.n, self.c = None, c
        self.h = h
        
        # Kernels
        # Input: p_0
        pass
        # Mix: p_1
        self.p_1_w = nn.Parameter(torch.ones((self.c, 1, )), requires_grad=True)
        bound = math.sqrt(3.0 / (self.c))
        nn.init.uniform_(self.p_1_w, a=-bound, b=bound)
        # Shift_0/1/C_K3: p_2
        self.p_2_0_1 = random.randint(-3, 3)
        # BAdd: p_3
        pass
        # BMul: p_4
        pass
        # ReLU: p_5
        pass
        # Group_0_C/G: p_6
        pass
        # Scale_0/1/C_1/0/H: p_7
        self.p_7_w = nn.Parameter(torch.ones((1, self.c, self.h,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_7_w, std=.02)
        # BMM_1_0: p_8
        pass
        # BMM_0_1: p_9
        pass
        # BMM_0_1: p_10
        pass
        # BMin: p_11
        pass
        # BMM_1_0: p_12
        pass
        # BMM_1_0: p_13
        pass
        # BMM_1_1: p_14
        pass
        # BMM_0_0: p_15
        pass
        # BMM_1_1: p_16
        pass
        # BMM_1_0: p_17
        pass
        # Output: p_18
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h) == tuple(t_0.size())
        # Mix: p_1
        t_1 = torch.einsum('abc,bd->adc', [t_0, self.p_1_w]).view(self.n, self.h).contiguous()
        # Shift_0/1/C_K3: p_2
        t_2 = torch.roll(t_0, self.p_2_0_1, 1)
        # BAdd: p_3
        t_3_lhs = t_1.view(self.n, 1, self.h)
        t_3_rhs = t_2.view(self.n, self.c, self.h)
        t_3 = t_3_lhs + t_3_rhs
        t_3 = t_3.view(self.n, self.c, self.h)
        # BMul: p_4
        t_4 = t_0 * t_3
        # ReLU: p_5
        t_5 = torch.relu(t_0)
        # Group_0_C/G: p_6
        t_6 = t_3.view(self.n, self.c // self.g, self.g, self.h)
        # Scale_0/1/C_1/0/H: p_7
        t_7 = self.p_7_w * t_5
        # BMM_1_0: p_8
        t_8_lhs = t_5.view(self.n, self.c, self.h).transpose(1, 2)        
        t_8_rhs = t_0.view(self.n, self.c, self.h)        
        t_8 = F.softmax(torch.bmm(t_8_lhs, t_8_rhs) / math.sqrt(self.c), dim=-1)
        t_8 = t_8.view(self.n, self.h, self.h)
        # BMM_0_1: p_9
        t_9_lhs = t_5.view(self.n, self.c, self.h)        
        t_9_rhs = t_4.view(self.n, self.c, self.h).transpose(1, 2)        
        t_9 = torch.bmm(t_9_lhs, t_9_rhs) / math.sqrt(self.h)
        t_9 = t_9.view(self.n, self.c, self.c)
        # BMM_0_1: p_10
        t_10_lhs = t_7.view(self.n, self.c, self.h)        
        t_10_rhs = t_6.view(self.n, self.c, self.h).transpose(1, 2)        
        t_10 = F.softmax(torch.bmm(t_10_lhs, t_10_rhs) / math.sqrt(self.h), dim=-1)
        t_10 = t_10.view(self.n, self.c, self.c // self.g, self.g)
        # BMin: p_11
        t_11 = torch.minimum(t_4, t_0)
        # BMM_1_0: p_12
        t_12_lhs = t_9.view(self.n, self.c, self.c).transpose(1, 2)        
        t_12_rhs = t_7.view(self.n, self.c, self.h)        
        t_12 = torch.bmm(t_12_lhs, t_12_rhs) / math.sqrt(self.c)
        t_12 = t_12.view(self.n, self.c, self.h)
        # BMM_1_0: p_13
        t_13_lhs = t_11.view(self.n, self.c, self.h).transpose(1, 2)        
        t_13_rhs = t_2.view(self.n, self.c, self.h)        
        t_13 = torch.bmm(t_13_lhs, t_13_rhs) / math.sqrt(self.c)
        t_13 = t_13.view(self.n, self.h, self.h)
        # BMM_1_1: p_14
        t_14_lhs = t_9.view(self.n, self.c, self.c).transpose(1, 2)        
        t_14_rhs = t_10.view(self.n, self.c, self.c).transpose(1, 2)        
        t_14 = F.softmax(torch.bmm(t_14_lhs, t_14_rhs) / math.sqrt(self.c), dim=-1)
        t_14 = t_14.view(self.n, self.c, self.c)
        # BMM_0_0: p_15
        t_15_lhs = t_13.view(self.n, self.h, self.h)        
        t_15_rhs = t_8.view(self.n, self.h, self.h)        
        t_15 = F.softmax(torch.bmm(t_15_lhs, t_15_rhs) / math.sqrt(self.h), dim=-1)
        t_15 = t_15.view(self.n, self.h, self.h)
        # BMM_1_1: p_16
        t_16_lhs = t_12.view(self.n, self.c, self.h).transpose(1, 2)        
        t_16_rhs = t_14.view(self.n, self.c, self.c).transpose(1, 2)        
        t_16 = torch.bmm(t_16_lhs, t_16_rhs) / math.sqrt(self.c)
        t_16 = t_16.view(self.n, self.h, self.c)
        # BMM_1_0: p_17
        t_17_lhs = t_16.view(self.n, self.h, self.c).transpose(1, 2)        
        t_17_rhs = t_15.view(self.n, self.h, self.h)        
        t_17 = torch.bmm(t_17_lhs, t_17_rhs) / math.sqrt(self.h)
        t_17 = t_17.view(self.n, self.c, self.h)
        # Output: p_18
        return t_17.view(self.n, self.c, self.h)

