from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


class TopkR(nn.Module):
    def __init__(self, qk_dim, topk=4, qk_scale=None, param_routing=False, diff_routing=False):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        self.diff_routing = diff_routing

        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()

        self.routing_act = nn.Softmax(dim=-1)
    
    def forward(self, query:Tensor, key:Tensor)->Tuple[Tensor]:
        """
        Args:
            q, k: (n, p^2, c) tensor
        Return:
            r_weight, topk_index: (n, p^2, topk) tensor
        """
        if not self.diff_routing:
            query, key = query.detach(), key.detach()
        query_hat, key_hat = self.emb(query), self.emb(key) 
        attn_logit = (query_hat*self.scale) @ key_hat.transpose(-2, -1) 
        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1) 
        r_weight = self.routing_act(topk_attn_logit) 
        
        return r_weight, topk_index
        

class KVGather(nn.Module):
    def __init__(self, mul_weight='none'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight

    def forward(self, r_idx:Tensor, r_weight:Tensor, kv:Tensor):
        # select kv according to routing index
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)
        # print(r_idx.size(), r_weight.size())

        bb = kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1)
        topk_kv = torch.gather(bb, 
                                dim=2,
                                index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv) # (n, p^2, k, w^2, c_kv)
                               )

        if self.mul_weight == 'soft':
            topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv # (n, p^2, k, w^2, c_kv)
        elif self.mul_weight == 'hard':
            raise NotImplementedError('differentiable hard routing TBA')

        return topk_kv

class QKVLinear(nn.Module):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)
    
    def forward(self, x):
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim+self.dim], dim=-1)
        return q, kv
    
    
class SCA(nn.Module):
    def __init__(self, dim, num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=1, kv_downsample_mode='identity',
                 topk=4, param_routing=False, diff_routing=False, soft_routing=False, side_dwconv=3,
                 auto_pad=False):
        super().__init__()

        self.dim = dim
        self.n_win = n_win 
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        assert self.qk_dim % num_heads == 0 and self.dim % num_heads==0, 'qk_dim and dim must be divisible by num_heads!'
        self.scale = qk_scale or self.qk_dim ** -0.5

        self.lepe_x = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv//2, groups=dim) if side_dwconv > 0 else \
                    lambda x: torch.zeros_like(x)
        self.lepe_y = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv//2, groups=dim) if side_dwconv > 0 else \
                    lambda x: torch.zeros_like(x)

        self.topk = topk
        self.param_routing = param_routing
        self.diff_routing = diff_routing
        self.soft_routing = soft_routing
        # router
        assert not (self.param_routing and not self.diff_routing) 
        self.router = TopkR(qk_dim=self.qk_dim,
                                  qk_scale=self.scale,
                                  topk=self.topk,
                                  diff_routing=self.diff_routing,
                                  param_routing=self.param_routing)
        if self.soft_routing: 
            mul_weight = 'soft'
        elif self.diff_routing: 
            mul_weight = 'hard'
        else: 
            mul_weight = 'none'
        self.kv_gather = KVGather(mul_weight=mul_weight)


        self.qkv_x = QKVLinear(self.dim, self.qk_dim)
        self.qkv_y = QKVLinear(self.dim, self.qk_dim)
        self.wo = nn.Linear(dim, dim)
        
        self.kv_downsample_mode = kv_downsample_mode
        self.kv_per_win = kv_per_win
        self.kv_downsample_ratio = kv_downsample_ratio
        self.kv_downsample_kenel = kv_downsample_kernel
        if self.kv_downsample_mode == 'ada_avgpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'ada_maxpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveMaxPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'maxpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.MaxPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'avgpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.AvgPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'identity': # no kv downsampling
            self.kv_down = nn.Identity()
        elif self.kv_downsample_mode == 'fracpool':
            
            raise NotImplementedError('fracpool policy is not implemented yet!')
        elif kv_downsample_mode == 'conv':

            raise NotImplementedError('conv policy is not implemented yet!')
        else:
            raise ValueError(f'kv_down_sample_mode {self.kv_downsaple_mode} is not surpported!')


        self.attn_act = nn.Softmax(dim=-1)

        self.auto_pad=auto_pad

    def forward(self, x, y,ret_attn_mask=False):
        N, H, W, C = x.size()

        x = rearrange(x, "n (j h) (i w) c -> n (j i) h w c", j=self.n_win, i=self.n_win)
        y = rearrange(y, "n (j h) (i w) c -> n (j i) h w c", j=self.n_win, i=self.n_win)

        q_x, kv_x = self.qkv_x(x) 
        q_y, kv_y = self.qkv_y(y)

        q_x_pix = rearrange(q_x, 'n p2 h w c -> n p2 (h w) c')
        kv_x_pix = self.kv_down(rearrange(kv_x, 'n p2 h w c -> (n p2) c h w'))
        kv_x_pix = rearrange(kv_x_pix, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win)
        q_y_pix = rearrange(q_y, 'n p2 h w c -> n p2 (h w) c')
        kv_y_pix = self.kv_down(rearrange(kv_y, 'n p2 h w c -> (n p2) c h w'))
        kv_y_pix = rearrange(kv_y_pix, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win)
        

        q_x_win, k_x_win = q_x.mean([2, 3]), kv_x[..., 0:self.qk_dim].mean([2, 3])
        q_y_win, k_y_win = q_y.mean([2, 3]), kv_y[..., 0:self.qk_dim].mean([2, 3])

        lepe_x = self.lepe_x(rearrange(kv_x[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=self.n_win, i=self.n_win).contiguous())
        lepe_x = rearrange(lepe_x, 'n c (j h) (i w) -> n (j h) (i w) c', j=self.n_win, i=self.n_win)
        lepe_y = self.lepe_y(rearrange(kv_y[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=self.n_win, i=self.n_win).contiguous())
        lepe_y = rearrange(lepe_y, 'n c (j h) (i w) -> n (j h) (i w) c', j=self.n_win, i=self.n_win)

        r_x_weight, r_x_idx = self.router(q_x_win, k_x_win)
        r_y_weight, r_y_idx = self.router(q_y_win, k_y_win)
        
        
        kv_x_pix_sel = self.kv_gather(r_idx=r_x_idx, r_weight=r_x_weight, kv=kv_x_pix) 
        k_x_pix_sel, v_x_pix_sel = kv_x_pix_sel.split([self.qk_dim, self.dim], dim=-1)
        kv_y_pix_sel = self.kv_gather(r_idx=r_y_idx, r_weight=r_y_weight, kv=kv_y_pix) 
        k_y_pix_sel, v_y_pix_sel = kv_y_pix_sel.split([self.qk_dim, self.dim], dim=-1)

        k_x_pix_sel = rearrange(k_x_pix_sel, 'n p2 k w2 (m c) -> (n p2) m c (k w2)', m=self.num_heads) 
        v_x_pix_sel = rearrange(v_x_pix_sel, 'n p2 k w2 (m c) -> (n p2) m (k w2) c', m=self.num_heads) 
        q_x_pix = rearrange(q_x_pix, 'n p2 w2 (m c) -> (n p2) m w2 c', m=self.num_heads) 

        k_y_pix_sel = rearrange(k_y_pix_sel, 'n p2 k w2 (m c) -> (n p2) m c (k w2)', m=self.num_heads) 
        v_y_pix_sel = rearrange(v_y_pix_sel, 'n p2 k w2 (m c) -> (n p2) m (k w2) c', m=self.num_heads) 
        q_y_pix = rearrange(q_y_pix, 'n p2 w2 (m c) -> (n p2) m w2 c', m=self.num_heads) 
        
        
        # param-free multihead attention
        attn_weight_x = (q_x_pix * self.scale) @ k_x_pix_sel 
        attn_weight_x = self.attn_act(attn_weight_x)
        attn_weight_y = (q_y_pix * self.scale) @ k_y_pix_sel 
        attn_weight_y = self.attn_act(attn_weight_y)
        
        out_x = attn_weight_y @ v_x_pix_sel 
        out_x = rearrange(out_x, '(n j i) m (h w) c -> n (j h) (i w) (m c)', j=self.n_win, i=self.n_win,
                        h=H//self.n_win, w=W//self.n_win)
        out_y = attn_weight_x @ v_y_pix_sel 
        out_y = rearrange(out_y, '(n j i) m (h w) c -> n (j h) (i w) (m c)', j=self.n_win, i=self.n_win,
                        h=H//self.n_win, w=W//self.n_win)
        
        out_x = out_x + lepe_x
        out_y = out_y + lepe_y
        
        return out_x,out_y
