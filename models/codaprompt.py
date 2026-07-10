import logging
from typing import Iterable

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.vit as vit

logger = logging.getLogger()

def ortho_penalty(t):
    identity = torch.eye(t.shape[0], device=t.device, dtype=t.dtype)
    return ((t @ t.T - identity) ** 2).mean()

def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p


class CodaPrompt(nn.Module):
    def __init__(self,
                 pos_e_prompt   : Iterable[int] = (0,1,2,3,4),
                 len_e_prompt   : int   = 20,
                 e_pool         : int   = 10,
                 task_num       : int   = 10,
                 num_classes    : int   = 100,
                 backbone_name  : str   = None,
                 key_dim        : int   = 768,
                 ortho_mu       : float = 0,
                 **kwargs):
        super().__init__()

        self.kwargs = kwargs
        pretrained = bool(kwargs.get("pretrained", True))
        self.ortho_mu = ortho_mu
        self.task_num = int(task_num)
        self.num_classes = num_classes

        if self.task_num <= 0:
            raise ValueError(f"task_num must be positive, got {self.task_num}")
        requested_e_pool = int(e_pool)
        if requested_e_pool <= 0:
            raise ValueError(f"e_pool must be positive, got {requested_e_pool}")

        self.task_count = 0

        # Backbone
        assert backbone_name is not None, 'backbone_name must be specified'
        # Use custom ViT model from models.vit to support local .npz loading
        if hasattr(vit, backbone_name):
            logger.info(f'Using custom ViT model: {backbone_name}')
            self.add_module('backbone', getattr(vit, backbone_name)(pretrained=pretrained, num_classes=num_classes))
        else:
            logger.info(f'Using timm model: {backbone_name}')
            self.add_module('backbone', timm.create_model(backbone_name, pretrained=pretrained, num_classes=num_classes))
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
        self.backbone.fc.weight.requires_grad = True
        self.backbone.fc.bias.requires_grad   = True

        # Slice the eprompt
        self.key_d = int(key_dim)
        self.len_e_prompt = int(len_e_prompt)
        if self.len_e_prompt <= 0:
            raise ValueError(
                f"len_e_prompt must be positive, got {self.len_e_prompt}"
            )
        self.requested_e_pool = requested_e_pool
        self.num_pt_per_task = max(
            1,
            (requested_e_pool + self.task_num - 1) // self.task_num,
        )
        self.e_pool = self.num_pt_per_task * self.task_num
        max_orthogonal_components = min(
            self.key_d,
            self.len_e_prompt * int(self.backbone.num_features),
        )
        if self.e_pool > max_orthogonal_components:
            raise ValueError(
                "CodaPrompt cannot construct the requested orthogonal pool: "
                f"effective e_pool={self.e_pool} exceeds feature capacity "
                f"{max_orthogonal_components}."
            )
        if self.e_pool != requested_e_pool:
            logger.info(
                "Expanding CodaPrompt e_pool from %s to %s so %s tasks receive %s components each",
                requested_e_pool,
                self.e_pool,
                self.task_num,
                self.num_pt_per_task,
            )
        self.e_length = len(pos_e_prompt) if pos_e_prompt else 0
        self.register_buffer('pos_e_prompt', torch.tensor(pos_e_prompt, dtype=torch.int64))
        for e in self.pos_e_prompt:
            p = tensor_prompt(self.e_pool, self.len_e_prompt, self.backbone.num_features)
            k = tensor_prompt(self.e_pool, self.key_d)
            a = tensor_prompt(self.e_pool, self.key_d)
            with torch.no_grad():
                p.copy_(self.gram_schmidt(p))
                k.copy_(self.gram_schmidt(k))
                a.copy_(self.gram_schmidt(a))
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)
            setattr(self, f'e_a_{e}',a)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        for e in self.pos_e_prompt:
            for kind in ("p", "k", "a"):
                key = f"e_{kind}_{int(e)}"
                saved = state_dict.get(key)
                current = getattr(self, key, None)
                if (
                    torch.is_tensor(saved)
                    and torch.is_tensor(current)
                    and saved.ndim > 0
                    and saved.shape != current.shape
                    and int(saved.shape[0]) == self.requested_e_pool
                    and self.requested_e_pool != self.e_pool
                ):
                    raise RuntimeError(
                        "This CodaPrompt checkpoint uses the legacy non-divisible "
                        f"prompt pool shape ({self.requested_e_pool}); the corrected "
                        f"effective pool is {self.e_pool}. Regenerate the base "
                        "checkpoint before rerunning CodaPrompt."
                    )
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def prompt_tuning(self,
                      x        : torch.Tensor,
                      g_prompt : torch.Tensor,
                      e_prompt : torch.Tensor,
                      **kwargs):

        B, N, C = x.size()

        e_prompt = e_prompt.contiguous().view(B, self.e_length, self.len_e_prompt, C)
        e_prompt = e_prompt + self.backbone.pos_embed[:,:1,:].unsqueeze(1).expand(B, self.e_length, self.len_e_prompt, C)

        for n, block in enumerate(self.backbone.blocks):
            pos_e = ((self.pos_e_prompt.eq(n)).nonzero()).squeeze()
            if pos_e.numel() != 0:
                x = torch.cat((x, e_prompt[:, pos_e]), dim = 1)

            x = block(x)
            x = x[:, :N, :]
        return x

    def forward(self, inputs : torch.Tensor) :
        with torch.no_grad():
            x = self.backbone.patch_embed(inputs)
            B, N, D = x.size()

            cls_token = self.backbone.cls_token.expand(B, -1, -1)
            token_appended = torch.cat((cls_token, x), dim=1)
            x = self.backbone.pos_drop(token_appended + self.backbone.pos_embed)
            query = self.backbone.blocks(x)
            query = self.backbone.norm(query)[:, 0]

        g_p = None
        e_p = None
        s = self.task_count * self.num_pt_per_task
        f = (self.task_count+1) * self.num_pt_per_task
        loss = 0
        for e in self.pos_e_prompt:
            K = getattr(self,f'e_k_{e}')
            A = getattr(self,f'e_a_{e}')
            p = getattr(self,f'e_p_{e}')
            if self.training:
                if self.task_count > 0:
                    K = torch.cat((K[:s].detach().clone(),K[s:f]), dim=0)
                    A = torch.cat((A[:s].detach().clone(),A[s:f]), dim=0)
                    p = torch.cat((p[:s].detach().clone(),p[s:f]), dim=0)
                else:
                    K = K[s:f]
                    A = A[s:f]
                    p = p[s:f]
            else:
                K = K[0:f]
                A = A[0:f]
                p = p[0:f]

            # with attention and cosine sim
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            a_querry = torch.einsum('bd,kd->bkd', query, A)
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            P_ = torch.einsum('bk,kld->bld', aq_k, p) # B, len_e_prompt, d

            if e_p is None:
                e_p = P_
            else:
                e_p = torch.cat((e_p, P_), dim=1)

            if self.training and self.ortho_mu > 0:
                loss += ortho_penalty(K) * self.ortho_mu
                loss += ortho_penalty(A) * self.ortho_mu
                loss += ortho_penalty(p.view(p.shape[0], -1)) * self.ortho_mu

        e_p = e_p.unsqueeze(1)
        x = self.prompt_tuning(self.backbone.pos_drop(token_appended + self.backbone.pos_embed), g_p, e_p)
        x = self.backbone.norm(x)
        cls_token = x[:, 0]
        x = self.backbone.fc(cls_token)

        if self.training:
            return x, loss
        else:
            return x

    def process_task_count(self):
        self.task_count += 1

        if self.task_count < self.task_num:

            # in the spirit of continual learning, we will reinit the new components
            # for the new task with Gram Schmidt
            #
            # in the original paper, we used ortho init at the start - this modification is more 
            # fair in the spirit of continual learning and has little affect on performance
            # 
            # code for this function is modified from:
            # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
            for e in self.pos_e_prompt:
                K = getattr(self,f'e_k_{e}')
                A = getattr(self,f'e_a_{e}')
                P = getattr(self,f'e_p_{e}')
                with torch.no_grad():
                    K.copy_(self.gram_schmidt(K))
                    A.copy_(self.gram_schmidt(A))
                    P.copy_(self.gram_schmidt(P))

    def loss_fn(self, output, target):
        return F.cross_entropy(output, target)

    @torch.no_grad()
    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = vv.shape
            vv = vv.view(vv.shape[0],-1)

        # swap rows and columns
        vv = vv.T

        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        pt = self.num_pt_per_task
        s = int(self.task_count * pt)
        f = int((self.task_count + 1) * pt)
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:,k])
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            logger.info('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T 

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)
        
        return uu
