import torch
import numpy as np
from torch import nn
import math


class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        # pickup
        self.W1_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W1_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W1_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.W2_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W2_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W2_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.W3_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W3_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W3_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        # delivery
        self.W4_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W4_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W4_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))
        
        self.W5_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W5_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W5_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))
        
        self.W6_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W6_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W6_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)  # [batch_size * graph_size, embed_dim]
        qflat = q.contiguous().view(-1, input_dim)  # [batch_size * n_query, embed_dim]

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # pickup -> its delivery attention
        n_pick = (graph_size - 1) // 2
        shp_delivery = (self.n_heads, batch_size, n_pick, -1)
        shp_q_pick = (self.n_heads, batch_size, n_pick, -1)

        # pickup -> all pickups attention
        shp_allpick = (self.n_heads, batch_size, n_pick, -1)
        shp_q_allpick = (self.n_heads, batch_size, n_pick, -1)

        # pickup -> all pickups attention
        shp_alldelivery = (self.n_heads, batch_size, n_pick, -1)
        shp_q_alldelivery = (self.n_heads, batch_size, n_pick, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # pickup -> its delivery
        pick_flat = h[:, 1:n_pick + 1, :].contiguous().view(-1, input_dim)  # [batch_size * n_pick, embed_dim]
        delivery_flat = h[:, n_pick + 1:, :].contiguous().view(-1, input_dim)  # [batch_size * n_pick, embed_dim]


        # pickup -> its delivery attention
        Q_pick = torch.matmul(pick_flat, self.W1_query).view(shp_q_pick)  # (self.n_heads, batch_size, n_pick, key_size)
        K_delivery = torch.matmul(delivery_flat, self.W_key).view(shp_delivery)  # (self.n_heads, batch_size, n_pick, -1)
        V_delivery = torch.matmul(delivery_flat, self.W_val).view(shp_delivery)  # (n_heads, batch_size, n_pick, key/val_size)

        # pickup -> all pickups attention
        Q_pick_allpick = torch.matmul(pick_flat, self.W2_query).view(shp_q_allpick) # (self.n_heads, batch_size, n_pick, -1)
        K_allpick = torch.matmul(pick_flat, self.W_key).view(shp_allpick)  # [self.n_heads, batch_size, n_pick, key_size]
        V_allpick = torch.matmul(pick_flat, self.W_val).view(shp_allpick)  # [self.n_heads, batch_size, n_pick, key_size]

        # pickup -> all delivery
        Q_pick_alldelivery = torch.matmul(pick_flat, self.W3_query).view(shp_q_alldelivery)  # (self.n_heads, batch_size, n_pick, key_size)
        K_alldelivery = torch.matmul(delivery_flat, self.W_key).view(shp_alldelivery)  # (self.n_heads, batch_size, n_pick, -1)
        V_alldelivery = torch.matmul(delivery_flat, self.W_val).view(shp_alldelivery)  # (n_heads, batch_size, n_pick, key/val_size)

        # pickup -> its delivery
        V_additional_delivery = torch.cat([  # [n_heads, batch_size, graph_size, key_size]
            torch.zeros(self.n_heads, batch_size, 1, self.input_dim // self.n_heads, dtype=V.dtype, device=V.device),
            V_delivery,  # [n_heads, batch_size, n_pick, key/val_size]
            torch.zeros(self.n_heads, batch_size, n_pick, self.input_dim // self.n_heads, dtype=V.dtype, device=V.device)
            ], 2)


        # delivery -> its pickup attention
        Q_delivery = torch.matmul(delivery_flat, self.W4_query).view(shp_delivery)  # (self.n_heads, batch_size, n_pick, key_size)
        K_pick = torch.matmul(pick_flat, self.W_key).view(shp_q_pick)  # (self.n_heads, batch_size, n_pick, -1)
        V_pick = torch.matmul(pick_flat, self.W_val).view(shp_q_pick)  # (n_heads, batch_size, n_pick, key/val_size)

        # delivery -> all delivery attention
        Q_delivery_alldelivery = torch.matmul(delivery_flat, self.W5_query).view(shp_alldelivery) # (self.n_heads, batch_size, n_pick, -1)
        K_alldelivery2 = torch.matmul(delivery_flat, self.W_key).view(shp_alldelivery)  # [self.n_heads, batch_size, n_pick, key_size]
        V_alldelivery2 = torch.matmul(delivery_flat, self.W_val).view(shp_alldelivery)  # [self.n_heads, batch_size, n_pick, key_size]
        
        # delivery -> all pickup
        Q_delivery_allpickup = torch.matmul(delivery_flat, self.W6_query).view(shp_alldelivery)  # (self.n_heads, batch_size, n_pick, key_size)
        K_allpickup2 = torch.matmul(pick_flat, self.W_key).view(shp_q_alldelivery)  # (self.n_heads, batch_size, n_pick, -1)
        V_allpickup2 = torch.matmul(pick_flat, self.W_val).view(shp_q_alldelivery)  # (n_heads, batch_size, n_pick, key/val_size)

        # delivery -> its pick up
#        V_additional_pick = torch.cat([  # [n_heads, batch_size, graph_size, key_size]
#            torch.zeros(self.n_heads, batch_size, 1, self.input_dim // self.n_heads, dtype=V.dtype, device=V.device),
#            V_delivery2,  # [n_heads, batch_size, n_pick, key/val_size]
#            torch.zeros(self.n_heads, batch_size, n_pick, self.input_dim // self.n_heads, dtype=V.dtype, device=V.device)
#            ], 2)        
        V_additional_pick = torch.cat([  # [n_heads, batch_size, graph_size, key_size]
            torch.zeros(self.n_heads, batch_size, 1, self.input_dim // self.n_heads, dtype=V.dtype, device=V.device),
            torch.zeros(self.n_heads, batch_size, n_pick, self.input_dim // self.n_heads, dtype=V.dtype, device=V.device),
            V_pick  # [n_heads, batch_size, n_pick, key/val_size]
            ], 2)  


        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        ##Pick up
        # ??pair???attention??
        compatibility_pick_delivery = self.norm_factor * torch.sum(Q_pick * K_delivery, -1)  # element_wise, [n_heads, batch_size, n_pick]
        # [n_heads, batch_size, n_pick, n_pick]
        compatibility_pick_allpick = self.norm_factor * torch.matmul(Q_pick_allpick, K_allpick.transpose(2, 3))# [n_heads, batch_size, n_pick, n_pick]

        compatibility_pick_alldelivery = self.norm_factor * torch.matmul(Q_pick_alldelivery, K_alldelivery.transpose(2, 3))  # [n_heads, batch_size, n_pick, n_pick]
        
        ##Delivery
        compatibility_delivery_pick = self.norm_factor * torch.sum(Q_delivery * K_pick, -1)  # element_wise, [n_heads, batch_size, n_pick]
        
        compatibility_delivery_alldelivery = self.norm_factor * torch.matmul(Q_delivery_alldelivery, K_alldelivery2.transpose(2, 3))# [n_heads, batch_size, n_pick, n_pick]
        
        compatibility_delivery_allpick = self.norm_factor * torch.matmul(Q_delivery_allpickup, K_allpickup2.transpose(2, 3))  # [n_heads, batch_size, n_pick, n_pick]
        
        ##Pick up->
        # compatibility_additional?pickup????delivery????attention(size 1),1:n_pick+1??attention,depot?delivery??
        compatibility_additional_delivery = torch.cat([  # [n_heads, batch_size, graph_size, 1]
            -np.inf * torch.ones(self.n_heads, batch_size, 1, dtype=compatibility.dtype, device=compatibility.device),
            compatibility_pick_delivery,  # [n_heads, batch_size, n_pick]
            -np.inf * torch.ones(self.n_heads, batch_size, n_pick, dtype=compatibility.dtype, device=compatibility.device)
            ], -1).view(self.n_heads, batch_size, graph_size, 1)

        compatibility_additional_allpick = torch.cat([  # [n_heads, batch_size, graph_size, n_pick]
            -np.inf * torch.ones(self.n_heads, batch_size, 1, n_pick, dtype=compatibility.dtype, device=compatibility.device),
            compatibility_pick_allpick,  # [n_heads, batch_size, n_pick, n_pick]
            -np.inf * torch.ones(self.n_heads, batch_size, n_pick, n_pick, dtype=compatibility.dtype, device=compatibility.device)
            ], 2).view(self.n_heads, batch_size, graph_size, n_pick)

        compatibility_additional_alldelivery = torch.cat([  # [n_heads, batch_size, graph_size, n_pick]
            -np.inf * torch.ones(self.n_heads, batch_size, 1, n_pick, dtype=compatibility.dtype, device=compatibility.device),
            compatibility_pick_alldelivery,  # [n_heads, batch_size, n_pick, n_pick]
            -np.inf * torch.ones(self.n_heads, batch_size, n_pick, n_pick, dtype=compatibility.dtype, device=compatibility.device)
        ], 2).view(self.n_heads, batch_size, graph_size, n_pick)
        # [n_heads, batch_size, n_query, graph_size+1+n_pick+n_pick]

        ##Delivery->
        compatibility_additional_pick = torch.cat([  # [n_heads, batch_size, graph_size, 1]
            -np.inf * torch.ones(self.n_heads, batch_size, 1, dtype=compatibility.dtype, device=compatibility.device),
            -np.inf * torch.ones(self.n_heads, batch_size, n_pick, dtype=compatibility.dtype, device=compatibility.device),
            compatibility_delivery_pick  # [n_heads, batch_size, n_pick]
            ], -1).view(self.n_heads, batch_size, graph_size, 1)        
        
        compatibility_additional_alldelivery2 = torch.cat([  # [n_heads, batch_size, graph_size, n_pick]
            -np.inf * torch.ones(self.n_heads, batch_size, 1, n_pick, dtype=compatibility.dtype, device=compatibility.device),
            -np.inf * torch.ones(self.n_heads, batch_size, n_pick, n_pick, dtype=compatibility.dtype, device=compatibility.device),
            compatibility_delivery_alldelivery  # [n_heads, batch_size, n_pick, n_pick]
            ], 2).view(self.n_heads, batch_size, graph_size, n_pick)    
    
        compatibility_additional_allpick2 = torch.cat([  # [n_heads, batch_size, graph_size, n_pick]
            -np.inf * torch.ones(self.n_heads, batch_size, 1, n_pick, dtype=compatibility.dtype, device=compatibility.device),
            -np.inf * torch.ones(self.n_heads, batch_size, n_pick, n_pick, dtype=compatibility.dtype, device=compatibility.device),
            compatibility_delivery_allpick  # [n_heads, batch_size, n_pick, n_pick]
        ], 2).view(self.n_heads, batch_size, graph_size, n_pick)     
    
        compatibility = torch.cat([compatibility, compatibility_additional_delivery, compatibility_additional_allpick, compatibility_additional_alldelivery,
                                   compatibility_additional_pick, compatibility_additional_alldelivery2, compatibility_additional_allpick2], dim=-1)


        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = torch.softmax(compatibility, dim=-1)  # [n_heads, batch_size, n_query, graph_size+1+n_pick*2] (graph_size include depot)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc
        # heads: [n_heads, batrch_size, n_query, val_size], attn????pick?deliver?attn
        heads = torch.matmul(attn[:, :, :, :graph_size], V)  # V: (self.n_heads, batch_size, graph_size, val_size)

        # heads??pick -> its delivery
        heads = heads + attn[:, :, :, graph_size].view(self.n_heads, batch_size, graph_size, 1) * V_additional_delivery  # V_addi:[n_heads, batch_size, graph_size, key_size]

        # heads??pick -> otherpick, V_allpick: # [n_heads, batch_size, n_pick, key_size]
        # heads: [n_heads, batch_size, graph_size, key_size]
        heads = heads + torch.matmul(attn[:, :, :, graph_size+1:graph_size+1+n_pick].view(self.n_heads, batch_size, graph_size, n_pick), V_allpick)

        # V_alldelivery: # (n_heads, batch_size, n_pick, key/val_size)
        heads = heads + torch.matmul(attn[:, :, :, graph_size+1+n_pick :graph_size+1 + 2 * n_pick].view(self.n_heads, batch_size, graph_size, n_pick), V_alldelivery)

        # delivery
        heads = heads + attn[:, :, :, graph_size+1 + 2 * n_pick].view(self.n_heads, batch_size, graph_size, 1) * V_additional_pick 
        
        heads = heads + torch.matmul(attn[:, :, :, graph_size+1 + 2 * n_pick+1:graph_size+1 + 3 * n_pick+1].view(self.n_heads, batch_size, graph_size, n_pick), V_alldelivery2)
        
        heads = heads + torch.matmul(attn[:, :, :, graph_size+1 + 3 * n_pick+1:].view(self.n_heads, batch_size, graph_size, n_pick), V_allpickup2)


        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class MultiHeadAttentionLayer(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        )


class GraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(GraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None

        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))

    def forward(self, x, mask=None):

        assert mask is None, "TODO mask not yet supported!"

        # Batch multiply to get initial embeddings of nodes
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x

        h = self.layers(h)

        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )
